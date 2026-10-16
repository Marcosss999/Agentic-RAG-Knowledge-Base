"""
大模型与多模态组件统一服务驱动 (LLM Provider Service)
职责：
1. 封装三大核心 AI 能力：对话大模型 (Chat LLM)、文本嵌入 (Embedding) 与 交叉重排 (Cross-Encoder Rerank)。
2. 统一底层调用协议：基于 AsyncOpenAI 异步客户端与 httpx，支持高并发无阻塞调用。
3. 原生支持流式传输 (SSE Generator)：提供逐字吐词的异步生成器，满足前端交互要求。
4. 内置离线防呆机制 (Mock Fallback)：未配置 API Key 时自动平滑回退至测试桩，确保初学者可在本地跑通业务逻辑闭环。
"""

import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider:
    """企业级统一大模型服务门面"""

    def __init__(self):
        # 1. 检查是否为有效配置（未填写有效Key则标记为演示模式）
        self.is_llm_configured = bool(settings.llm.API_KEY and "your_api_key" not in settings.llm.API_KEY)
        self.is_embed_configured = bool(settings.embedding.API_KEY and "your_api_key" not in settings.embedding.API_KEY)
        self.is_rerank_configured = bool(settings.reranker.API_KEY and "your_api_key" not in settings.reranker.API_KEY)

        # 2. 初始化 OpenAI 兼容异步客户端
        if self.is_llm_configured:
            self.chat_client = AsyncOpenAI(
                api_key=settings.llm.API_KEY,
                base_url=settings.llm.BASE_URL,
                timeout=settings.llm.REQUEST_TIMEOUT
            )
        else:
            self.chat_client = None

        if self.is_embed_configured:
            self.embed_client = AsyncOpenAI(
                api_key=settings.embedding.API_KEY,
                base_url=settings.embedding.BASE_URL,
                timeout=settings.llm.REQUEST_TIMEOUT
            )
        else:
            self.embed_client = None

    # =========================================================================
    # 能力一：对话与文本生成 (Chat Completion & Streaming)
    # =========================================================================

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        一次性同步非流式对话
        :param messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        :return: 模型完整回答文本
        """
        if not self.is_llm_configured:
            logger.warning("[LLMProvider] 未检测到有效 LLM_API_KEY，启用本地模拟输出。")
            return f"【本地Mock响应】已收到对系统提问的分析请求。这是一条模拟的测试回复，当前查询消息量为 {len(messages)} 条。"

        response = await self.chat_client.chat.completions.create(
            model=model or settings.llm.DEFAULT_MODEL,
            messages=messages,
            temperature=temperature if temperature is not None else settings.llm.TEMPERATURE,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        异步流式对话生成器 (用于 SSE 前端实时逐字渲染)
        :yield: 每一个 token 增量片段
        """
        if not self.is_llm_configured:
            logger.warning("[LLMProvider] 未检测到有效 LLM_API_KEY，启用流式模拟输出。")
            mock_text = "【本地Mock流式】正在根据知识库进行自省检索... 答案已生成完成。"
            for char in mock_text:
                yield char
            return

        stream = await self.chat_client.chat.completions.create(
            model=model or settings.llm.DEFAULT_MODEL,
            messages=messages,
            temperature=temperature if temperature is not None else settings.llm.TEMPERATURE,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # =========================================================================
    # 能力二：文本向量化 (Text Embedding)
    # =========================================================================

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化
        :param texts: 待向量化的字符串列表
        :return: 二维浮点数组，外层长度为 len(texts)，内层维度为 DIMENSION (如 1024)
        """
        if not texts:
            return []

        if not self.is_embed_configured:
            logger.warning("[LLMProvider] 未检测到有效 EMBEDDING_API_KEY，生成确定性伪随机向量。")
            # 基于文本 hash 生成伪随机浮点向量，确保相同文本在测试时余弦距离为 1
            import hashlib
            results = []
            dim = settings.embedding.DIMENSION
            for text in texts:
                h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
                # 生成固定长度的归一化浮点特征
                vec = [((h >> (i % 32)) & 0xFF) / 255.0 for i in range(dim)]
                # L2 归一化
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                results.append([x / norm for x in vec])
            return results

        # 线上真实调用
        response = await self.embed_client.embeddings.create(
            model=settings.embedding.MODEL_NAME,
            input=texts
        )
        return [item.embedding for item in response.data]

    async def get_embedding(self, text: str) -> List[float]:
        """单条文本向量化的便捷方法"""
        res = await self.get_embeddings([text])
        return res[0]

    # =========================================================================
    # 能力三：交叉重排 (Cross-Encoder Rerank)
    # =========================================================================

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        利用交叉编码模型对候选片段进行深层注意力打分
        :param query: 用户问题
        :param documents: 候选文档文本列表
        :param top_n: 保留最高分的前 N 个
        :return: 列表结构 [{'index': 原始索引, 'relevance_score': 浮点相关度, 'text': 文档原文}]
        """
        if not documents:
            return []

        limit = top_n or settings.reranker.TOP_N

        if not self.is_rerank_configured:
            logger.warning("[LLMProvider] 未检测到有效 RERANKER_API_KEY，启用简易词频重排模拟。")
            # 简单按照包含 query 字符的比例进行降序模拟
            scored = []
            q_chars = set(query)
            for idx, doc in enumerate(documents):
                match_cnt = sum(1 for c in q_chars if c in doc)
                fake_score = round(0.2 + (match_cnt / (len(q_chars) or 1)) * 0.7, 4)
                scored.append({"index": idx, "relevance_score": fake_score, "text": doc})
            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scored[:limit]

        # 调用百炼/DashScope 标准重排接口
        headers = {
            "Authorization": f"Bearer {settings.reranker.API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.reranker.MODEL_NAME,
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "top_n": limit,
                "return_documents": False
            }
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(settings.reranker.BASE_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("output", {}).get("results", [])

            reranked = []
            for item in raw_results:
                orig_idx = item["index"]
                score = item["relevance_score"]
                reranked.append({
                    "index": orig_idx,
                    "relevance_score": score,
                    "text": documents[orig_idx]
                })
            return reranked


# 全局单例对象
llm_provider = LLMProvider()
