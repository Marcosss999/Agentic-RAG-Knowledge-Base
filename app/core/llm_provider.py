"""
大模型与多模态组件统一服务驱动 (LLM Provider Service)
职责：
1. 封装三大核心 AI 能力：对话大模型 (Chat LLM)、文本嵌入 (Embedding) 与 交叉重排 (Cross-Encoder Rerank)。
2. 统一底层调用协议：基于 AsyncOpenAI 异步客户端与 httpx，支持高并发无阻塞调用。
3. 原生支持流式传输 (SSE Generator)：提供逐字吐词的异步生成器，满足前端交互要求。
4. 内置离线防呆机制 (Mock Fallback)：未配置 API Key 时自动平滑回退至测试桩，确保初学者可在本地跑通业务逻辑闭环。
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from urllib.parse import urlparse
import httpx
import numpy as np
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

# 尝试导入分词工具，用于离线哈希特征向量的分词
try:
    import jieba
    JIEBA_LOADED = True
except ImportError:
    JIEBA_LOADED = False


def _compute_semantic_projection_embedding(text: str, dim: int = 1024) -> List[float]:
    """
    离线哈希特征向量（测试桩，Feature Hashing + 稀疏随机投影）

    做法：把文本切成词与字符二元组 (bi-gram)，每个 token 用哈希值作随机种子，在 dim 维里
    随机选 16 个维度加 ±1，最后 L2 归一化。两段文本共享的 token 越多，余弦相似度越高。

    它能做什么、不能做什么（RET-1）：
    - 只反映"词面重叠"，适合在没有 API Key 时把流程跑通；
    - 没有语义：同义词（如"向量检索"与"稠密召回"）的余弦接近 0。真正的语义相似来自训练过的
      Embedding 模型（如 text-embedding-v4），这一点不能混淆。
    - 原实现用 int.from_bytes(token, "little") & 0x7FFFFFFF 当种子，小端序下只取到前 4 个字节，
      前缀相同的词（retrieval/retrieve、embedding/embed）得到完全相同的向量；现改用 blake2b 摘要。
    """
    if not text.strip():
        return [0.0] * dim

    clean_text = text.lower().strip()
    tokens: List[str] = []

    if JIEBA_LOADED:
        tokens.extend(list(jieba.cut_for_search(clean_text)))
    else:
        tokens.extend(clean_text.split())

    # 提取字符级双字 (Bi-gram) 以捕获未登录词、专有名词和型号特征
    chars = [c for c in clean_text if c.strip()]
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])

    vec = np.zeros(dim, dtype=np.float32)

    for token in tokens:
        if not token.strip():
            continue
        # 基于 token 字符串生成稳定且确定性的非负整数种子
        h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
        token_rng = np.random.RandomState(h % (2**31 - 1))
        # 稀疏正负随机投影 (16 个随机维度激活)
        indices = token_rng.choice(dim, size=min(16, dim), replace=False)
        signs = token_rng.choice([-1.0, 1.0], size=min(16, dim))
        vec[indices] += signs

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec.tolist()


def _usable_key(key: Optional[str]) -> str:
    """过滤空值与 .env.example 中的占位值"""
    key = (key or "").strip()
    return "" if (not key or "your_api_key" in key) else key


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _resolve_key(own_key: str, own_base_url: str) -> Tuple[str, bool]:
    """返回 (实际使用的 Key, 是否复用了对话模型的 Key)。
    自己填了就用自己的；留空时，只有与对话模型是同一服务商（BASE_URL 域名相同）才复用。"""
    own = _usable_key(own_key)
    if own:
        return own, False
    llm_key = _usable_key(settings.llm.API_KEY)
    if llm_key and _host(own_base_url) and _host(own_base_url) == _host(settings.llm.BASE_URL):
        return llm_key, True
    return "", False


def _offline_rerank(query: str, documents: List[str], limit: int) -> List[Dict[str, Any]]:
    """离线重排打分（测试桩）：词级 Jaccard + 字符重合 + 哈希特征余弦的加权和。纯 CPU 计算。"""
    # 本地多信号词法与语义交叉深度精排算法
    scored = []
    q_clean = query.lower().strip()
    q_words = set(jieba.cut(q_clean)) if JIEBA_LOADED else set(q_clean.split())
    q_chars = set(q_clean)
    q_vec = np.array(_compute_semantic_projection_embedding(query, dim=512))

    for idx, doc in enumerate(documents):
        d_clean = doc.lower().strip()
        d_words = set(jieba.cut(d_clean)) if JIEBA_LOADED else set(d_clean.split())

        # 1. 词级 Jaccard 相似度 (权重 0.40)
        intersection_w = len(q_words & d_words)
        union_w = len(q_words | d_words) or 1
        jaccard_word = intersection_w / union_w

        # 2. 字符重合与实体包含度 (权重 0.25)
        char_overlap = sum(1 for c in q_chars if c in d_clean) / (len(q_chars) or 1)
        substring_bonus = 0.20 if query in doc else (0.10 if any(len(w) >= 3 and w in doc for w in q_words) else 0.0)

        # 3. 语义随机投影余弦相关度 (权重 0.35)
        d_vec = np.array(_compute_semantic_projection_embedding(doc[:400], dim=512))
        cosine_sim = max(0.0, float(np.dot(q_vec, d_vec)))

        composite_score = round(jaccard_word * 0.40 + (char_overlap + substring_bonus) * 0.25 + cosine_sim * 0.35, 4)
        final_score = min(0.98, max(0.10, composite_score))
        scored.append({"index": idx, "relevance_score": final_score, "text": doc})

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:limit]


class LLMProvider:
    """对话、向量、重排三类模型调用的统一入口（未配置 Key 时使用离线测试桩）"""

    def __init__(self):
        self.reload_config()

    def reload_config(self):
        """重新加载配置并按需更新客户端连接。

        API-6：Embedding / Reranker 的 Key 留空时，原实现只在进程启动那一刻用对话模型的 Key 兜底：
        - 运行中在网页上填写对话模型 Key，向量和重排并不会跟着"复用"（界面却写着"留空则自动复用"）；
        - 重启后又会无条件复用——当对话模型是 DeepSeek、向量模型是阿里云百炼时，等于拿 DeepSeek 的 Key
          去调百炼接口，必然 401，导致启动重建索引失败、问答报错。
        现在每次重载都动态解析，并且只在两者属于同一服务商（同一域名）时才复用。
        """
        self.llm_api_key = _usable_key(settings.llm.API_KEY)
        self.embed_api_key, self.embed_key_inherited = _resolve_key(
            settings.embedding.API_KEY, settings.embedding.BASE_URL)
        self.rerank_api_key, self.rerank_key_inherited = _resolve_key(
            settings.reranker.API_KEY, settings.reranker.BASE_URL)

        self.is_llm_configured = bool(self.llm_api_key)
        self.is_embed_configured = bool(self.embed_api_key)
        self.is_rerank_configured = bool(self.rerank_api_key)

        if self.is_llm_configured:
            self.chat_client = AsyncOpenAI(
                api_key=self.llm_api_key,
                base_url=settings.llm.BASE_URL,
                timeout=settings.llm.REQUEST_TIMEOUT,
                max_retries=settings.llm.MAX_RETRIES,  # SDK 内置指数退避重试（429/5xx/超时）
            )
        else:
            self.chat_client = None

        if self.is_embed_configured:
            self.embed_client = AsyncOpenAI(
                api_key=self.embed_api_key,
                base_url=settings.embedding.BASE_URL,
                timeout=settings.llm.REQUEST_TIMEOUT,
                max_retries=settings.llm.MAX_RETRIES,
            )
        else:
            self.embed_client = None

        logger.info(
            f"[LLMProvider] 配置已重载: LLM={self.is_llm_configured} ({settings.llm.DEFAULT_MODEL}), "
            f"Embedding={self.is_embed_configured} ({settings.embedding.MODEL_NAME}), "
            f"Reranker={self.is_rerank_configured} ({settings.reranker.MODEL_NAME})"
        )

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

    @property
    def embedding_signature(self) -> str:
        """当前向量空间的标识。不同模型产出的向量不在同一个空间里，不能混在一个索引中 (API-2)。"""
        if self.is_embed_configured:
            return f"api:{settings.embedding.BASE_URL}:{settings.embedding.MODEL_NAME}"
        # v2：RET-1 修改了哈希函数，旧版离线向量与新版不兼容，版本号变化会触发索引重建
        return f"offline-hash-v2:{settings.embedding.DIMENSION}"

    async def get_embeddings(self, texts: List[str], progress=None) -> List[List[float]]:
        """
        批量文本向量化
        :param texts: 待向量化的字符串列表
        :param progress: 可选回调 progress(done, total)，每完成一批调用一次（后台入库显示进度用）
        :return: 二维浮点数组，外层长度为 len(texts)，顺序与输入一致

        API-1：服务商对单次请求的条数有上限（百炼 text-embedding-v4 为 10 条），
        原实现把整篇文档的全部子块一次发出，超过 10 个子块的文档必然失败。
        现在按 BATCH_SIZE 分批，并用信号量限制同时在途的请求数，避免触发限流。
        DATA-7：在线模式先查本地缓存，只对未命中的文本调用接口，结果写回缓存。
        """
        if not texts:
            return []

        if not self.is_embed_configured:
            # 离线哈希特征向量是 CPU 计算，放到线程池，避免大文档阻塞事件循环 (CONC-2)
            dim = settings.embedding.DIMENSION
            vectors = await asyncio.to_thread(
                lambda: [_compute_semantic_projection_embedding(t, dim=dim) for t in texts])
            if progress:
                progress(len(texts), len(texts))
            return vectors

        from app.core.embedding_cache import get_cache
        cache = get_cache()
        signature = self.embedding_signature
        results: List[Optional[List[float]]] = [None] * len(texts)
        if cache is not None:
            for i, vec in (await asyncio.to_thread(cache.get_many, signature, texts)).items():
                results[i] = vec
        missing = [i for i, v in enumerate(results) if v is None]
        total, done = len(texts), len(texts) - len(missing)
        if progress:
            progress(done, total)

        batch_size = max(1, settings.embedding.BATCH_SIZE)
        semaphore = asyncio.Semaphore(max(1, settings.embedding.MAX_CONCURRENCY))
        batches = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]

        async def embed_batch(idx: List[int]) -> None:
            nonlocal done
            batch = [texts[i] for i in idx]
            async with semaphore:
                response = await self.embed_client.embeddings.create(
                    model=settings.embedding.MODEL_NAME, input=batch)
            # 按返回的 index 排序，保证向量与输入一一对应
            data = sorted(response.data, key=lambda d: getattr(d, "index", 0))
            if len(data) != len(batch):
                raise RuntimeError(f"Embedding 返回条数 {len(data)} 与请求条数 {len(batch)} 不一致")
            vectors = [d.embedding for d in data]
            for i, v in zip(idx, vectors):
                results[i] = v
            if cache is not None:
                await asyncio.to_thread(cache.put_many, signature, batch, vectors)
            done += len(idx)
            if progress:
                progress(done, total)

        await asyncio.gather(*[embed_batch(b) for b in batches])
        return results  # type: ignore[return-value]

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
        调用重排模型为候选片段打分（离线时为词面打分桩）
        :param query: 用户问题
        :param documents: 候选文档文本列表
        :param top_n: 保留最高分的前 N 个
        :return: 列表结构 [{'index': 原始索引, 'relevance_score': 浮点相关度, 'text': 文档原文}]
        """
        if not documents:
            return []

        limit = top_n or settings.reranker.TOP_N

        if not self.is_rerank_configured:
            # 离线打分是 CPU 计算，放到线程池执行 (CONC-2)
            return await asyncio.to_thread(_offline_rerank, query, documents, limit)

        # 调用百炼/DashScope 标准重排接口
        headers = {
            "Authorization": f"Bearer {self.rerank_api_key}",
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

        async with httpx.AsyncClient(timeout=settings.reranker.REQUEST_TIMEOUT) as client:
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

    # =========================================================================
    # 连通性测试探针 (Connectivity Health Check)
    # =========================================================================

    async def test_connectivity(self) -> Dict[str, Any]:
        """全面测试 Chat LLM、Embedding 与 Reranker 的真实连通性与网络延迟"""
        results = {}

        # 1. 测试 Chat LLM
        if self.is_llm_configured:
            t0 = time.time()
            try:
                resp = await self.chat_client.chat.completions.create(
                    model=settings.llm.DEFAULT_MODEL,
                    messages=[{"role": "user", "content": "你好"}],
                    max_tokens=5
                )
                cost_ms = round((time.time() - t0) * 1000, 1)
                results["llm"] = {
                    "status": "connected",
                    "latency_ms": cost_ms,
                    "model": settings.llm.DEFAULT_MODEL,
                    "message": "大模型网关连通正常"
                }
            except Exception as e:
                results["llm"] = {
                    "status": "error",
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "model": settings.llm.DEFAULT_MODEL,
                    "error": str(e)
                }
        else:
            results["llm"] = {
                "status": "mock",
                "model": settings.llm.DEFAULT_MODEL,
                "message": "当前处于离线测试桩模式 (未填 API Key)"
            }

        # 2. 测试 Embedding
        if self.is_embed_configured:
            t0 = time.time()
            try:
                resp = await self.embed_client.embeddings.create(
                    model=settings.embedding.MODEL_NAME,
                    input=["ping"]
                )
                cost_ms = round((time.time() - t0) * 1000, 1)
                results["embedding"] = {
                    "status": "connected",
                    "latency_ms": cost_ms,
                    "model": settings.embedding.MODEL_NAME,
                    "dim": len(resp.data[0].embedding),
                    "message": "向量模型连通正常"
                }
            except Exception as e:
                results["embedding"] = {
                    "status": "error",
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "model": settings.embedding.MODEL_NAME,
                    "error": str(e)
                }
        else:
            results["embedding"] = {
                "status": "local_engine",
                "model": f"offline-hash-{settings.embedding.DIMENSION}",
                "message": "未配置向量模型 Key，使用离线哈希特征向量（测试桩，只反映词面重叠，没有语义）"
            }

        # 3. 测试 Reranker
        if self.is_rerank_configured:
            t0 = time.time()
            try:
                headers = {
                    "Authorization": f"Bearer {self.rerank_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": settings.reranker.MODEL_NAME,
                    "input": {"query": "测试", "documents": ["测试文档A", "测试文档B"]},
                    "parameters": {"top_n": 2, "return_documents": False}
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(settings.reranker.BASE_URL, headers=headers, json=payload)
                    r.raise_for_status()
                cost_ms = round((time.time() - t0) * 1000, 1)
                results["reranker"] = {
                    "status": "connected",
                    "latency_ms": cost_ms,
                    "model": settings.reranker.MODEL_NAME,
                    "message": "重排模型服务连通正常"
                }
            except Exception as e:
                results["reranker"] = {
                    "status": "error",
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "model": settings.reranker.MODEL_NAME,
                    "error": str(e)
                }
        else:
            results["reranker"] = {
                "status": "local_engine",
                "model": "offline-lexical-scorer",
                "message": "未配置重排模型 Key，使用离线词面打分（测试桩，不是交叉编码器）"
            }

        return results


# 全局单例对象
llm_provider = LLMProvider()
