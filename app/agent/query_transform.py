"""
查询意图分析、指代消解与多查询/HyDE改写器 (Query Transformation Engine)
职责：
1. 消除多轮对话中的代词指代与信息缺失 (Coreference Resolution)。
2. 实现多查询扩展 (Multi-Query Expansion)，多视角捕捉潜在证据。
3. 实现 HyDE (假设性文档嵌入) 技术，以假定答案的特征对齐真实文档向量分布。
4. 输出标准化的 QueryTransformationResult 实体契约，供给下游调度。
"""

import json
import logging
from typing import List, Dict, Any, Optional
from app.core.llm_provider import llm_provider
from app.models.schemas import QueryTransformationResult

logger = logging.getLogger(__name__)


class QueryTransformer:
    """查询改写：指代消解、多查询扩展、HyDE"""

    def __init__(self):
        self.llm = llm_provider

    # =========================================================================
    # 策略一：指代消解与独立问题重塑 (Disambiguation)
    # =========================================================================

    async def disambiguate(
        self,
        current_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        基于历史记忆消除代词（如“它”、“后者”、“第二条”），将其还原为脱离上下文也能看懂的独立Query
        """
        if not chat_history:
            return current_query.strip()

        # 取最近 3 轮对话作为历史背景
        recent_history = chat_history[-6:]
        history_text = "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in recent_history])

        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一名搜索查询重写专家。请根据提供的对话历史，将用户的最新问题改写为一个语义完整、"
                    "不依赖上下文、消除了所有代词（它、这个、后者等）的独立查询语句。\n"
                    "要求：\n"
                    "1. 保留用户的核心疑问，补充必要的主语和背景名词。\n"
                    "2. 如果最新问题本身已经是独立完整的，直接原样输出，不要画蛇添足。\n"
                    "3. 仅输出改写后的一句话，严禁附带任何解释或前缀。"
                )
            },
            {
                "role": "user",
                "content": f"【对话历史】:\n{history_text}\n\n【最新问题】: {current_query}\n\n【改写后的独立查询】:"
            }
        ]

        rewritten = await self.llm.chat(prompt, temperature=0.0)
        cleaned = rewritten.strip().replace('"', '').replace("'", "")
        logger.info(f"[QueryTransformer] 指代消解: '{current_query}' -> '{cleaned}'")
        return cleaned or current_query

    # =========================================================================
    # 策略二：多查询扩展 (Multi-Query Expansion)
    # =========================================================================

    async def generate_multi_queries(self, query: str, count: int = 3) -> List[str]:
        """
        从不同视角或同义表达生成多个搜索 Query，用于弥补单一表达词汇覆盖不足的缺陷
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    f"你是一名信息检索优化专家。请针对用户的问题，从不同措辞、技术术语或同义表达角度，"
                    f"生成 {count} 个具有互补性的搜索关键词或短语。\n"
                    "输出要求：严格输出为合法 JSON 字符串数组，例如：[\"查询1\", \"查询2\", \"查询3\"]，不要包含任何多余文字。"
                )
            },
            {
                "role": "user",
                "content": f"用户原始问题: {query}"
            }
        ]

        raw_resp = await self.llm.chat(prompt, temperature=0.3)
        try:
            # 提取 JSON 结构
            start_idx = raw_resp.find("[")
            end_idx = raw_resp.rfind("]")
            if start_idx != -1 and end_idx != -1:
                queries = json.loads(raw_resp[start_idx: end_idx + 1])
                if isinstance(queries, list) and queries:
                    return [str(q).strip() for q in queries[:count]]
        except Exception as e:
            logger.warning(f"[QueryTransformer] 多查询解析失败 ({e})，使用原始问题备选。")

        return [query]

    # =========================================================================
    # 策略三：HyDE 假设性文档生成 (Hypothetical Document Embeddings)
    # =========================================================================

    async def generate_hyde_passage(self, query: str) -> str:
        """
        假设性文档生成：让模型先脑补一段专业解答文本，用该段落的稠密向量去匹配真实知识库
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一名资深企业技术专家。请针对用户的疑问，写一段专业、客观、严谨的模拟解答段落。\n"
                    "要求：\n"
                    "1. 直接给出包含专业术语和逻辑推导的回答段落，约 150-250 字。\n"
                    "2. 即使你并不完全确定具体事实，也要采用真实技术文档的风格和词汇结构进行组织。\n"
                    "3. 仅输出模拟回答正文，不要输出任何寒暄或免责声明。"
                )
            },
            {
                "role": "user",
                "content": f"问题: {query}"
            }
        ]

        hyde_text = await self.llm.chat(prompt, temperature=0.1)
        logger.debug(f"[QueryTransformer] 生成 HyDE 假设性段落: {hyde_text[:60]}...")
        return hyde_text.strip()

    # =========================================================================
    # 总体门面调度入口 (Transform Pipeline)
    # =========================================================================

    async def transform(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        enable_multi_query: bool = False,
        enable_hyde: bool = False
    ) -> QueryTransformationResult:
        """
        统一查询重塑总入口
        """
        # 1. 优先执行代词消解与上下文对齐
        independent_query = await self.disambiguate(query, chat_history)

        rewritten_list = [independent_query]
        hyde_passage = None

        # 2. 按需执行多查询扩展
        if enable_multi_query:
            multi_queries = await self.generate_multi_queries(independent_query, count=3)
            rewritten_list.extend([q for q in multi_queries if q != independent_query])

        # 3. 按需生成 HyDE 模拟文档
        if enable_hyde:
            hyde_passage = await self.generate_hyde_passage(independent_query)

        return QueryTransformationResult(
            original_query=query,
            rewritten_queries=rewritten_list,
            hyde_passage=hyde_passage,
            is_complex=len(rewritten_list) > 2
        )


# 全局单例
query_transformer = QueryTransformer()
