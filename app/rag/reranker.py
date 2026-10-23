"""
Cross-Encoder 交叉重排与母块回填引擎 (Reranker & Context Expansion)
职责：
1. 解决双塔向量检索只衡量“主题相关”而无法衡量“直接解答度”的深层缺陷。
2. 调用 Cross-Encoder 交叉注意力重排模型 (如 gte-rerank-v2)，对候选集做深度交叉打分。
3. 相关度阈值过滤 (Threshold Filter)：过滤低分片段，减少无关资料进入上下文（阈值尚未标定，见 API-4）。
4. 完成“父子分块”的终极闭环：小块打分排序，胜出者自动回填母块完整上下文 (Parent Context Expansion)。
"""

import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.models.schemas import SearchResultItem, ParentChunk, RetrievalSource

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    交叉重排器（在线调用百炼 gte-rerank；离线时为词面打分桩，不是交叉编码器）
    具备“精准精排 + 门控截断 + 母块回填”三重能力
    """

    def __init__(
        self,
        top_n: Optional[int] = None,
        min_score_threshold: Optional[float] = None
    ):
        self.top_n = top_n or settings.reranker.TOP_N
        self.min_score = min_score_threshold if min_score_threshold is not None else settings.reranker.MIN_SCORE_THRESHOLD

    async def rerank(
        self,
        query: str,
        candidates: List[SearchResultItem],
        parent_docs_map: Optional[Dict[str, ParentChunk]] = None,
        top_n: Optional[int] = None
    ) -> List[SearchResultItem]:
        """
        对多路召回融合后的候选切片执行二次交叉精排
        :param query: 搜索提问
        :param candidates: 粗筛候选切片 (通常为 RRF 融合出的 Top-20~Top-50)
        :param parent_docs_map: 母块实体字典映射 {parent_id: ParentChunk}，用于回填
        :param top_n: 最终保留的精选条数
        :return: 经过深度交叉打分、截断与母块回填的最终片段列表
        """
        if not candidates:
            return []

        limit = top_n or self.top_n
        parent_map = parent_docs_map or {}

        doc_texts = [item.content for item in candidates]

        # 1. 调用重排接口（离线时为本地打分桩）。
        # API-3：重排服务出错（超时/限流/5xx）时不让整个问答失败，退回 RRF 融合排序，并在元数据里标明。
        rerank_failed = False
        try:
            raw_rerank_results = await llm_provider.rerank(query=query, documents=doc_texts, top_n=len(doc_texts))
        except Exception as e:
            logger.warning(f"[Reranker] 重排服务调用失败，退回 RRF 排序: {e}")
            rerank_failed = True
            raw_rerank_results = [
                {"index": i, "relevance_score": c.score} for i, c in enumerate(candidates)
            ]

        final_items: List[SearchResultItem] = []
        seen_parents = set()

        for res in raw_rerank_results:
            score = float(res["relevance_score"])
            # 2. 阈值过滤（RRF 回退时分数尺度不同，不做阈值过滤）
            if not rerank_failed and score < self.min_score:
                logger.debug(f"[Reranker] 淘汰: 得分 {score} 低于阈值 {self.min_score}")
                continue

            orig = candidates[res["index"]]
            parent_id = orig.parent_id

            # 3. DATA-4：同一母块只保留得分最高的子块。原实现 5 个结果可能全来自同一母块，
            #    母块回填后同一段文字在上下文里重复 5 次（实测 6139 字，几乎全是重复）。
            #    结果已按得分降序，第一次出现的就是该母块下得分最高的子块。
            dedup_key = parent_id or orig.chunk_id
            if dedup_key in seen_parents:
                continue
            seen_parents.add(dedup_key)

            parent_text = parent_map[parent_id].content if parent_id and parent_id in parent_map else None

            final_items.append(SearchResultItem(
                chunk_id=orig.chunk_id,
                doc_id=orig.doc_id,
                doc_name=orig.doc_name,
                parent_id=parent_id,
                content=orig.content,
                parent_content=parent_text,
                score=round(score, 4),
                source=RetrievalSource.HYBRID_RRF if rerank_failed else RetrievalSource.RERANKED,
                rank=len(final_items) + 1,
                metadata={
                    **orig.metadata,
                    "rerank_score": None if rerank_failed else round(score, 4),
                    "rerank_fallback": rerank_failed,
                    "has_parent_context": bool(parent_text),
                },
            ))
            if len(final_items) >= limit:
                break

        logger.info(f"[Reranker] 输入 {len(candidates)} 条候选，保留 {len(final_items)} 条"
                    f"（按母块去重{'，重排失败已回退 RRF' if rerank_failed else ''}）")
        return final_items


# 全局单例重排器
cross_encoder_reranker = CrossEncoderReranker()
