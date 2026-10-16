"""
Cross-Encoder 交叉重排与母块回填引擎 (Reranker & Context Expansion)
职责：
1. 解决双塔向量检索只衡量“主题相关”而无法衡量“直接解答度”的深层缺陷。
2. 调用 Cross-Encoder 交叉注意力重排模型 (如 gte-rerank-v2)，对候选集做深度交叉打分。
3. 实施相关度硬阈值门控 (Threshold Filter)，彻底拦截垃圾噪音，防止下游幻觉。
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
    企业级 Cross-Encoder 交叉重排器
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

        # 提取候选文档的原文列表送给模型
        doc_texts = [item.content for item in candidates]

        # 1. 调用底层的重排接口 (或本地 Mock 模拟重排)
        raw_rerank_results = await llm_provider.rerank(
            query=query,
            documents=doc_texts,
            top_n=len(doc_texts)  # 先全量打分，后续结合阈值截取
        )

        final_items: List[SearchResultItem] = []

        # 2. 遍历重排打分结果并执行阈值过滤
        for rank, res in enumerate(raw_rerank_results, start=1):
            score = float(res["relevance_score"])

            # 门控拦截：低于最低阈值的候选直接淘汰，防范幻觉
            if score < self.min_score:
                logger.debug(f"[Reranker] 切片淘汰: 得分 {score} 低于设定门控阈值 {self.min_score}")
                continue

            orig_candidate = candidates[res["index"]]
            parent_id = orig_candidate.parent_id
            
            # 3. 执行母块回填 (Parent Context Expansion)
            parent_text = None
            if parent_id and parent_id in parent_map:
                parent_text = parent_map[parent_id].content

            # 组装最终重排实体
            reranked_item = SearchResultItem(
                chunk_id=orig_candidate.chunk_id,
                doc_id=orig_candidate.doc_id,
                doc_name=orig_candidate.doc_name,
                parent_id=parent_id,
                content=orig_candidate.content,
                parent_content=parent_text, # 注入母块大段上下文
                score=round(score, 4),
                source=RetrievalSource.RERANKED,
                rank=len(final_items) + 1,
                metadata={
                    **orig_candidate.metadata,
                    "rerank_score": round(score, 4),
                    "has_parent_context": bool(parent_text)
                }
            )
            final_items.append(reranked_item)

            if len(final_items) >= limit:
                break

        logger.info(f"[Reranker] 精排完成: 输入 {len(candidates)} 条候选，通过门控阈值并保留 Top-{len(final_items)} 条高质量结果")
        return final_items


# 全局单例重排器
cross_encoder_reranker = CrossEncoderReranker()
