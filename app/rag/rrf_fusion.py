"""
RRF 互逆排名融合算法模块 (Reciprocal Rank Fusion)
职责：
1. 解决双路检索（向量检索 Cosine 与 BM25 词频得分）分数量纲完全不一致的物理冲突。
2. 摆脱绝对分值的误导，基于候选切片在各路检索中的“名次倒数”进行公平、鲁棒的无量纲融合。
3. 支持动态权重配比，输出统一规范的 Top-K 融合结果候选池，传递给后续重排器。
"""

import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.models.schemas import SearchResultItem, RetrievalSource

logger = logging.getLogger(__name__)


class RRFFusion:
    """
    工业级 RRF 倒数名次互逆融合器
    公式：RRF_Score(d) = SUM_m [ weight_m / (k + rank_m(d)) ]
    """

    def __init__(self, k: int = None):
        # 默认采用行业黄金平滑常数 60 (源自 Cormack et al. 经典论文)
        self.k = k or settings.rag.RRF_K

    def fuse(
        self,
        ranked_lists: List[List[SearchResultItem]],
        weights: Optional[List[float]] = None,
        top_n: int = None
    ) -> List[SearchResultItem]:
        """
        执行多路候选结果的名次融合
        :param ranked_lists: 多路检索排序结果列表 [dense_results, sparse_results, ...]
        :param weights: 各路检索源的加权系数 (如 [1.0, 0.8])，默认均为 1.0
        :param top_n: 最终融合后保留的最优候选数量
        :return: 按 RRF 综合得分降序排列的统一结果列表
        """
        limit = top_n or settings.rag.DENSE_TOP_K
        if not ranked_lists:
            return []

        # 默认权重平权为 1.0
        if not weights or len(weights) != len(ranked_lists):
            weights = [1.0] * len(ranked_lists)

        # 聚合每个文档的 RRF 累计得分
        doc_scores: Dict[str, float] = {}
        # 记录切片实体对象与各路原始名次信息
        doc_details: Dict[str, Dict[str, Any]] = {}

        for list_idx, single_list in enumerate(ranked_lists):
            w = weights[list_idx]
            for rank, item in enumerate(single_list, start=1):
                cid = item.chunk_id

                if cid not in doc_details:
                    doc_details[cid] = {
                        "item": item,
                        "ranks": {},
                        "raw_scores": {}
                    }

                # 记录在当前路检索中的排名与原始分数
                doc_details[cid]["ranks"][item.source.value] = rank
                doc_details[cid]["raw_scores"][item.source.value] = item.score

                # 累加 RRF 得分
                rank_contribution = w / (self.k + rank)
                doc_scores[cid] = doc_scores.get(cid, 0.0) + rank_contribution

        # 按照融合总得分降序排序
        sorted_candidates = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        fused_results: List[SearchResultItem] = []
        for rank, (cid, final_rrf_score) in enumerate(sorted_candidates[:limit], start=1):
            detail = doc_details[cid]
            orig_item: SearchResultItem = detail["item"]

            merged_item = SearchResultItem(
                chunk_id=orig_item.chunk_id,
                doc_id=orig_item.doc_id,
                doc_name=orig_item.doc_name,
                parent_id=orig_item.parent_id,
                content=orig_item.content,
                parent_content=orig_item.parent_content,
                score=round(final_rrf_score, 5),
                source=RetrievalSource.HYBRID_RRF,
                rank=rank,
                metadata={
                    **orig_item.metadata,
                    "rrf_score": round(final_rrf_score, 5),
                    "original_ranks": detail["ranks"],
                    "original_raw_scores": detail["raw_scores"]
                }
            )
            fused_results.append(merged_item)

        logger.info(f"[RRFFusion] 成功融合 {len(ranked_lists)} 路检索源，产生 Top-{len(fused_results)} 候选集")
        return fused_results


# 全局单例
rrf_fusion = RRFFusion()
