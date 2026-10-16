"""
多路召回与重排协同总调度器 (Hybrid Retrieval Orchestrator)
职责：
1. 贯穿数据摄取链路：文档切分 -> 母块持久化缓存 -> 子块双向索引 (BM25 + Qdrant 向量库)。
2. 贯穿查询检索链路：并发双路检索 (asyncio.gather) -> RRF 名次融合 -> Cross-Encoder 精排 -> 母块回填。
3. 提供统一的高级检索门面与提示词上下文组装器 (Context Assembler)。
"""

import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings
from app.models.schemas import ParentChunk, ChildChunk, SearchResultItem, CitationReference
from app.rag.splitter import ParentChildSplitter
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import vector_store_manager
from app.rag.rrf_fusion import rrf_fusion
from app.rag.reranker import cross_encoder_reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """企业级端到端混合检索管道调度器"""

    def __init__(self):
        self.splitter = ParentChildSplitter()
        self.bm25 = BM25Index()
        self.vector_store = vector_store_manager
        self.fusion = rrf_fusion
        self.reranker = cross_encoder_reranker

        # 母块本地实体存储字典 {parent_id: ParentChunk}
        # (实际生产中可平滑接入 PostgreSQL / MongoDB / Redis)
        self.parent_docs_store: Dict[str, ParentChunk] = {}

    # =========================================================================
    # 链路一：文档解析与多级索引建立 (Indexing Pipeline)
    # =========================================================================

    async def index_document(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int]:
        """
        全量摄取单篇文档建立父子双层与双路检索索引
        :return: (母块数量, 子块数量)
        """
        logger.info(f"[HybridRetriever] 开始处理文档: {doc_name} (ID: {doc_id})")

        # 1. 执行父子切分
        parents, children = self.splitter.process_document(
            raw_text=raw_text,
            doc_id=doc_id,
            doc_name=doc_name,
            metadata=metadata
        )

        # 2. 缓存母块实体供后续回填
        for p in parents:
            self.parent_docs_store[p.chunk_id] = p

        # 3. 灌入 BM25 稀疏倒排索引
        self.bm25.build_index(children)

        # 4. 批量计算 Embedding 并存入 Qdrant 向量库
        stored_vectors_count = await self.vector_store.add_chunks(children)

        logger.info(f"[HybridRetriever] 文档 '{doc_name}' 索引构建完成: 产生 {len(parents)} 个母块, {stored_vectors_count} 个子块向量")
        return len(parents), len(children)

    # =========================================================================
    # 链路二：端到端多路召回与重排 (Retrieval Pipeline)
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_rerank: bool = True,
        enable_hybrid: bool = True
    ) -> List[SearchResultItem]:
        """
        全流程混合检索入口
        """
        limit = top_k or settings.rag.FINAL_TOP_K

        # 1. 并发执行多路召回 (Concurrent Retrieval)
        if enable_hybrid:
            # 包装为异步协程并发调度
            bm25_task = asyncio.to_thread(self.bm25.search, query, settings.rag.SPARSE_TOP_K)
            dense_task = self.vector_store.search_by_text(query, settings.rag.DENSE_TOP_K)

            # 同时等待两路结果
            sparse_results, dense_results = await asyncio.gather(bm25_task, dense_task)

            # 2. 使用 RRF 算法进行无量纲名次倒数融合
            fused_candidates = self.fusion.fuse(
                ranked_lists=[dense_results, sparse_results],
                weights=[1.0, 0.8], # 向量路权重稍高，BM25补充专有名词
                top_n=settings.rag.DENSE_TOP_K
            )
        else:
            # 降级：仅走纯向量检索
            fused_candidates = await self.vector_store.search_by_text(query, settings.rag.DENSE_TOP_K)

        if not fused_candidates:
            logger.warning(f"[HybridRetriever] 针对查询 '{query}' 未召回到任何候选片段。")
            return []

        # 3. 执行 Cross-Encoder 深度交叉精排与母块回填
        if enable_rerank:
            final_results = await self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                parent_docs_map=self.parent_docs_store,
                top_n=limit
            )
        else:
            final_results = fused_candidates[:limit]

        return final_results

    # =========================================================================
    # 链路三：上下文装配器 (Context & Citation Assembler)
    # =========================================================================

    def build_context_prompt(self, search_results: List[SearchResultItem]) -> Tuple[str, List[CitationReference]]:
        """
        将检索出的候选片段转化为规范的上下文提示词字符串与前端引用卡片
        :return: (组装好的Context字符串, Citation引用列表)
        """
        if not search_results:
            return "【知识库提示】未检索到与提问相关的有效资料片段。", []

        context_blocks: List[str] = []
        citations: List[CitationReference] = []

        for idx, item in enumerate(search_results, start=1):
            # 优先采用母块大段上下文 (若有)，否则使用子块
            text_for_llm = item.parent_content or item.content
            
            block = (
                f"--- 证据资料 [{idx}] (来源文档: {item.doc_name}, 置信度: {item.score}) ---\n"
                f"{text_for_llm}\n"
            )
            context_blocks.append(block)

            # 记录证据卡片 (摘取前 120 字高亮摘要)
            citations.append(CitationReference(
                index=idx,
                doc_name=item.doc_name,
                snippet=item.content[:120] + ("..." if len(item.content) > 120 else ""),
                relevance_score=item.score,
                metadata=item.metadata
            ))

        formatted_context = "\n".join(context_blocks)
        return formatted_context, citations


# 全局单例
hybrid_retriever = HybridRetriever()
