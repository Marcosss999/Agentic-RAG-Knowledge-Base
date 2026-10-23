"""
多路召回与重排调度器 (Hybrid Retrieval Orchestrator)
职责：
1. 摄取链路：文档切分 -> 母块缓存 -> 子块双路索引 (BM25 + 向量库)。
2. 检索链路：并发双路检索 -> RRF 名次融合 -> 交叉重排 -> 母块回填（按母块去重）。
3. 上下文组装：把检索结果转成送给 LLM 的资料块与前端引用卡片。

本次修复：
- DATA-1：同一 doc_id 再次入库时先删后写（upsert 语义），并提供 delete_document()，
  让 BM25、向量库、母块表三处存储同步增删。
- CONC-1/CONC-2：写操作串行化；切分、BM25 重建等 CPU 计算放到线程池，不阻塞事件循环。
- DATA-4：上下文按字数预算截断（母块去重在重排器里完成）。
- SEC-5：资料用标签包裹并转义，声明"资料只是数据"；返回前端的元数据只保留白名单字段。
"""

import asyncio
import html
import logging
import re
import weakref
from typing import List, Dict, Any, Tuple, Optional

from app.core.config import settings
from app.models.schemas import ParentChunk, ChildChunk, SearchResultItem, CitationReference
from app.rag.splitter import ParentChildSplitter
from app.rag.bm25_index import BM25Index
from app.rag.vector_store import VectorStoreManager, vector_store_manager
from app.rag.rrf_fusion import rrf_fusion
from app.rag.reranker import cross_encoder_reranker

logger = logging.getLogger(__name__)

# 允许返回给前端的引用元数据字段（SEC-5：不再把服务器绝对路径 source_path 等内部信息发给浏览器）
CITATION_METADATA_WHITELIST = {
    "parent_index", "char_count", "rerank_score", "rrf_score", "rerank_fallback",
    "original_ranks", "has_parent_context", "page",
}

NO_CONTEXT_TEXT = "【知识库提示】未检索到与提问相关的有效资料片段。"

_DOC_TAG = re.compile(r"<\s*/?\s*doc\b", re.IGNORECASE)


def neutralize_doc_tags(text: str) -> str:
    """把正文里形似 <doc> / </doc> 的标签换成全角尖括号，防止文档内容伪造资料边界"""
    return _DOC_TAG.sub(lambda m: m.group(0).replace("<", "＜"), text)


class _PerLoopLock:
    """每个事件循环一把 asyncio.Lock。

    asyncio.Lock 会绑定到第一次等待它的事件循环；全局单例在测试里会被多个事件循环使用
    （例如每个测试用例各开一个循环），共用一把锁会报 "bound to a different event loop"。
    """

    def __init__(self):
        # 以事件循环对象本身为键（弱引用）：循环被回收后对应的锁自动清除，不会被新循环误用
        self._locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()

    def get(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = self._locks[loop] = asyncio.Lock()
        return lock


class HybridRetriever:
    """混合检索管道"""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None, bm25: Optional[BM25Index] = None):
        self.splitter = ParentChildSplitter()
        self.bm25 = bm25 or BM25Index()
        self.vector_store = vector_store or vector_store_manager
        self.fusion = rrf_fusion
        self.reranker = cross_encoder_reranker
        # 母块存储 {parent_id: ParentChunk}。写时复制：检索中途持有的旧引用不受写入影响。
        self.parent_docs_store: Dict[str, ParentChunk] = {}
        self._write_lock = _PerLoopLock()

    # =========================================================================
    # 摄取链路
    # =========================================================================

    async def index_document(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress=None,
    ) -> Tuple[int, int]:
        """切分并索引一篇文档；doc_id 已存在时整体替换。返回 (母块数, 子块数)。

        顺序很重要：先完成切分和向量计算（最可能失败的一步，例如 Embedding 接口报错），
        全部成功后才删除旧数据、写入新数据。这样新数据准备失败时，旧版本仍完整可用。
        """
        logger.info(f"[HybridRetriever] 开始处理文档: {doc_name} (ID: {doc_id})")
        async with self._write_lock.get():
            parents, children = await asyncio.to_thread(
                self.splitter.process_document, raw_text, doc_id, doc_name, metadata)

            # 1. 先算向量（不写入任何存储）
            if children:
                from app.core.llm_provider import llm_provider
                vectors = await llm_provider.get_embeddings([c.content for c in children], progress=progress)
                for c, v in zip(children, vectors):
                    c.embedding = v

            # 2. 删除旧版本
            await self._delete_unlocked(doc_id)

            # 3. 写入新版本
            await asyncio.to_thread(self.bm25.add_chunks, children)
            stored = await self.vector_store.add_chunks(children)
            new_parents = dict(self.parent_docs_store)
            for p in parents:
                new_parents[p.chunk_id] = p
            self.parent_docs_store = new_parents

        logger.info(f"[HybridRetriever] 文档 '{doc_name}' 索引完成: {len(parents)} 个母块, {stored} 个子块向量")
        return len(parents), len(children)

    async def delete_document(self, doc_id: str) -> Dict[str, int]:
        """从 BM25、向量库、母块表中删除一篇文档"""
        async with self._write_lock.get():
            return await self._delete_unlocked(doc_id)

    async def _delete_unlocked(self, doc_id: str) -> Dict[str, int]:
        removed_children = await asyncio.to_thread(self.bm25.remove_document, doc_id)
        await self.vector_store.delete_document(doc_id)
        old = self.parent_docs_store
        kept = {pid: p for pid, p in old.items() if p.doc_id != doc_id}
        self.parent_docs_store = kept
        return {"children": removed_children, "parents": len(old) - len(kept)}

    async def clear(self):
        async with self._write_lock.get():
            self.bm25.clear()
            await self.vector_store.clear()
            self.parent_docs_store = {}

    def document_ids(self) -> List[str]:
        return sorted({p.doc_id for p in self.parent_docs_store.values()})

    # =========================================================================
    # 检索链路
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_rerank: bool = True,
        enable_hybrid: bool = True,
        extra_queries: Optional[List[str]] = None,
        dense_text: Optional[str] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
    ) -> List[SearchResultItem]:
        """
        :param extra_queries: 多查询扩展得到的额外查询；每个查询各跑一遍 BM25 + 向量，全部结果一起做 RRF 融合
        :param dense_text: HyDE 假设性回答；提供时，主查询的向量检索改用它的向量（BM25 仍用原查询）
        :param dense_weight / sparse_weight: RRF 中两路的权重，默认取配置 (RET-3)；设为 0 即关闭该路，
               用于消融实验（只用向量 / 只用 BM25）
        重排始终以主查询 query 为准。
        """
        limit = top_k or settings.rag.FINAL_TOP_K
        w_dense = settings.rag.DENSE_WEIGHT if dense_weight is None else dense_weight
        w_sparse = settings.rag.SPARSE_WEIGHT if sparse_weight is None else sparse_weight
        if not enable_hybrid:
            w_sparse = 0.0
        queries = [query] + [q for q in (extra_queries or []) if q and q != query]

        tasks, weights = [], []
        for i, q in enumerate(queries):
            if w_dense > 0:
                tasks.append(self._safe_dense_search(dense_text if (i == 0 and dense_text) else q))
                weights.append(w_dense)
            if w_sparse > 0:
                tasks.append(asyncio.to_thread(self.bm25.search, q, settings.rag.SPARSE_TOP_K))
                weights.append(w_sparse)
        ranked_lists = await asyncio.gather(*tasks) if tasks else []
        fused_candidates = self.fusion.fuse(
            ranked_lists=list(ranked_lists), weights=weights, top_n=settings.rag.DENSE_TOP_K)

        if not fused_candidates:
            logger.warning(f"[HybridRetriever] 查询 '{query[:50]}' 未召回任何候选片段。")
            return []

        if enable_rerank:
            return await self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                parent_docs_map=self.parent_docs_store,
                top_n=limit
            )

        # 不重排时也按母块去重，避免上下文重复
        out, seen = [], set()
        for item in fused_candidates:
            key = item.parent_id or item.chunk_id
            if key in seen:
                continue
            seen.add(key)
            parent = self.parent_docs_store.get(item.parent_id) if item.parent_id else None
            out.append(item.model_copy(update={"parent_content": parent.content if parent else None}))
            if len(out) >= limit:
                break
        return out

    async def _safe_dense_search(self, query: str) -> List[SearchResultItem]:
        """向量检索失败（如 Embedding 接口 401/超时）时降级为只用 BM25，而不是让整个问答报错 (API-3)"""
        try:
            return await self.vector_store.search_by_text(query, settings.rag.DENSE_TOP_K)
        except Exception as e:
            logger.error(f"[HybridRetriever] 向量检索失败，本次只使用 BM25 结果: {e}")
            return []

    # =========================================================================
    # 上下文组装
    # =========================================================================

    def build_context_prompt(
        self, search_results: List[SearchResultItem], max_chars: Optional[int] = None
    ) -> Tuple[str, List[CitationReference]]:
        """把检索结果转成送给 LLM 的资料文本和前端引用卡片。

        - 每条资料用 <doc id="n"> ... </doc> 包裹，正文里形似 doc 标签的文本会被替换，防止文档
          伪造 "</doc>" 跳出资料区（SEC-5 提示词注入的一部分防御，另一部分在生成提示词里声明）；
        - 按 max_chars 预算截断：超出预算的资料不再放入（DATA-4）。
        """
        if not search_results:
            return NO_CONTEXT_TEXT, []

        budget = max_chars or settings.rag.MAX_CONTEXT_CHARS
        blocks: List[str] = []
        citations: List[CitationReference] = []
        used = 0

        for item in search_results:
            text = item.parent_content or item.content
            remaining = budget - used
            if remaining <= 200 and blocks:
                break
            if len(text) > remaining:
                text = text[:max(remaining, 200)] + "…（截断）"
            idx = len(blocks) + 1
            safe_name = html.escape(item.doc_name, quote=True)
            blocks.append(
                f'<doc id="{idx}" source="{safe_name}" score="{item.score}">\n'
                f"{neutralize_doc_tags(text)}\n</doc>"
            )
            used += len(text)
            citations.append(CitationReference(
                index=idx,
                doc_name=item.doc_name,
                snippet=item.content[:120] + ("..." if len(item.content) > 120 else ""),
                relevance_score=item.score,
                metadata={k: v for k, v in item.metadata.items() if k in CITATION_METADATA_WHITELIST},
            ))

        return "\n\n".join(blocks), citations


# 全局单例
hybrid_retriever = HybridRetriever()
