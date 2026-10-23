"""
密集向量存储与检索 (Dense Vector Store)
职责：
1. 存储子块（Child Chunk）向量，提供余弦相似度 Top-K 检索。
2. BaseVectorStore 统一接口，屏蔽 Qdrant 与纯内存实现的差异。
3. Qdrant 支持独立服务模式与嵌入式 ':memory:' 模式；未安装 qdrant-client 时使用纯内存实现。

本次修复：
- RET-2：类型标注原本在类定义时求值，没装 qdrant-client 时整个模块 NameError，
  "降级到纯内存"永远到不了。加 `from __future__ import annotations` 让标注延迟求值。
- CONC-2：qdrant-client 是同步客户端，原来直接在 async 函数里调用会卡住事件循环。
  现在通过 `asyncio.to_thread` 放到线程池执行；嵌入式 ':memory:' 模式内部是普通 Python 字典，
  不保证线程安全，因此额外加一把锁串行化访问。
- DATA-1：新增按 doc_id 删除，并由管理器在重复上传时先删后写。
- API-2：集合维度不再写死，以第一次写入的向量维度为准；向量模型变化时由上层调用 reset() 重建。
"""

from __future__ import annotations

import asyncio
import threading
import uuid
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.models.schemas import ChildChunk, SearchResultItem, RetrievalSource

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("[VectorStore] 未检测到 qdrant-client，将使用纯内存向量存储。")


class BaseVectorStore(ABC):
    """向量库统一抽象基类（所有方法要求 chunk.embedding 已计算好）"""

    @abstractmethod
    async def add_chunks(self, chunks: List[ChildChunk]) -> int: ...

    @abstractmethod
    async def search_by_vector(self, query_vector: List[float], top_k: int = None) -> List[SearchResultItem]: ...

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> bool: ...

    @abstractmethod
    async def clear(self): ...

    @abstractmethod
    async def count(self) -> int: ...


def _to_item(rank: int, score: float, chunk_id: str, doc_id: str, doc_name: str,
             parent_id: Optional[str], content: str, metadata: Dict[str, Any]) -> SearchResultItem:
    s = round(float(score), 4)
    return SearchResultItem(
        chunk_id=chunk_id, doc_id=doc_id, doc_name=doc_name, parent_id=parent_id,
        content=content, score=s, source=RetrievalSource.DENSE_VECTOR, rank=rank,
        metadata={**(metadata or {}), "raw_cosine_score": s},
    )


class QdrantVectorStore(BaseVectorStore):
    """基于 Qdrant 的向量存储"""

    def __init__(self, collection_name: str = None, force_in_memory: bool = False):
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client 未安装，无法初始化 QdrantVectorStore")
        self.collection_name = collection_name or settings.storage.COLLECTION_NAME
        self.dimension: Optional[int] = None
        self.client, self.is_local = self._init_client(force_in_memory)
        # 嵌入式模式不是线程安全的；独立服务模式走 HTTP，可以并发
        self._lock = threading.Lock() if self.is_local else None
        self._detect_existing_dimension()

    def _init_client(self, force_in_memory: bool):
        if force_in_memory or settings.storage.VECTOR_DB_TYPE == "memory":
            logger.info("[Qdrant] 使用嵌入式内存模式 (:memory:)")
            return QdrantClient(":memory:"), True
        try:
            client = QdrantClient(host=settings.storage.QDRANT_HOST, port=settings.storage.QDRANT_PORT, timeout=2.0)
            client.get_collections()
            logger.info(f"[Qdrant] 已连接 Qdrant 服务: {settings.storage.QDRANT_HOST}:{settings.storage.QDRANT_PORT}")
            return client, False
        except Exception as e:
            # API-5：若系统开了代理，访问 localhost 也可能被代理拦截（返回 502），需设置 NO_PROXY=localhost,127.0.0.1
            logger.warning(f"[Qdrant] 无法连接 Qdrant 服务 ({e})，降级为嵌入式内存模式 (:memory:)。"
                           f"如果本机确实运行着 Qdrant，请检查是否被系统代理拦截（设置 NO_PROXY=localhost,127.0.0.1）。")
            return QdrantClient(":memory:"), True

    async def _call(self, fn, *args, **kwargs):
        """在线程池中执行同步的 qdrant 调用；嵌入式模式下串行化"""
        def run():
            if self._lock is None:
                return fn(*args, **kwargs)
            with self._lock:
                return fn(*args, **kwargs)
        return await asyncio.to_thread(run)

    def _detect_existing_dimension(self):
        try:
            names = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in names:
                info = self.client.get_collection(self.collection_name)
                self.dimension = info.config.params.vectors.size
        except Exception as e:
            logger.debug(f"[Qdrant] 读取已有集合维度失败: {e}")

    def _ensure_collection(self, dim: int):
        if self.dimension == dim:
            return
        if self.dimension is not None and self.dimension != dim:
            raise ValueError(
                f"向量维度不一致：集合为 {self.dimension} 维，新向量为 {dim} 维。"
                f"更换向量模型后需要重建索引（见缺陷追踪 API-2）。")
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"[Qdrant] 创建集合 {self.collection_name}，维度 {dim}")
        self.dimension = dim

    async def add_chunks(self, chunks: List[ChildChunk]) -> int:
        if not chunks:
            return 0
        dim = len(chunks[0].embedding)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)),
                vector=c.embedding,
                payload={
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_name": c.doc_name,
                    "parent_id": c.parent_id, "content": c.content,
                    "chunk_index": c.chunk_index, "metadata": c.metadata,
                },
            )
            for c in chunks
        ]

        def write():
            self._ensure_collection(dim)
            self.client.upsert(collection_name=self.collection_name, points=points)
        await self._call(write)
        return len(points)

    async def search_by_vector(self, query_vector: List[float], top_k: int = None) -> List[SearchResultItem]:
        if self.dimension is None:
            return []
        limit = top_k or settings.rag.DENSE_TOP_K

        def query():
            if hasattr(self.client, "query_points"):
                return self.client.query_points(
                    collection_name=self.collection_name, query=query_vector,
                    limit=limit, with_payload=True).points
            return self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=limit)

        points = await self._call(query)
        out = []
        for rank, p in enumerate(points, start=1):
            pl = p.payload or {}
            out.append(_to_item(rank, p.score, pl.get("chunk_id", ""), pl.get("doc_id", ""),
                                pl.get("doc_name", ""), pl.get("parent_id"), pl.get("content", ""),
                                pl.get("metadata", {})))
        return out

    async def delete_by_doc_id(self, doc_id: str) -> bool:
        if self.dimension is None:
            return True

        def delete():
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            )
        await self._call(delete)
        return True

    async def clear(self):
        def drop():
            names = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in names:
                self.client.delete_collection(self.collection_name)
            self.dimension = None
        await self._call(drop)

    async def count(self) -> int:
        if self.dimension is None:
            return 0
        return await self._call(lambda: self.client.count(collection_name=self.collection_name, exact=True).count)


class InMemoryVectorStore(BaseVectorStore):
    """零外部依赖的纯内存向量存储（numpy 余弦相似度）"""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self.dimension: Optional[int] = None

    async def add_chunks(self, chunks: List[ChildChunk]) -> int:
        if not chunks:
            return 0
        dim = len(chunks[0].embedding)
        if self.dimension is not None and dim != self.dimension:
            raise ValueError(f"向量维度不一致：已有 {self.dimension} 维，新向量 {dim} 维，需要重建索引。")
        new_data = dict(self._data)  # 写时复制：检索线程持有旧引用时不受影响
        for c in chunks:
            new_data[c.chunk_id] = {"chunk": c, "vector": np.asarray(c.embedding, dtype=np.float32)}
        self._data = new_data
        self.dimension = dim
        return len(chunks)

    async def search_by_vector(self, query_vector: List[float], top_k: int = None) -> List[SearchResultItem]:
        limit = top_k or settings.rag.DENSE_TOP_K
        snapshot = list(self._data.values())
        if not snapshot:
            return []

        def compute():
            mat = np.stack([d["vector"] for d in snapshot])
            q = np.asarray(query_vector, dtype=np.float32)
            denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0))
            denom[denom == 0] = 1.0
            sims = mat @ q / denom
            order = np.argsort(-sims)[:limit]
            return [(float(sims[i]), snapshot[i]["chunk"]) for i in order]

        ranked = await asyncio.to_thread(compute)
        return [
            _to_item(rank, score, c.chunk_id, c.doc_id, c.doc_name, c.parent_id, c.content, c.metadata)
            for rank, (score, c) in enumerate(ranked, start=1)
        ]

    async def delete_by_doc_id(self, doc_id: str) -> bool:
        self._data = {cid: d for cid, d in self._data.items() if d["chunk"].doc_id != doc_id}
        return True

    async def clear(self):
        self._data = {}
        self.dimension = None

    async def count(self) -> int:
        return len(self._data)


class VectorStoreManager:
    """向量库门面：负责调用 Embedding、写入、检索、删除"""

    def __init__(self, store: Optional[BaseVectorStore] = None, force_in_memory: bool = False,
                 collection_name: Optional[str] = None):
        if store is not None:
            self.store = store
        elif QDRANT_AVAILABLE:
            # 优先 Qdrant（连不上独立服务时自动降级为嵌入式 :memory:）
            self.store = QdrantVectorStore(collection_name=collection_name, force_in_memory=force_in_memory)
        else:
            self.store = InMemoryVectorStore()

    async def add_chunks(self, chunks: List[ChildChunk]) -> int:
        """补齐缺失的向量后写入"""
        missing = [i for i, c in enumerate(chunks) if not c.embedding]
        if missing:
            vectors = await llm_provider.get_embeddings([chunks[i].content for i in missing])
            for i, v in zip(missing, vectors):
                chunks[i].embedding = v
        return await self.store.add_chunks(chunks)

    async def search_by_text(self, query: str, top_k: int = None) -> List[SearchResultItem]:
        query_vector = await llm_provider.get_embedding(query)
        return await self.store.search_by_vector(query_vector, top_k=top_k)

    async def delete_document(self, doc_id: str) -> bool:
        return await self.store.delete_by_doc_id(doc_id)

    async def clear(self):
        await self.store.clear()

    async def count(self) -> int:
        return await self.store.count()


# 全局单例管理器
vector_store_manager = VectorStoreManager()
