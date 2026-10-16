"""
密集语义向量存储与检索引擎 (Dense Vector Store Engine)
职责：
1. 负责子文档块（Child Chunk）高维语义向量的持久化存储与基于余弦相似度（Cosine Similarity）的高速最近邻搜索。
2. 抽象 BaseVectorStore 统一接口，屏蔽底层向量库驱动差异。
3. 深度适配 Qdrant 向量引擎（支持无依赖纯内存模式 ':memory:' 与生产环境独立集群模式），并附带零依赖纯 Python 矩阵备用方案。
4. 提供开箱即用的语义检索门面：支持直接输入文本搜索（内部自动调用 Embedding 模型转向量）。
"""

import uuid
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.models.schemas import ChildChunk, SearchResultItem, RetrievalSource

logger = logging.getLogger(__name__)

# 尝试导入 Qdrant 官方客户端
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("[VectorStore] 未检测到 qdrant-client，将使用内置纯内存备用向量引擎。")


class BaseVectorStore(ABC):
    """向量库统一抽象基类"""

    @abstractmethod
    async def add_chunks(self, chunks: List[ChildChunk]) -> int:
        """存入带向量的子切片列表"""
        pass

    @abstractmethod
    async def search_by_vector(self, query_vector: List[float], top_k: int = None) -> List[SearchResultItem]:
        """通过向量特征执行 Top-K 余弦最近邻搜索"""
        pass

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> bool:
        """根据文档 ID 级联删除该文档的所有切片向量"""
        pass

    @abstractmethod
    async def clear(self):
        """清空向量库"""
        pass


class QdrantVectorStore(BaseVectorStore):
    """基于 Qdrant 的企业级向量引擎实现"""

    def __init__(self, collection_name: str = None, force_in_memory: bool = False):
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client 未安装，无法初始化 QdrantVectorStore")

        self.collection_name = collection_name or settings.storage.COLLECTION_NAME
        self.dimension = settings.embedding.DIMENSION

        # 初始化客户端连接：如果指定为 memory 或外部服务连不上，自动安全降级至 :memory:
        self.client = self._init_client(force_in_memory)
        self._ensure_collection()

    def _init_client(self, force_in_memory: bool) -> QdrantClient:
        if force_in_memory or settings.storage.VECTOR_DB_TYPE == "memory":
            logger.info(f"[Qdrant] 启用嵌入式内存向量数据库模式 (:memory:)")
            return QdrantClient(":memory:")

        try:
            client = QdrantClient(
                host=settings.storage.QDRANT_HOST,
                port=settings.storage.QDRANT_PORT,
                timeout=2.0
            )
            # 主动试探远程端口连通性
            client.get_collections()
            logger.info(f"[Qdrant] 成功连接至独立 Qdrant 集群: {settings.storage.QDRANT_HOST}:{settings.storage.QDRANT_PORT}")
            return client
        except Exception as e:
            logger.warning(f"[Qdrant] 无法连接到远程 Qdrant 服务 ({e})，为保障单机开发平滑运行，已自动降级为内置内存模式 (:memory:)")
            return QdrantClient(":memory:")

    def _ensure_collection(self):
        """确保目标 Collection 存在且配置正确的余弦距离度量"""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
                )
                logger.info(f"[Qdrant] 成功创建向量集合: {self.collection_name}, 向量维度: {self.dimension}")
        except Exception as e:
            logger.error(f"[Qdrant] 初始化集合失败: {e}")
            raise

    async def add_chunks(self, chunks: List[ChildChunk]) -> int:
        if not chunks:
            return 0

        # 如果部分切片未计算向量，批量调用 Embedding 模型补齐
        missing_embed_indices = [i for i, c in enumerate(chunks) if not c.embedding]
        if missing_embed_indices:
            texts_to_embed = [chunks[i].content for i in missing_embed_indices]
            computed_vectors = await llm_provider.get_embeddings(texts_to_embed)
            for idx, vec in zip(missing_embed_indices, computed_vectors):
                chunks[idx].embedding = vec

        # 构建 Qdrant 数据点 PointStruct
        points = []
        for c in chunks:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id))
            payload = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "parent_id": c.parent_id,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata
            }
            points.append(PointStruct(id=point_id, vector=c.embedding, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"[Qdrant] 成功写入/更新 {len(points)} 条切片向量")
        return len(points)

    async def search_by_vector(self, query_vector: List[float], top_k: int = None) -> List[SearchResultItem]:
        limit = top_k or settings.rag.DENSE_TOP_K

        # 调用 Qdrant 执行余弦检索 (兼容旧版 search 与新版 query_points)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True
            )
            scored_points = response.points
        else:
            scored_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

        results: List[SearchResultItem] = []
        for rank, p in enumerate(scored_points, start=1):
            pl = p.payload or {}
            sim_score = round(float(p.score), 4)

            item = SearchResultItem(
                chunk_id=pl.get("chunk_id", ""),
                doc_id=pl.get("doc_id", ""),
                doc_name=pl.get("doc_name", ""),
                parent_id=pl.get("parent_id"),
                content=pl.get("content", ""),
                score=sim_score,
                source=RetrievalSource.DENSE_VECTOR,
                rank=rank,
                metadata={
                    **pl.get("metadata", {}),
                    "raw_cosine_score": sim_score
                }
            )
            results.append(item)

        return results

    async def delete_by_doc_id(self, doc_id: str) -> bool:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )
        return True

    async def clear(self):
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()


class InMemoryVectorStore(BaseVectorStore):
    """零外部依赖的纯 Python / 余弦相似度向量库备用实现"""

    def __init__(self):
        self.chunks_data: Dict[str, Dict[str, Any]] = {}

    async def add_chunks(self, chunks: List[ChildChunk]) -> int:
        if not chunks:
            return 0
        missing = [i for i, c in enumerate(chunks) if not c.embedding]
        if missing:
            vecs = await llm_provider.get_embeddings([chunks[i].content for i in missing])
            for i, v in zip(missing, vecs):
                chunks[i].embedding = v

        for c in chunks:
            self.chunks_data[c.chunk_id] = {
                "chunk": c,
                "vector": c.embedding
            }
        return len(chunks)

    async def search_by_vector(self, query_vector: List[float], top_k: int = None) -> List[SearchResultItem]:
        limit = top_k or settings.rag.DENSE_TOP_K
        if not self.chunks_data:
            return []

        def cosine_sim(v1: List[float], v2: List[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm_a = sum(a * a for a, b in zip(v1, v2)) ** 0.5
            norm_b = sum(b * b for a, b in zip(v1, v2)) ** 0.5
            return dot / (norm_a * norm_b or 1.0)

        scored_list = []
        for cid, data in self.chunks_data.items():
            sim = cosine_sim(query_vector, data["vector"])
            scored_list.append((sim, data["chunk"]))

        scored_list.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, chunk) in enumerate(scored_list[:limit], start=1):
            results.append(SearchResultItem(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_name,
                parent_id=chunk.parent_id,
                content=chunk.content,
                score=round(float(score), 4),
                source=RetrievalSource.DENSE_VECTOR,
                rank=rank,
                metadata={**chunk.metadata, "raw_cosine_score": round(float(score), 4)}
            ))
        return results

    async def delete_by_doc_id(self, doc_id: str) -> bool:
        to_del = [cid for cid, d in self.chunks_data.items() if d["chunk"].doc_id == doc_id]
        for cid in to_del:
            del self.chunks_data[cid]
        return True

    async def clear(self):
        self.chunks_data.clear()


class VectorStoreManager:
    """向量库统一管理门面类"""

    def __init__(self):
        if QDRANT_AVAILABLE:
            # 默认优先采用 Qdrant (如无外部集群服务则自动降级为 :memory: 模式)
            self.store: BaseVectorStore = QdrantVectorStore()
        else:
            self.store: BaseVectorStore = InMemoryVectorStore()

    async def add_chunks(self, chunks: List[ChildChunk]) -> int:
        return await self.store.add_chunks(chunks)

    async def search_by_text(self, query: str, top_k: int = None) -> List[SearchResultItem]:
        """高级便捷方法：输入文本 -> 自动向量化 -> 执行余弦搜索"""
        query_vector = await llm_provider.get_embedding(query)
        return await self.store.search_by_vector(query_vector, top_k=top_k)


# 全局单例管理器
vector_store_manager = VectorStoreManager()
