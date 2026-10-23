"""
并发与外部 API 回归测试（缺陷追踪 CONC-1、API-1 ~ API-3、RET-2）
"""

import sys
import asyncio
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.models.schemas import ChildChunk, SearchResultItem, RetrievalSource
from app.rag.bm25_index import BM25Index
from app.rag.reranker import CrossEncoderReranker
from app.rag.vector_store import InMemoryVectorStore


class TestBM25Concurrency(unittest.TestCase):
    def test_search_during_writes_never_crashes(self):
        """CONC-1 回归：原实现边写边搜时，300 次检索中出现 19 次
        RuntimeError: dictionary changed size during iteration"""
        def mk(i):
            return ChildChunk(chunk_id=f"c{i}", doc_id="d", doc_name="d", parent_id="p",
                              content=f"词{i} 共同词 token{i % 97}")
        b = BM25Index()
        b.build_index([mk(i) for i in range(1000)])
        errors = []

        def searcher():
            for _ in range(200):
                try:
                    b.search("共同词 token5", 20)
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))

        t = threading.Thread(target=searcher)
        t.start()
        for k in range(20):
            b.build_index([mk(100000 + k * 100 + i) for i in range(100)])
            b.remove_chunks([f"c{k}"])
        t.join()
        self.assertEqual(errors, [])


class _FakeEmbeddings:
    """记录每次请求条数的假 Embedding 客户端；超过 10 条时像百炼一样报错"""

    def __init__(self, limit=10):
        self.limit = limit
        self.batch_sizes = []

    async def create(self, model, input):
        self.batch_sizes.append(len(input))
        if len(input) > self.limit:
            raise RuntimeError("batch size is invalid, it should not be larger than 10")
        # 故意打乱返回顺序，验证代码按 index 还原
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), float(i)]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


class TestEmbeddingBatching(unittest.IsolatedAsyncioTestCase):
    async def test_batches_respect_provider_limit_and_keep_order(self):
        """API-1 回归：原实现一次发送全部子块，超过 10 条必然失败"""
        fake = _FakeEmbeddings()
        texts = [f"文本{i}" + "x" * i for i in range(23)]
        with mock.patch.object(llm_provider, "is_embed_configured", True), \
             mock.patch.object(llm_provider, "embed_client", SimpleNamespace(embeddings=fake), create=True), \
             mock.patch.object(settings.embedding, "BATCH_SIZE", 10), \
             mock.patch.object(settings.embedding, "CACHE_ENABLED", False):
            vectors = await llm_provider.get_embeddings(texts)
        self.assertEqual(sorted(fake.batch_sizes), [3, 10, 10])
        self.assertEqual(len(vectors), 23)
        self.assertEqual([v[0] for v in vectors], [float(len(t)) for t in texts])


class TestRerankFallback(unittest.IsolatedAsyncioTestCase):
    async def test_rerank_failure_falls_back_to_rrf_order(self):
        """API-3：重排服务出错时退回 RRF 排序，而不是让整个问答 500"""
        cands = [
            SearchResultItem(chunk_id=f"c{i}", doc_id="d", doc_name="d", parent_id=f"p{i}",
                             content=f"内容{i}", score=1.0 - i * 0.1, source=RetrievalSource.HYBRID_RRF)
            for i in range(4)
        ]
        rr = CrossEncoderReranker(top_n=3, min_score_threshold=0.5)
        with mock.patch.object(llm_provider, "rerank", side_effect=TimeoutError("rerank timeout")):
            out = await rr.rerank("问题", cands)
        self.assertEqual([x.chunk_id for x in out], ["c0", "c1", "c2"])
        self.assertTrue(all(x.metadata["rerank_fallback"] for x in out))


class TestVectorStore(unittest.IsolatedAsyncioTestCase):
    async def test_dimension_mismatch_is_reported(self):
        store = InMemoryVectorStore()
        c1 = ChildChunk(chunk_id="a", doc_id="d", doc_name="d", parent_id="p", content="x", embedding=[1.0, 0.0])
        c2 = ChildChunk(chunk_id="b", doc_id="d", doc_name="d", parent_id="p", content="y", embedding=[1.0, 0.0, 0.0])
        await store.add_chunks([c1])
        with self.assertRaises(ValueError):
            await store.add_chunks([c2])

    def test_module_imports_without_qdrant(self):
        """RET-2 回归：没装 qdrant-client 时，原实现导入阶段就 NameError。
        在子进程中屏蔽 qdrant_client，避免影响当前进程里已导入的模块。"""
        import subprocess
        code = (
            "import sys, os; sys.modules['qdrant_client'] = None; sys.modules['qdrant_client.models'] = None\n"
            "os.environ.update(LLM_API_KEY='', EMBEDDING_API_KEY='', RERANKER_API_KEY='')\n"
            "import app.rag.vector_store as vs\n"
            "assert not vs.QDRANT_AVAILABLE\n"
            "assert isinstance(vs.vector_store_manager.store, vs.InMemoryVectorStore)\n"
            "print('OK')\n"
        )
        proc = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=str(project_root),
                              capture_output=True, text=True, timeout=120)
        self.assertIn("OK", proc.stdout, msg=(proc.stderr or "")[-2000:])


if __name__ == "__main__":
    unittest.main()
