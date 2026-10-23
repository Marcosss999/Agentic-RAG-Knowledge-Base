"""
第 7 组回归测试：Embedding 磁盘缓存 (DATA-7)、后台入库 (CONC-2)、PDF 页码与 CSV 表头 (ENG-7)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.embedding_cache import EmbeddingCache
from app.core.llm_provider import llm_provider
from app.rag.bm25_index import BM25Index
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.ingest_tasks import IngestTaskManager
from app.rag.knowledge_base import KnowledgeBase
from app.rag.loaders import extract_text
from app.rag.splitter import ParentChildSplitter
from app.rag.vector_store import VectorStoreManager, InMemoryVectorStore
from tests.test_data_consistency import minimal_pdf


class _CountingEmbeddings:
    def __init__(self):
        self.texts_sent = []

    async def create(self, model, input):
        self.texts_sent.extend(input)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(len(t)), 1.0, 0.5])
                                     for i, t in enumerate(input)])


class OnlineEmbeddingMixin:
    """模拟"已配置在线向量模型"，并把缓存指向临时文件"""

    def setUp_online(self, tmp: Path):
        self.fake = _CountingEmbeddings()
        for p in (
            mock.patch.object(llm_provider, "is_embed_configured", True),
            mock.patch.object(llm_provider, "embed_client", SimpleNamespace(embeddings=self.fake), create=True),
            mock.patch.object(settings.embedding, "CACHE_ENABLED", True),
            mock.patch.object(settings.storage, "EMBEDDING_CACHE_PATH", tmp / "cache.sqlite"),
        ):
            p.start()
            self.addCleanup(p.stop)


class TestEmbeddingCache(OnlineEmbeddingMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.setUp_online(Path(self.tmp.name))
        self.addCleanup(self._close_global_cache)

    @staticmethod
    def _close_global_cache():
        import app.core.embedding_cache as ec
        if ec._cache is not None:
            ec._cache.close()
            ec._cache = None

    def test_roundtrip_and_signature_isolation(self):
        c = EmbeddingCache(Path(self.tmp.name) / "c.sqlite")
        self.addCleanup(c.close)
        c.put_many("model-a", ["你好", "世界"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(c.get_many("model-a", ["世界", "未知", "你好"]), {0: [3.0, 4.0], 2: [1.0, 2.0]})
        self.assertEqual(c.get_many("model-b", ["你好"]), {})   # 换模型不会误用旧向量

    async def test_second_call_hits_cache(self):
        texts = [f"片段{i}" for i in range(23)]
        progress = []
        v1 = await llm_provider.get_embeddings(texts, progress=lambda d, t: progress.append((d, t)))
        self.assertEqual(len(self.fake.texts_sent), 23)
        self.assertEqual(progress[-1], (23, 23))
        v2 = await llm_provider.get_embeddings(texts + ["新片段"])
        self.assertEqual(self.fake.texts_sent[23:], ["新片段"])     # 只为未命中的文本调用接口
        self.assertEqual(v2[:23], v1)

    async def test_restart_rebuild_does_not_call_api_again(self):
        """DATA-7：重启后按清单重建索引，向量全部来自缓存"""
        tmp = Path(self.tmp.name)
        make = lambda: HybridRetriever(vector_store=VectorStoreManager(store=InMemoryVectorStore()), bm25=BM25Index())
        kb = KnowledgeBase(retriever=make(), upload_dir=tmp / "u", manifest_path=tmp / "m.json")
        await kb.add_document("差旅报销需要提交发票原件。".encode("utf-8") * 60, "a.txt", "d1")
        sent_before = len(self.fake.texts_sent)
        self.assertGreater(sent_before, 0)
        kb2 = KnowledgeBase(retriever=make(), upload_dir=tmp / "u", manifest_path=tmp / "m.json")
        await kb2.rebuild_from_disk()
        self.assertEqual(len(self.fake.texts_sent), sent_before)
        self.assertEqual(kb2.retriever.bm25.corpus_size, kb.retriever.bm25.corpus_size)


class TestBackgroundIngest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.kb = KnowledgeBase(
            retriever=HybridRetriever(vector_store=VectorStoreManager(store=InMemoryVectorStore()), bm25=BM25Index()),
            upload_dir=tmp / "u", manifest_path=tmp / "m.json")
        self.tasks = IngestTaskManager()

    async def test_task_reports_progress_and_result(self):
        data = "年假制度：入职满一年享有 5 天带薪年假。".encode("utf-8") * 40
        stages = []

        async def job(progress):
            def spy(stage, done, total):
                stages.append(stage)
                progress(stage, done, total)
            return await self.kb.add_document(data, "年假.txt", "leave", progress=spy)

        tid = self.tasks.submit("年假.txt", job)
        info = await self.tasks.wait(tid)
        self.assertEqual(info["status"], "done")
        self.assertEqual(info["result"]["doc_id"], "leave")
        self.assertEqual([s for s in dict.fromkeys(stages)], ["解析文本", "向量化", "写入索引"])

    async def test_failure_is_reported(self):
        data = "重复内容".encode("utf-8") * 30
        await self.kb.add_document(data, "a.txt", "a")
        tid = self.tasks.submit("b.txt", lambda progress: self.kb.add_document(data, "b.txt", "b", progress=progress))
        info = await self.tasks.wait(tid)
        self.assertEqual(info["status"], "failed")
        self.assertIn("相同内容", info["error"])
        self.assertIsNone(self.tasks.get("not-exist"))


class TestPagesAndCsv(unittest.TestCase):
    def test_pdf_children_carry_page_numbers(self):
        """ENG-7：引用可以显示"第 n 页" """
        page1 = "Page one discusses invoices and reimbursement policy " * 12
        page2 = "Page two discusses annual leave approval rules " * 12
        text = extract_text(minimal_pdf([page1[:200], page2[:200]]), "x.pdf")
        parents, children = ParentChildSplitter(child_chunk_size=120, child_chunk_overlap=0).process_document(text, "d", "x.pdf")
        pages = {c.metadata.get("page") for c in children}
        self.assertTrue({1, 2} <= pages, pages)
        for c in children:
            if "annual leave" in c.content and "invoices" not in c.content:
                self.assertEqual(c.metadata["page"], 2)

    def test_plain_text_has_no_page(self):
        _, children = ParentChildSplitter().process_document("普通文本，没有页码标记。" * 10, "d", "a.txt")
        self.assertTrue(all("page" not in c.metadata for c in children))

    def test_csv_rows_carry_header(self):
        csv_bytes = "城市,住宿标准,伙食补助\n北京,600,100\n成都,400,100\n,,\n".encode("utf-8")
        text = extract_text(csv_bytes, "标准.csv")
        lines = text.splitlines()
        self.assertEqual(len(lines), 2)                   # 空行被跳过
        self.assertIn("城市：成都", lines[1])
        self.assertIn("住宿标准：400", lines[1])

    def test_citation_metadata_includes_page(self):
        from app.models.schemas import SearchResultItem, RetrievalSource
        r = HybridRetriever.__new__(HybridRetriever)
        item = SearchResultItem(chunk_id="c", doc_id="d", doc_name="x.pdf", content="内容", score=0.9,
                                source=RetrievalSource.RERANKED, metadata={"page": 3, "source_path": "secret"})
        _, cits = HybridRetriever.build_context_prompt(r, [item])
        self.assertEqual(cits[0].metadata, {"page": 3})


if __name__ == "__main__":
    unittest.main()
