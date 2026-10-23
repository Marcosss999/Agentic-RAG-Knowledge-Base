"""
数据一致性回归测试（缺陷追踪 DATA-1 ~ DATA-6、SEC-2）
每个测试使用独立的临时目录与独立的检索器实例，不污染全局单例和 data/ 目录。
"""

import sys
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.security import SecurityError
from app.models.schemas import ChildChunk
from app.rag.bm25_index import BM25Index
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.knowledge_base import KnowledgeBase, DuplicateContentError, DocumentNotFound
from app.rag.loaders import extract_text
from app.rag.splitter import ParentChildSplitter
from app.rag.vector_store import VectorStoreManager, InMemoryVectorStore


def make_retriever() -> HybridRetriever:
    return HybridRetriever(vector_store=VectorStoreManager(store=InMemoryVectorStore()), bm25=BM25Index())


def minimal_pdf(pages):
    """手工构造一个带文字的最小 PDF（不依赖 reportlab），每个元素是一页的 ASCII 文本"""
    objs = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objs.append("<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>")
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


class TestBM25Correctness(unittest.TestCase):
    def _c(self, cid, text, doc="d1"):
        return ChildChunk(chunk_id=cid, doc_id=doc, doc_name=doc, parent_id="p", content=text)

    def test_incremental_avgdl_equals_full_rebuild(self):
        """DATA-3 回归：原实现第二批写入后 avgdl=16.67，真值 18.33"""
        inc = BM25Index()
        inc.build_index([self._c("a", "apple banana apple"), self._c("b", "banana cherry")])
        inc.build_index([self._c("c", "date " * 50)])
        full = BM25Index()
        full.build_index([self._c("a", "apple banana apple"), self._c("b", "banana cherry"), self._c("c", "date " * 50)])
        self.assertAlmostEqual(inc.avgdl, full.avgdl)
        self.assertAlmostEqual(inc.avgdl, sum(inc.doc_lengths.values()) / inc.corpus_size)

    def test_replacing_chunk_removes_old_postings(self):
        b = BM25Index()
        b.build_index([self._c("a", "oldword xyz")])
        b.build_index([self._c("a", "newword")])
        self.assertEqual(b.search("oldword"), [])
        self.assertEqual(b.search("newword")[0].chunk_id, "a")

    def test_remove_document(self):
        b = BM25Index()
        b.build_index([self._c("a", "alpha", "d1"), self._c("b", "alpha beta", "d2")])
        self.assertEqual(b.remove_document("d1"), 1)
        self.assertEqual([r.doc_id for r in b.search("alpha")], ["d2"])


class TestSplitter(unittest.TestCase):
    def test_zero_overlap_is_honored(self):
        """DATA-5 回归：原来 `0 or 150` 得到 150"""
        s = ParentChildSplitter(parent_chunk_overlap=0, child_chunk_overlap=0)
        self.assertEqual((s.p_overlap, s.c_overlap), (0, 0))


class TestRetrieverConsistency(unittest.IsolatedAsyncioTestCase):
    async def test_reindex_same_doc_id_does_not_duplicate(self):
        """DATA-1 回归：原实现同一 doc_id 索引两次后块数翻倍"""
        r = make_retriever()
        text = "天琴系统使用 RRF 融合 BM25 与向量检索。" * 20
        p1, c1 = await r.index_document(text, "d1", "a.txt")
        await r.index_document(text, "d1", "a.txt")
        self.assertEqual(r.bm25.corpus_size, c1)
        self.assertEqual(await r.vector_store.count(), c1)
        self.assertEqual(len(r.parent_docs_store), p1)

    async def test_delete_removes_from_all_stores(self):
        r = make_retriever()
        await r.index_document("文档一：报销需要发票。" * 10, "d1", "a.txt")
        await r.index_document("文档二：年假需要审批。" * 10, "d2", "b.txt")
        await r.delete_document("d1")
        self.assertTrue(all(c.doc_id == "d2" for c in r.bm25.doc_id_to_chunk.values()))
        self.assertTrue(all(p.doc_id == "d2" for p in r.parent_docs_store.values()))
        hits = await r.retrieve("报销发票", enable_rerank=False)
        self.assertTrue(all(h.doc_id == "d2" for h in hits))

    async def test_failed_embedding_keeps_old_version(self):
        """新版本准备失败时（如 Embedding 接口报错），旧版本必须完整保留"""
        r = make_retriever()
        await r.index_document("旧版本内容：报销流程。" * 10, "d1", "a.txt")
        before = r.bm25.corpus_size
        from app.core import llm_provider as lp
        with mock.patch.object(lp.llm_provider, "get_embeddings", side_effect=RuntimeError("embedding down")):
            with self.assertRaises(RuntimeError):
                await r.index_document("新版本内容。" * 10, "d1", "a.txt")
        self.assertEqual(r.bm25.corpus_size, before)
        self.assertTrue(r.bm25.search("报销流程"))

    async def test_results_deduplicated_by_parent_and_context_budget(self):
        """DATA-4 回归：原实现 5 个结果全来自同一母块，同一段文字在上下文里重复 5 次"""
        r = make_retriever()
        text = "".join(f"第{i}条：报销流程规定员工需在{i}日内提交发票并经主管审批。" for i in range(40))
        await r.index_document(text, "fin", "报销制度.txt")
        results = await r.retrieve("报销发票提交审批流程")
        parent_ids = [x.parent_id for x in results]
        self.assertEqual(len(parent_ids), len(set(parent_ids)))
        ctx, _ = r.build_context_prompt(results, max_chars=800)
        self.assertLess(len(ctx), 800 + 400)


class TestKnowledgeBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.upload_dir = root / "uploads"
        self.manifest = root / "manifest.json"
        self.kb = KnowledgeBase(retriever=make_retriever(), upload_dir=self.upload_dir, manifest_path=self.manifest)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_upload_replace_delete_cycle(self):
        rec = await self.kb.add_document("报销制度 v1：需要发票。".encode("utf-8") * 20, "报销.txt", "policy")
        self.assertFalse(rec["replaced"])
        self.assertTrue((self.upload_dir / "policy.txt").exists())
        rec2 = await self.kb.add_document("报销制度 v2：需要电子发票。".encode("utf-8") * 20, "报销.txt", "policy")
        self.assertTrue(rec2["replaced"])
        self.assertEqual(len(self.kb.list_documents()), 1)
        self.assertEqual(self.kb.retriever.document_ids(), ["policy"])

        await self.kb.delete_document("policy")
        self.assertFalse((self.upload_dir / "policy.txt").exists())
        self.assertEqual(self.kb.retriever.bm25.corpus_size, 0)
        self.assertEqual(json.loads(self.manifest.read_text(encoding="utf-8"))["documents"], {})
        with self.assertRaises(DocumentNotFound):
            await self.kb.delete_document("policy")

    async def test_duplicate_content_rejected(self):
        data = "同一份内容".encode("utf-8") * 30
        await self.kb.add_document(data, "a.txt", "doc_a")
        with self.assertRaises(DuplicateContentError) as cm:
            await self.kb.add_document(data, "b.txt", "doc_b")
        self.assertEqual(cm.exception.existing_doc_id, "doc_a")

    async def test_rejects_unsafe_input(self):
        """SEC-2 回归：doc_id 路径穿越、不支持的类型、超大文件"""
        with self.assertRaises(SecurityError):
            await self.kb.add_document(b"x" * 100, "a.txt", "../../evil")
        with self.assertRaises(SecurityError):
            await self.kb.add_document(b"x" * 100, "run.exe", None)
        from app.core.config import settings
        with mock.patch.object(settings.security, "MAX_UPLOAD_MB", 0.001):
            with self.assertRaises(SecurityError):
                await self.kb.add_document(b"x" * 5000, "a.txt", None)
        # 文件名里的目录成分被去掉，磁盘文件名由服务端生成
        rec = await self.kb.add_document("正常内容".encode("utf-8") * 20, "../../../etc/报告.md", "doc_ok")
        self.assertEqual(rec["doc_name"], "报告.md")
        self.assertEqual(sorted(p.name for p in self.upload_dir.iterdir()), ["doc_ok.md"])

    async def test_restart_rebuild_keeps_doc_ids_and_pdf(self):
        """DATA-2 / DATA-6 回归：重启后 PDF 仍可检索，doc_id 不变"""
        pdf = minimal_pdf(["Reimbursement requires invoices", "Annual leave needs approval"])
        text_now = extract_text(pdf, "policy.pdf")
        self.assertIn("--- [第 1 页] ---", text_now)
        await self.kb.add_document(pdf, "policy.pdf", "pdf_policy")
        await self.kb.add_document("纯文本文档：年假审批流程。".encode("utf-8") * 20, "leave.txt", "txt_leave")

        # 模拟重启：全新的检索器与知识库实例，只依赖磁盘上的文件与清单
        kb2 = KnowledgeBase(retriever=make_retriever(), upload_dir=self.upload_dir, manifest_path=self.manifest)
        result = await kb2.rebuild_from_disk()
        self.assertEqual(result, {"indexed": 2, "failed": 0})
        self.assertEqual(kb2.retriever.document_ids(), ["pdf_policy", "txt_leave"])
        hits = await kb2.retriever.retrieve("invoices reimbursement", enable_rerank=False)
        self.assertEqual(hits[0].doc_id, "pdf_policy")
        # 上传与重启两条路径切出的块内容完全一致
        before = sorted(c.content for c in self.kb.retriever.bm25.doc_id_to_chunk.values() if c.doc_id == "pdf_policy")
        after = sorted(c.content for c in kb2.retriever.bm25.doc_id_to_chunk.values() if c.doc_id == "pdf_policy")
        self.assertEqual(before, after)

    async def test_legacy_files_are_registered(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "doc_old_01_旧文档.txt").write_text("历史文档内容。" * 20, encoding="utf-8")
        result = await self.kb.rebuild_from_disk()
        self.assertEqual(result["indexed"], 1)
        doc_id = self.kb.list_documents()[0]["doc_id"]
        self.assertRegex(doc_id, r"^[A-Za-z0-9_\-]+$")

    async def test_embedding_model_change_triggers_rebuild(self):
        """API-2：向量模型变化后必须重建，否则新旧向量混在一个索引里"""
        await self.kb.add_document("内容".encode("utf-8") * 50, "a.txt", "d1")
        self.assertFalse(await self.kb.ensure_embedding_consistency())
        from app.core import llm_provider as lp
        with mock.patch.object(type(lp.llm_provider), "embedding_signature",
                               new_callable=mock.PropertyMock, return_value="api:x:new-model"):
            self.assertTrue(await self.kb.ensure_embedding_consistency())
            self.assertEqual(self.kb.stats()["embedding_signature"], "api:x:new-model")


if __name__ == "__main__":
    unittest.main()
