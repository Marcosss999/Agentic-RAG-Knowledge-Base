"""
评测流水线测试（缺陷追踪 EVAL-1 ~ EVAL-3、RET-3）
- 指标计算的数学正确性
- 评测集完整性：每道可回答题的证据都能在对应文档的抽取文本中找到（也能发现不同 pypdf 版本抽取结果不一致的问题）
- 检索消融的开关确实生效（只用 BM25 时不调用向量检索）
- 端到端评测：判官只做逐条判断，比例由代码计算
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.llm_provider import llm_provider
from app.eval import pipeline as P
from app.rag.loaders import extract_text


def _res(content, parent=None):
    return SimpleNamespace(content=content, parent_content=parent)


class TestMetrics(unittest.TestCase):
    def test_relevance_ignores_whitespace_and_case(self):
        self.assertTrue(P.is_relevant(_res("…appear in two feature\ndimensions (1415 and 2533)…"), ["Two feature dimensions (1415 and 2533)"]))
        self.assertTrue(P.is_relevant(_res("子块", parent="一线城市住宿费每晚不超过 600 元"), ["一线城市住宿费每晚不超过600元"]))
        self.assertFalse(P.is_relevant(_res("无关内容"), ["600 元"]))
        self.assertFalse(P.is_relevant(_res("任何内容"), []))

    def test_rank_summary(self):
        s = P.summarize_ranks([1, 3, None, 5])
        self.assertAlmostEqual(s["hit@1"], 0.25)
        self.assertAlmostEqual(s["hit@3"], 0.5)
        self.assertAlmostEqual(s["hit@5"], 0.75)
        self.assertAlmostEqual(s["mrr"], (1 + 1 / 3 + 1 / 5) / 4)

    def test_paired_bootstrap(self):
        d, lo, hi = P.paired_bootstrap([0.5] * 20, [0.5] * 20)
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))
        d, lo, hi = P.paired_bootstrap([0.0] * 30, [1.0] * 30)
        self.assertEqual((d, lo, hi), (1.0, 1.0, 1.0))
        d, lo, hi = P.paired_bootstrap([0, 1] * 10, [1, 0] * 10)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_threshold_analysis(self):
        scores = {
            "a": {"answerable": True, "scores": [0.8, 0.2], "relevant": [True, False]},
            "b": {"answerable": True, "scores": [0.3, 0.1], "relevant": [False, True]},
            "u": {"answerable": False, "scores": [0.25], "relevant": [False]},
        }
        ta = P.threshold_analysis(scores, current=0.15)
        row = {r["threshold"]: r for r in ta["grid"]}
        self.assertEqual(row[0.0]["answerable_hit@5"], 1.0)
        self.assertEqual(row[0.15]["answerable_hit@5"], 0.5)      # b 的相关片段只有 0.1 分，被过滤
        self.assertEqual(row[0.2]["unanswerable_filtered"], 0.0)
        self.assertEqual(row[0.3]["unanswerable_filtered"], 1.0)


class TestDatasetIntegrity(unittest.TestCase):
    def test_bundled_dataset_evidence_exists_in_sources(self):
        ds = P.load_dataset()
        texts = {}
        for src in ds["sources"]:
            path = project_root / src["path"]
            self.assertTrue(path.exists(), f"评测文档缺失: {path}")
            texts[src["key"]] = P._norm(extract_text(path.read_bytes(), src["doc_name"]))
        ids = [it["id"] for it in ds["items"]]
        self.assertEqual(len(ids), len(set(ids)), "题目 id 重复")
        for it in ds["items"]:
            if not it["answerable"]:
                self.assertEqual(it["evidence"], [])
                continue
            self.assertTrue(any(P._norm(e) in texts[it["doc"]] for e in it["evidence"]),
                            f"{it['id']} 的证据在文档抽取文本中找不到：{it['evidence']}")
        self.assertGreaterEqual(sum(1 for it in ds["items"] if not it["answerable"]), 5)


def _tiny_dataset(tmp: Path):
    (tmp / "a.txt").write_text("报销制度：住宿费每晚不超过 600 元。伙食补助每天 100 元，无需发票。" * 3, encoding="utf-8")
    (tmp / "b.txt").write_text("年假制度：入职满一年享有 5 天带薪年假。" * 3, encoding="utf-8")
    ds = {
        "name": "tiny", "sources": [{"key": "a", "path": "a.txt", "doc_name": "a.txt"},
                                    {"key": "b", "path": "b.txt", "doc_name": "b.txt"}],
        "items": [
            {"id": "q1", "question": "住宿费每晚上限多少", "reference_answer": "600 元", "evidence": ["每晚不超过 600 元"],
             "doc": "a", "answerable": True, "lang": "zh"},
            {"id": "q2", "question": "年假有几天", "reference_answer": "5 天", "evidence": ["5 天带薪年假"],
             "doc": "b", "answerable": True, "lang": "zh"},
            {"id": "u1", "question": "股票期权怎么行权", "reference_answer": "无", "evidence": [], "doc": None,
             "answerable": False, "lang": "zh"},
        ],
    }
    path = tmp / "ds.json"
    path.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")
    return path


class TestRetrievalEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_runs_and_bm25_only_skips_dense(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ds = P.load_dataset(_tiny_dataset(tmp))
            retriever = await P.build_eval_retriever(ds, root=tmp)
            configs = [P.RetrievalConfig("bm25_only", "BM25", 0.0, 1.0),
                       P.RetrievalConfig("hybrid_rerank", "线上", 1.0, 0.8, rerank=True)]
            dense_calls = []
            original = retriever._safe_dense_search

            async def spy(q):
                dense_calls.append(q)
                return await original(q)

            with mock.patch.object(retriever, "_safe_dense_search", spy):
                out = await P.evaluate_retrieval(ds, retriever, configs=[configs[0]], include_llm_configs=False)
            self.assertEqual(dense_calls, [])          # 只用 BM25 时完全不调用向量检索
            self.assertEqual(out["configs"][0]["hit@5"], 1.0)

            out = await P.evaluate_retrieval(ds, retriever, configs=configs, include_llm_configs=False)
            names = [c["name"] for c in out["configs"]]
            self.assertEqual(names, ["bm25_only", "hybrid_rerank"])
            self.assertIn("delta_vs_baseline", out["configs"][0])
            self.assertIsNotNone(out["threshold_analysis"])
            # 评测结束后恢复原状：重排阈值与 get_embedding 不被评测代码改掉
            self.assertNotEqual(retriever.reranker.min_score, 0.0)
            self.assertFalse(hasattr(llm_provider.get_embedding, "__self__") and
                             isinstance(llm_provider.get_embedding.__self__, P._QueryEmbeddingCache))


class TestEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def test_judges_feed_code_computed_metrics(self):
        from tests.test_agent_flow import FakeLLM

        class JudgeAwareLLM(FakeLLM):
            def _next(self, kind, messages):
                system = messages[0]["content"]
                if "阅卷人" in system:
                    return '{"verdict": "correct", "reason": "ok"}'
                if "拆成独立的事实论断" in system:
                    return '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}'
                if "是否明确告诉用户" in system:
                    return '{"refused": true}'
                return super()._next(kind, messages)

        fake = JudgeAwareLLM()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ds = P.load_dataset(_tiny_dataset(tmp))
            retriever = await P.build_eval_retriever(ds, root=tmp)
            with mock.patch.object(llm_provider, "is_llm_configured", True), \
                 mock.patch.object(llm_provider, "chat", fake.chat), \
                 mock.patch.object(llm_provider, "chat_stream", fake.chat_stream):
                out = await P.evaluate_end_to_end(ds, retriever, judge_repeats=2)
        s = out["summary"]
        self.assertEqual(out["n"], 3)
        self.assertEqual(s["correctness"], 1.0)
        self.assertEqual(s["judge_agreement"], 1.0)
        self.assertEqual(s["refusal_accuracy"], 1.0)
        rows = {r["id"]: r for r in out["rows"]}
        if rows["q1"].get("faithfulness") is not None:
            self.assertAlmostEqual(rows["q1"]["faithfulness"], 0.5)   # 2 条论断支持 1 条，由代码计算
        self.assertGreater(rows["q1"]["system_llm_calls"], 0)

    async def test_skipped_without_llm(self):
        with mock.patch.object(llm_provider, "is_llm_configured", False):
            out = await P.evaluate_end_to_end({"items": []}, retriever=None)
        self.assertTrue(out["skipped"])


if __name__ == "__main__":
    unittest.main()
