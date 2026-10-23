"""
FastAPI 接口集成测试
- 管理接口的权限控制、CORS 白名单、上传校验、文档增删、评测报告名校验（SEC-1/2/4/6、DATA-1）
- 文件写入全部重定向到临时目录，不会改动 data/ 与 .env
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from main import app
from app.core.config import settings
from app.rag.knowledge_base import knowledge_base

TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}


async def _no_web(query):  # 测试不访问外网
    return None


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.patches = [
            mock.patch.object(settings.security, "ADMIN_TOKEN", TOKEN),
            mock.patch.object(settings.storage, "EVAL_DATASET_DIR", root / "eval"),
            mock.patch.object(knowledge_base, "upload_dir", root / "uploads"),
            mock.patch.object(knowledge_base, "manifest_path", root / "manifest.json"),
            mock.patch.object(knowledge_base, "_manifest", {"version": 1, "embedding_signature": None, "documents": {}}),
            mock.patch("app.agent.self_rag_graph._fetch_external_web_knowledge", _no_web),
        ]
        for p in cls.patches:
            p.start()
        (root / "uploads").mkdir()
        (root / "eval").mkdir()
        from app.eval.ragas_evaluator import ragas_evaluator
        cls.eval_patch = mock.patch.object(ragas_evaluator, "eval_output_dir", root / "eval")
        cls.eval_patch.start()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.eval_patch.stop()
        for p in reversed(cls.patches):
            p.stop()
        cls.tmp.cleanup()

    # ------------------------------------------------------------- 基础
    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_rag_status_endpoint(self):
        response = self.client.get("/api/v1/rag/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("vector_db_type", data)
        self.assertIn("documents_indexed", data)

    def test_get_settings_masked(self):
        data = self.client.get("/api/v1/rag/settings").json()
        for k in ("llm", "embedding", "reranker"):
            self.assertIn(k, data)
            self.assertNotIn("api_key", data[k])  # 只有 api_key_masked / api_key_configured
        self.assertTrue(data["admin_token_required"])

    # ------------------------------------------------------------- 管理权限 (SEC-4)
    def test_admin_endpoints_require_token(self):
        self.assertEqual(self.client.post("/api/v1/rag/settings/test").status_code, 401)
        self.assertEqual(self.client.post("/api/v1/rag/settings", json={"llm_default_model": "x"}).status_code, 401)
        self.assertEqual(self.client.post("/api/v1/rag/evaluation/run", json={}).status_code, 401)
        self.assertEqual(self.client.delete("/api/v1/rag/documents/whatever").status_code, 401)

    def test_settings_test_connectivity_with_token(self):
        response = self.client.post("/api/v1/rag/settings/test", headers=ADMIN)
        self.assertEqual(response.status_code, 200)
        self.assertIn("llm", response.json()["components"])

    def test_settings_rejects_unsafe_values(self):
        r = self.client.post("/api/v1/rag/settings", headers=ADMIN, json={"llm_base_url": "http://evil.example/v1"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/v1/rag/settings", headers=ADMIN, json={"llm_default_model": "qwen\nADMIN_TOKEN=x"})
        self.assertEqual(r.status_code, 400)

    def test_cors_only_allows_whitelisted_origins(self):
        def preflight(origin):
            return self.client.options("/api/v1/rag/settings", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"})
        self.assertNotEqual(preflight("https://evil.example").headers.get("access-control-allow-origin"),
                            "https://evil.example")
        self.assertEqual(preflight("http://localhost:8000").headers.get("access-control-allow-origin"),
                         "http://localhost:8000")

    # ------------------------------------------------------------- 上传与文档管理 (SEC-2 / DATA-1)
    def _upload(self, content: bytes, filename="a.txt", doc_id=None):
        data = {"doc_id": doc_id} if doc_id else {}
        return self.client.post("/api/v1/rag/documents/upload", files={"file": (filename, content)}, data=data)

    def test_upload_rejects_unsafe_input(self):
        self.assertEqual(self._upload(b"hello" * 20, doc_id="../../evil").status_code, 400)
        self.assertEqual(self._upload(b"MZ...", filename="run.exe").status_code, 400)
        with mock.patch.object(settings.security, "MAX_UPLOAD_MB", 0.001):
            self.assertEqual(self._upload(b"x" * 5000).status_code, 413)
        self.assertEqual(list((Path(self.tmp.name) / "uploads").glob("*evil*")), [])

    def test_upload_list_duplicate_delete_cycle(self):
        content = "API 测试文档：差旅报销需要提供发票原件。".encode("utf-8") * 10
        r = self._upload(content, "差旅.txt", "api_doc")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertGreater(r.json()["child_chunks"], 0)

        ids = [d["doc_id"] for d in self.client.get("/api/v1/rag/documents").json()["documents"]]
        self.assertIn("api_doc", ids)

        self.assertEqual(self._upload(content, "copy.txt", "api_doc_copy").status_code, 409)

        self.assertEqual(self.client.delete("/api/v1/rag/documents/api_doc", headers=ADMIN).status_code, 200)
        self.assertEqual(self.client.delete("/api/v1/rag/documents/api_doc", headers=ADMIN).status_code, 404)

    def test_background_upload_and_task_polling(self):
        """CONC-2：后台入库立即返回 202 + 任务 ID，轮询到 done。
        需要用 with 保持同一个事件循环存活，后台任务才能在请求之间继续运行。"""
        import time as _time
        content = "后台入库测试：会议室预订需提前一天在 OA 提交。".encode("utf-8") * 20
        with TestClient(app) as c:
            r = c.post("/api/v1/rag/documents/upload?background=true",
                       files={"file": ("会议室.txt", content)}, data={"doc_id": "bg_doc"})
            self.assertEqual(r.status_code, 202, r.text)
            task_id = r.json()["task_id"]
            info = {}
            for _ in range(100):
                info = c.get(f"/api/v1/rag/documents/tasks/{task_id}").json()
                if info["status"] in ("done", "failed"):
                    break
                _time.sleep(0.05)
            self.assertEqual(info["status"], "done", info)
            self.assertEqual(info["result"]["doc_id"], "bg_doc")
            self.assertEqual(c.get("/api/v1/rag/documents/tasks/nope").status_code, 404)
            # 后台模式同样先做同步校验
            bad = c.post("/api/v1/rag/documents/upload?background=true",
                         files={"file": ("x.exe", b"MZ")})
            self.assertEqual(bad.status_code, 400)
            c.delete("/api/v1/rag/documents/bg_doc", headers=ADMIN)

    # ------------------------------------------------------------- 评测 (SEC-1)
    def test_evaluation_report_name_traversal_rejected(self):
        r = self.client.post("/api/v1/rag/evaluation/run", headers=ADMIN, json={"report_name": "../../main.py"})
        self.assertEqual(r.status_code, 400)

    # ------------------------------------------------------------- 问答 (SEC-6)
    def test_chat_query_endpoint(self):
        response = self.client.post("/api/v1/rag/chat/query", json={"query": "你好，请简要介绍平台", "max_retries": 1})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for k in ("answer", "steps_timeline", "citations", "quality"):
            self.assertIn(k, data)
        self.assertIn("audit_status", data["quality"])

    def test_chat_query_too_long_rejected(self):
        r = self.client.post("/api/v1/rag/chat/query", json={"query": "x" * (settings.security.MAX_QUERY_CHARS + 1)})
        self.assertEqual(r.status_code, 422)

    def test_chat_stream_ends_with_done(self):
        with self.client.stream("POST", "/api/v1/rag/chat/stream", json={"query": "测试流式", "max_retries": 0}) as r:
            self.assertEqual(r.status_code, 200)
            events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("quality", events[-1])
        self.assertIn("token", [e["type"] for e in events])  # 真流式：生成节点推送的 token


if __name__ == "__main__":
    unittest.main()
