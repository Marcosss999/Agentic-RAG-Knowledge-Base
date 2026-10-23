"""
API Key 复用规则回归测试（缺陷追踪 API-6）

原问题：Embedding / Reranker 的 Key 留空时
- 运行中在网页填写对话模型 Key，向量与重排不会跟着复用（界面却写着"留空则自动复用"）；
- 重启后又无条件复用：对话模型是 DeepSeek、向量模型是阿里云百炼时，拿 DeepSeek 的 Key 调百炼，必然 401。
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.llm_provider import llm_provider

DEEPSEEK = "https://api.deepseek.com"
DASHSCOPE_EMB = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_RERANK = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class TestKeyResolution(unittest.TestCase):
    def _configure(self, llm_key, llm_url, emb_key, emb_url, rr_key="", rr_url=DASHSCOPE_RERANK):
        patches = [
            mock.patch.object(settings.llm, "API_KEY", llm_key),
            mock.patch.object(settings.llm, "BASE_URL", llm_url),
            mock.patch.object(settings.embedding, "API_KEY", emb_key),
            mock.patch.object(settings.embedding, "BASE_URL", emb_url),
            mock.patch.object(settings.reranker, "API_KEY", rr_key),
            mock.patch.object(settings.reranker, "BASE_URL", rr_url),
        ]
        # addCleanup 按后进先出执行：先登记的 reload_config 最后执行，即在所有补丁还原之后重载配置，
        # 否则测试用的假 Key 会残留在全局 llm_provider 里，污染后续测试
        self.addCleanup(llm_provider.reload_config)
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        llm_provider.reload_config()

    def test_different_provider_does_not_inherit(self):
        """对话用 DeepSeek、向量/重排用百炼：不能把 DeepSeek 的 Key 发给百炼"""
        self._configure("sk-deepseek", DEEPSEEK, "", DASHSCOPE_EMB)
        self.assertTrue(llm_provider.is_llm_configured)
        self.assertFalse(llm_provider.is_embed_configured)
        self.assertFalse(llm_provider.is_rerank_configured)
        self.assertEqual(llm_provider.embed_api_key, "")

    def test_same_provider_inherits(self):
        self._configure("sk-dashscope", DASHSCOPE_EMB, "", DASHSCOPE_EMB)
        self.assertTrue(llm_provider.is_embed_configured)
        self.assertTrue(llm_provider.embed_key_inherited)
        self.assertEqual(llm_provider.embed_api_key, "sk-dashscope")
        self.assertTrue(llm_provider.is_rerank_configured)  # 重排接口同为 dashscope.aliyuncs.com

    def test_own_key_wins_and_placeholder_ignored(self):
        self._configure("sk-deepseek", DEEPSEEK, "sk-emb", DASHSCOPE_EMB, rr_key="your_api_key_here")
        self.assertEqual(llm_provider.embed_api_key, "sk-emb")
        self.assertFalse(llm_provider.embed_key_inherited)
        self.assertFalse(llm_provider.is_rerank_configured)

    def test_runtime_update_propagates(self):
        """运行中更新对话模型 Key 后，同服务商的向量模型立即可用（原实现要重启才生效）"""
        self._configure("", DASHSCOPE_EMB, "", DASHSCOPE_EMB)
        self.assertFalse(llm_provider.is_embed_configured)
        settings.llm.API_KEY = "sk-new"   # 相当于 save_env_settings 更新了对话模型 Key（_configure 的补丁会还原）
        llm_provider.reload_config()
        self.assertTrue(llm_provider.is_embed_configured)


class TestDenseFailureDegrades(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_error_falls_back_to_bm25(self):
        from app.rag.bm25_index import BM25Index
        from app.rag.hybrid_retriever import HybridRetriever
        from app.rag.vector_store import VectorStoreManager, InMemoryVectorStore
        r = HybridRetriever(vector_store=VectorStoreManager(store=InMemoryVectorStore()), bm25=BM25Index())
        await r.index_document("报销需要提供发票原件。" * 10, "d1", "a.txt")
        with mock.patch.object(llm_provider, "get_embedding", side_effect=RuntimeError("401 invalid api key")):
            hits = await r.retrieve("报销发票", enable_rerank=False)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "d1")


if __name__ == "__main__":
    unittest.main()
