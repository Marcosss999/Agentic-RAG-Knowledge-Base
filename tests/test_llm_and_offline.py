"""
Unit Tests for Subword Random Projection Embeddings & Offline Reranker
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
import numpy as np
from app.core.llm_provider import llm_provider, _compute_semantic_projection_embedding


class TestLLMAndOffline(unittest.IsolatedAsyncioTestCase):
    async def test_subword_embedding_geometry(self):
        """离线 1024 维哈希特征向量：维度、单位范数、确定性，以及词面重叠越多越相近"""
        q_target = "天勤量化交易开发接口规范与API"
        q_similar = "天勤量化Python开发SDK接口"
        q_dissimilar = "川味红烧牛肉面的地道烹饪秘方"

        vec_target = await llm_provider.get_embedding(q_target)
        vec_similar = await llm_provider.get_embedding(q_similar)
        vec_dissimilar = await llm_provider.get_embedding(q_dissimilar)

        # 1. 维度与单位范数校验
        self.assertEqual(len(vec_target), 1024)
        norm_target = np.linalg.norm(vec_target)
        self.assertAlmostEqual(norm_target, 1.0, places=4)

        # 2. 确定性测试
        vec_target_repeat = await llm_provider.get_embedding(q_target)
        self.assertTrue(np.allclose(vec_target, vec_target_repeat))

        # 3. 语义几何距离校验: 相关语义的余弦相似度必须明显高于无关文本
        cos_similar = float(np.dot(vec_target, vec_similar))
        cos_dissimilar = float(np.dot(vec_target, vec_dissimilar))

        # 离线哈希向量只反映词面重叠（RET-1）：共享"天勤量化""开发""接口"等词的文本应明显更近。
        # 原来写死的 0.35 是针对旧哈希函数调出来的魔法数，这里改为比较相对差距。
        self.assertGreater(cos_similar, cos_dissimilar + 0.2)

    async def test_hash_seed_has_no_prefix_collisions(self):
        """RET-1 回归：原种子只取 token 的前 4 个字节，前缀相同的词得到完全相同的向量"""
        from app.core.llm_provider import _compute_semantic_projection_embedding as emb
        for a, b in [("retrieval", "retrieve"), ("embedding", "embed"), ("information", "informal")]:
            self.assertLess(float(np.dot(emb(a), emb(b))), 0.99, msg=f"{a} vs {b}")

    def test_offline_signature_versioned(self):
        """哈希函数变了，离线向量签名必须变化，才能触发已有索引重建"""
        if not llm_provider.is_embed_configured:
            self.assertEqual(llm_provider.embedding_signature, "offline-hash-v2:1024")

    async def test_offline_reranker_cross_scoring(self):
        """测试离线重排多信号加权打分 (Jaccard + 实体重合 + 语义投影)"""
        query = "天勤量化支持哪些期货交易功能"
        docs = [
            "天勤量化（TqSdk）支持国内全品种期货实盘交易，具备穿透式CTP直连和高速目标持仓管控功能。",
            "Python 是一种广泛使用的高级编程语言，常用于网络爬虫与数据可视化分析。",
            "天勤量化提供全自动回测与模拟盘撮合功能，满足期货量化策略多阶段验证要求。"
        ]

        scored = await llm_provider.rerank(query=query, documents=docs, top_n=3)
        self.assertEqual(len(scored), 3)

        # 第 0 项与第 2 项关于天勤量化的文档得分必须高于完全不相干的 Python 爬虫文档
        rel_indices = [item["index"] for item in scored[:2]]
        self.assertIn(0, rel_indices)
        self.assertIn(2, rel_indices)
        self.assertEqual(scored[-1]["index"], 1)

    async def test_connectivity_probe_structure(self):
        """测试网关连通性探测接口返回数据结构规范"""
        res = await llm_provider.test_connectivity()
        self.assertIn("llm", res)
        self.assertIn("embedding", res)
        self.assertIn("reranker", res)
        self.assertIn("status", res["llm"])


if __name__ == "__main__":
    unittest.main()
