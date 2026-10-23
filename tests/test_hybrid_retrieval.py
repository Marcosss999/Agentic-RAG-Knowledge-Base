"""
Unit Tests for BM25, RRF Fusion & Hybrid Retrieval
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from app.models.schemas import ChildChunk, SearchResultItem, RetrievalSource
from app.rag.bm25_index import BM25Index
from app.rag.rrf_fusion import RRFFusion
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import VectorStoreManager, InMemoryVectorStore


class TestHybridRetrieval(unittest.IsolatedAsyncioTestCase):
    def test_bm25_indexing_and_search(self):
        """测试 Okapi BM25 稀疏检索倒排索引构建与查询"""
        bm25 = BM25Index()
        chunks = [
            ChildChunk(
                chunk_id="c1",
                doc_id="d1",
                doc_name="finance.txt",
                parent_id="p1",
                content="天勤量化平台支持期货行情订阅与CTP穿透式直连交易。",
                chunk_index=0
            ),
            ChildChunk(
                chunk_id="c2",
                doc_id="d1",
                doc_name="finance.txt",
                parent_id="p1",
                content="Python 深度学习框架 PyTorch 广泛应用于自然语言处理和图像识别。",
                chunk_index=1
            )
        ]
        bm25.build_index(chunks)
        results = bm25.search(query="天勤量化 CTP", top_k=2)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertGreater(results[0].score, 0.0)

    def test_rrf_fusion_ranking(self):
        """测试 RRF (Reciprocal Rank Fusion) 倒数名次融合加权算法"""
        fusion = RRFFusion(k=60)
        
        # 模拟路1 (Dense) 召回结果 [c1, c2, c3]
        list1 = [
            SearchResultItem(chunk_id="c1", doc_id="d1", doc_name="t1", parent_id="p1", content="t1", score=0.9, source=RetrievalSource.DENSE_VECTOR),
            SearchResultItem(chunk_id="c2", doc_id="d1", doc_name="t1", parent_id="p1", content="t2", score=0.8, source=RetrievalSource.DENSE_VECTOR),
            SearchResultItem(chunk_id="c3", doc_id="d1", doc_name="t1", parent_id="p1", content="t3", score=0.7, source=RetrievalSource.DENSE_VECTOR),
        ]
        # 模拟路2 (Sparse) 召回结果 [c2, c1, c4]
        list2 = [
            SearchResultItem(chunk_id="c2", doc_id="d1", doc_name="t1", parent_id="p1", content="t2", score=5.0, source=RetrievalSource.SPARSE_BM25),
            SearchResultItem(chunk_id="c1", doc_id="d1", doc_name="t1", parent_id="p1", content="t1", score=4.0, source=RetrievalSource.SPARSE_BM25),
            SearchResultItem(chunk_id="c4", doc_id="d1", doc_name="t1", parent_id="p1", content="t4", score=3.0, source=RetrievalSource.SPARSE_BM25),
        ]

        fused = fusion.fuse(ranked_lists=[list1, list2], weights=[1.0, 1.0], top_n=3)
        self.assertEqual(len(fused), 3)

        # c1 和 c2 均在两路的前列，融合得分必然高于只在单路出现的 c3 和 c4
        top_two_ids = {fused[0].chunk_id, fused[1].chunk_id}
        self.assertEqual(top_two_ids, {"c1", "c2"})

    async def test_end_to_end_hybrid_index_and_retrieve(self):
        """测试端到端摄取文档、构建混合索引并检索回填母块上下文"""
        doc_text = (
            "天勤量化（TqSdk）平台架构分为三层：\n"
            "第一层为底层 CTP 网关通讯，支持实时行情与申报回报的秒级处理。\n"
            "第二层为数据模型层，提供 Quote 与 Kline 序列的高性能缓存。\n"
            "第三层为策略执行层，提供 TargetPosTask 自动对冲与头寸平抑能力。"
        )
        # ENG-1：使用独立实例，不往全局单例里写数据，避免测试之间互相污染
        hybrid_retriever = HybridRetriever(vector_store=VectorStoreManager(store=InMemoryVectorStore()), bm25=BM25Index())
        parents_cnt, children_cnt = await hybrid_retriever.index_document(
            raw_text=doc_text,
            doc_id="test_suite_doc",
            doc_name="tqsdk_architecture.txt"
        )
        self.assertGreater(parents_cnt, 0)
        self.assertGreater(children_cnt, 0)

        # 执行混合检索
        results = await hybrid_retriever.retrieve(
            query="TargetPosTask 自动对冲",
            top_k=2,
            enable_rerank=True,
            enable_hybrid=True
        )
        self.assertGreater(len(results), 0)
        self.assertIn("TargetPosTask", results[0].content)


if __name__ == "__main__":
    unittest.main()
