"""
Unit Tests for LangGraph Self-RAG State Machine & Reflection Workflow
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from app.agent.state import create_initial_state
from app.agent.self_rag_graph import run_self_rag
from unittest import mock
import app.agent.self_rag_graph as graph
from app.rag.bm25_index import BM25Index
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import VectorStoreManager, InMemoryVectorStore


class TestSelfRAGGraph(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # 预先构建一个测试知识切片
        doc_text = "天勤量化（TqSdk）支持行情订阅、K线获取、目标持仓任务 TargetPosTask 以及 CTP 直连交易。"
        # ENG-1：每个测试用独立的检索器，并替换状态机使用的检索器，不污染全局单例
        hybrid_retriever = HybridRetriever(vector_store=VectorStoreManager(store=InMemoryVectorStore()), bm25=BM25Index())
        patcher = mock.patch.object(graph, "hybrid_retriever", hybrid_retriever)
        patcher.start()
        self.addCleanup(patcher.stop)
        await hybrid_retriever.index_document(
            raw_text=doc_text,
            doc_id="test_self_rag_doc",
            doc_name="tq_features.txt"
        )

    def test_initial_state_structure(self):
        """测试 Self-RAG 初始状态模型构造规范"""
        state = create_initial_state(
            query="天勤量化怎么做目标持仓控制",
            session_id="test_sess_01",
            max_retries=2
        )
        self.assertEqual(state["original_query"], "天勤量化怎么做目标持仓控制")
        self.assertEqual(state["retry_count"], 0)
        self.assertEqual(state["max_retries"], 2)
        self.assertEqual(state["retrieval_grade"], "init")
        self.assertEqual(state["hallucination_passed"], True)

    async def test_self_rag_pipeline_execution(self):
        """测试 Self-RAG 端到端状态机运行与生成回复"""
        final_state = await run_self_rag(
            query="天勤量化如何管理持仓？",
            session_id="test_sess_pipeline",
            max_retries=1
        )

        self.assertIn("generated_answer", final_state)
        self.assertIn("steps_timeline", final_state)
        self.assertGreater(len(final_state["steps_timeline"]), 0)

        # 检查时间线中包含核心执行步骤
        step_names = [s.get("step") for s in final_state["steps_timeline"]]
        self.assertIn("transform_query", step_names)
        self.assertIn("retrieval", step_names)


if __name__ == "__main__":
    unittest.main()
