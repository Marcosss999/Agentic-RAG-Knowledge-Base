"""
Unit Tests for RecursiveTextSplitter & ParentChildSplitter
"""

import sys
from pathlib import Path

# 将项目根目录置入 sys.path，支持任何路径下独立运行测试
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from app.rag.splitter import RecursiveTextSplitter, ParentChildSplitter


class TestSplitter(unittest.TestCase):
    def test_recursive_splitter_basic(self):
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "天勤量化平台（TqSdk）是一款专为期货及金融衍生品研发的高性能Python交易开发框架。\n\n" \
               "该平台支持全自动程序化交易，内建完善的穿透式监管CTP行情接入模块。\n\n" \
               "此外，天勤支持纳秒级事件驱动回测引擎与毫秒级实盘订单路由。"
        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 65)  # chunk_size with delimiter tolerance

    def test_splitter_invalid_overlap(self):
        with self.assertRaises(ValueError):
            RecursiveTextSplitter(chunk_size=50, chunk_overlap=50)

    def test_parent_child_splitter(self):
        splitter = ParentChildSplitter(
            parent_chunk_size=150,
            parent_chunk_overlap=20,
            child_chunk_size=50,
            child_chunk_overlap=10
        )
        sample_doc = (
            "天勤量化交易系统核心组件：\n"
            "1. TqApi：主交互网关，负责初始化账户、连接期货公司服务器并订阅实时K线与行情数据。\n"
            "2. TargetPosTask：智能目标持仓控制函数，能够自动化执行追单、拆单与多空头寸轧差。\n"
            "3. Backtest：本地历史仿真回测组件，支持Tick级别逐笔历史行情回放与滑点手续费计算。\n"
            "4. WebGui：提供本地可视化行情K线图表与实时持仓盈亏曲线监控。"
        )
        parents, children = splitter.process_document(
            raw_text=sample_doc,
            doc_id="test_doc_01",
            doc_name="test_tianqin.txt"
        )

        self.assertGreater(len(parents), 0)
        self.assertGreater(len(children), len(parents))

        # 验证每个 child 的 parent_id 均可在 parents 中找到
        parent_id_set = {p.chunk_id for p in parents}
        for child in children:
            self.assertIn(child.parent_id, parent_id_set)
            self.assertEqual(child.doc_id, "test_doc_01")

        # 验证 parent 中的 child_ids 包含对应的 child
        all_child_ids_in_parents = []
        for p in parents:
            all_child_ids_in_parents.extend(p.child_ids)
        for child in children:
            self.assertIn(child.chunk_id, all_child_ids_in_parents)


if __name__ == "__main__":
    unittest.main()
