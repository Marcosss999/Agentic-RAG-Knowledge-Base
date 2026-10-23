"""
Unit Tests for Hybrid & Mathematical Grounded Ragas Evaluator
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import unittest
from app.eval.ragas_evaluator import RagasEvaluator


class TestRagasEvaluator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.evaluator = RagasEvaluator()

    def test_math_context_recall_exact_and_partial(self):
        """测试上下文召回率的真实数学覆盖计算，严禁假静态常数"""
        query = "天勤量化如何订阅行情数据"
        ground_truth = "天勤量化通过 api.get_kline_serial 订阅实时K线，通过 quote 订阅实时报价。"
        
        # 1. 完全包含标准事实的上下文
        ctx_complete = [
            "在天勤量化中，开发者可以通过 api.get_kline_serial 函数订阅多周期K线数据，通过 quote 查看实时行情报价。"
        ]
        recall_full, reason_full = self.evaluator._compute_math_context_recall(
            query=query, contexts=ctx_complete, ground_truth=ground_truth
        )
        self.assertGreaterEqual(recall_full, 0.90)
        self.assertIn("100%", reason_full)

        # 2. 缺失部分关键事实的上下文
        ctx_partial = [
            "天勤量化平台支持期货行情展示，提供图表界面。"
        ]
        recall_part, reason_part = self.evaluator._compute_math_context_recall(
            query=query, contexts=ctx_partial, ground_truth=ground_truth
        )
        self.assertLess(recall_part, recall_full)
        self.assertIn("未命中", reason_part)

        # 3. 空上下文召回率必须严格为 0.0
        recall_empty, _ = self.evaluator._compute_math_context_recall(
            query=query, contexts=[], ground_truth=ground_truth
        )
        self.assertEqual(recall_empty, 0.0)

    def test_math_context_precision_map(self):
        """测试 MAP 上下文精准度加权平均计算"""
        query = "天勤量化 TargetPosTask 持仓控制"

        # 命中切片排在第一位 vs 排在第三位
        ctx_top1 = [
            "TargetPosTask 是天勤量化用于目标持仓自动化控制的核心组件，支持追单和轧差。",
            "Python 列表解析式是编写高可读性代码的重要手段。",
            "Linux 服务端进程守护可以通过 Supervisor 或 Systemd 托管。"
        ]
        score_top1, _ = self.evaluator._compute_math_context_precision(query, ctx_top1)

        ctx_bottom1 = [
            "Linux 服务端进程守护可以通过 Supervisor 或 Systemd 托管。",
            "Python 列表解析式是编写高可读性代码的重要手段。",
            "TargetPosTask 是天勤量化用于目标持仓自动化控制的核心组件，支持追单和轧差。"
        ]
        score_bottom1, _ = self.evaluator._compute_math_context_precision(query, ctx_bottom1)

        # 相关切片排在前面时，MAP 得分必须严格高于排在后面
        self.assertGreater(score_top1, score_bottom1)

    async def test_evaluate_sample_grounded_dynamic(self):
        """测试 evaluate_sample 端到端生成真实四维指标与审计底稿"""
        query = "天勤量化平台的核心特性是什么？"
        answer = "天勤量化提供全品种穿透式期货CTP行情接口，具备全自动回测和实盘持仓管理功能。"
        contexts = [
            "天勤量化（TqSdk）支持国内全品种期货实盘交易，具备穿透式CTP直连和高速目标持仓管控功能，内置高性能事件驱动回测引擎。"
        ]
        ground_truth = "天勤量化核心特性包括全品种期货CTP直连、自动化回测与实盘持仓管控。"

        metrics, reasons = await self.evaluator.evaluate_sample(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        self.assertGreater(metrics.faithfulness, 0.60)
        self.assertGreater(metrics.context_recall, 0.50)
        self.assertLessEqual(metrics.context_recall, 1.0)
        self.assertGreater(metrics.context_precision, 0.60)
        self.assertIn("context_recall_reason", reasons)
        self.assertIn("faithfulness_reason", reasons)


if __name__ == "__main__":
    unittest.main()
