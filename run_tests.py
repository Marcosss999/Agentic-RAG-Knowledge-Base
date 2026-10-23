"""
一键全量测试套件 (Automated Test Suite Runner)
功能：
1. 自动注册项目根目录至 sys.path。
2. 自动发现 tests/ 目录下全部 test_*.py 单元与集成测试用例。
3. 统计执行用例数、成功/失败与耗时。
4. 提供美观的终端控制台摘要输出，方便 CI/CD 流水线与交付验证。
"""

import os
import sys
import time
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 屏蔽/隔离外部 API Key 干扰，统一验证本地确定性与降级容灾逻辑
os.environ["LLM_API_KEY"] = ""
os.environ["EMBEDDING_API_KEY"] = ""
os.environ["RERANKER_API_KEY"] = ""


def run():
    print("=" * 70)
    print("  Enterprise Agentic RAG Platform - Automated Verification Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(project_root / "tests"),
        pattern="test_*.py",
        top_level_dir=str(project_root)
    )

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = round(time.time() - start_time, 2)

    print("\n" + "=" * 70)
    print(f"[TEST SUMMARY] 执行总用例: {result.testsRun} 个 | 耗时: {elapsed}s")
    if result.wasSuccessful():
        print(f"[STATUS: PASSED] 全部通过：0 错误, 0 失败, {len(result.skipped)} 跳过。")
        print("=" * 70)
        return 0
    else:
        print(f"[STATUS: FAILED] 失败: {len(result.failures)} 个, 错误: {len(result.errors)} 个")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run())
