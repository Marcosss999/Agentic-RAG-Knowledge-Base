"""
RAG 评测命令行入口

用法（在项目根目录）：
    python -m experiments.rag_eval                      # 只跑检索评测（较快，约几分钟）
    python -m experiments.rag_eval --mode full          # 检索 + 端到端（每题真跑 Self-RAG 并由判官打分）
    python -m experiments.rag_eval --mode full --judge-repeats 2   # 判官重复两次，报告一致率
    python -m experiments.rag_eval --mode full --limit 5           # 先用 5 道题试跑

结果写入 data/eval/rag_eval_<模式>_<时间>.json 与同名 .md。
说明：会调用 .env 中配置的对话、向量、重排模型（真实接口，会产生少量费用）；不会修改网页上的知识库。
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.eval.pipeline import DEFAULT_DATASET, run_evaluation, save_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RAG 评测")
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--judge-repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="端到端只跑前 N 题")
    ap.add_argument("--no-llm-configs", action="store_true", help="检索评测跳过多查询扩展与 HyDE（不调用对话模型）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    start = time.time()

    def progress(msg):
        print(f"[{time.time() - start:6.1f}s] {msg}", flush=True)

    report = asyncio.run(run_evaluation(
        mode=args.mode, dataset_path=Path(args.dataset), include_llm_configs=not args.no_llm_configs,
        judge_repeats=args.judge_repeats, limit=args.limit, progress=progress))
    stem = f"rag_eval_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    jp, mp = save_report(report, settings.storage.EVAL_DATASET_DIR, stem)
    print(f"\n完成，用时 {time.time() - start:.0f} 秒。\n报告：{mp}\n数据：{jp}")
    for w in report["warnings"]:
        print("注意：" + w)


if __name__ == "__main__":
    main()
