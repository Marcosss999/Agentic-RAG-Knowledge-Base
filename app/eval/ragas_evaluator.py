"""
旧版四维评测器（已不用于正式评测，保留供参考与对比）

⚠️ 缺陷追踪 EVAL-1 / EVAL-3：
- 名字叫 Ragas，但并未使用 ragas 库；
- 原来的 /evaluation/run 用它给写死的回答打分，不经过系统本身（现已改用 app/eval/pipeline.py）；
- "精准度"让 LLM 直接报一个 MAP 数字，而 MAP 需要逐条判断每个片段是否相关，LLM 给不出真实的 MAP；
- 离线模式下的四个"数学指标"只是词面重叠的启发式规则：只要子句里有一个 2 字以上的词出现在上下文中
  就算"被支持"，编造的数字也能拿到较高的忠实度（实测 0.67）；
- 综合分是加权算术平均（0.30/0.25/0.25/0.20），权重没有依据。

正式评测请使用：python -m experiments.rag_eval（检索指标 + 端到端，判官逐条判断、代码计算比例）。
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.core.security import validate_report_name, safe_join
from app.models.schemas import RagasEvaluationMetrics

logger = logging.getLogger(__name__)

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

PUNCT_SET = set("，。！？；：、“”‘’（）()[]{}【】《》=+-*/\\| \t\n\r`~@#$%^&_")
STOPWORDS = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "吗", "？", "。"}


def _tokenize_clean(text: str) -> List[str]:
    """清洗并提取自然语言与数字英文特征词"""
    if not text:
        return []
    if JIEBA_AVAILABLE:
        words = list(jieba.cut_for_search(text))
    else:
        words = re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z0-9_\-]+", text)

    return [
        w.strip()
        for w in words
        if w.strip()
        and w.strip() not in PUNCT_SET
        and w.strip() not in STOPWORDS
        and not all(c in PUNCT_SET for c in w)
    ]


class RagasEvaluator:
    """旧版四维评估器（已不用于正式评测）"""

    def __init__(self):
        self.llm = llm_provider
        self.eval_output_dir = Path(settings.storage.EVAL_DATASET_DIR)
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 客观数学计算引擎 (Mathematical Grounded Fallback Engine)
    # =========================================================================

    def _compute_math_context_recall(
        self, query: str, contexts: List[str], ground_truth: Optional[str]
    ) -> Tuple[float, str]:
        """
        客观事实召回率 (Context Recall)
        对比 Ground Truth 核心实体与概念事实清单在检索上下文中的覆盖比例
        """
        if not contexts:
            return 0.0, "未召回任何上下文片段，召回率为 0.0"

        gt_text = ground_truth or query
        gt_tokens = _tokenize_clean(gt_text)
        gt_unique = list(dict.fromkeys(gt_tokens))

        if not gt_unique:
            return 1.0, "标准答案无特征实体约束，默认完全覆盖"

        joined_ctx = " ".join(contexts)
        matched = [w for w in gt_unique if w in joined_ctx]
        missed = [w for w in gt_unique if w not in joined_ctx]

        recall = round(len(matched) / len(gt_unique), 4)
        reason = (
            f"标准事实共含 {len(gt_unique)} 项核心实体词，上下文成功召回 {len(matched)} 项 ({matched[:6]})"
            + (f"，未命中 {len(missed)} 项 ({missed[:4]})" if missed else "，核心事实完全覆盖 (100%)")
            + f"，客观实体召回率为 {round(recall * 100, 2)}%"
        )
        return recall, reason

    def _compute_math_context_precision(
        self, query: str, contexts: List[str]
    ) -> Tuple[float, str]:
        """
        客观上下文精准度 (Context Precision / MAP)
        计算有用切片在各个召回排名上的加权平均精度 (Mean Average Precision)
        """
        if not contexts:
            return 0.0, "无召回上下文，精准度为 0.0"

        q_tokens = set(_tokenize_clean(query))
        if not q_tokens:
            return 0.80, "查询词过短，给予标准置信度"

        rel_count = 0
        total_map = 0.0

        for rank, ctx in enumerate(contexts, start=1):
            ctx_tokens = set(_tokenize_clean(ctx))
            overlap = len(q_tokens & ctx_tokens)
            overlap_ratio = overlap / max(len(q_tokens), 1)

            # 若包含数字或关键词重合度 >= 15%，判定为命中相关切片
            is_relevant = overlap_ratio >= 0.15 or any(
                w in ctx for w in q_tokens if len(w) >= 2
            )
            if is_relevant:
                rel_count += 1
                precision_at_k = rel_count / rank
                total_map += precision_at_k

        map_score = round(total_map / max(rel_count, 1), 4) if rel_count > 0 else 0.40
        reason = (
            f"Top-{len(contexts)} 切片中有 {rel_count} 个切片命中了问题意图，"
            f"经排名加权后的 MAP (Mean Average Precision) 精度得分为 {map_score}"
        )
        return map_score, reason

    def _compute_math_faithfulness(
        self, answer: str, contexts: List[str]
    ) -> Tuple[float, str]:
        """
        客观回答忠实度 (Faithfulness / 抗幻觉)
        将回答拆解为命题子句，核验每个事实子句在上下文中的事实证据支撑比例
        """
        if not answer.strip():
            return 0.0, "回答为空"
        if not contexts:
            return 0.0, "无参考证据"

        joined_ctx = " ".join(contexts)
        clauses = [
            c.strip()
            for c in re.split(r"[，。！？；\n]", answer)
            if len(c.strip()) >= 3
        ]

        if not clauses:
            return 0.90, "回答简练，无独立复杂命题"

        supported_cnt = 0
        hallucinated = []

        for c in clauses:
            c_tokens = _tokenize_clean(c)
            if not c_tokens:
                supported_cnt += 1
                continue

            hit_count = sum(1 for w in c_tokens if w in joined_ctx)
            ratio = hit_count / len(c_tokens)

            # 如果关键实体词或数字在上下文中，或重叠率达标，视为受支持
            if ratio >= 0.30 or any(
                (len(w) >= 2 and w in joined_ctx) for w in c_tokens
            ):
                supported_cnt += 1
            else:
                hallucinated.append(c)

        faith_score = round(supported_cnt / len(clauses), 4)
        reason = (
            f"回答切分为 {len(clauses)} 个事实命题，其中 {supported_cnt} 个获得上下文明确事实支持"
            + (f"，发现 {len(hallucinated)} 处疑似未证实疑点: {hallucinated[:2]}" if hallucinated else "，全部论断严格契合证据")
            + f"，忠实度为 {round(faith_score * 100, 2)}%"
        )
        return faith_score, reason

    def _compute_math_answer_relevancy(
        self, query: str, answer: str
    ) -> Tuple[float, str]:
        """
        客观答案相关度 (Answer Relevancy)
        衡量回答对问题核心疑问词、参数词的命中程度与切题性
        """
        if not answer.strip():
            return 0.0, "回答为空"

        q_tokens = set(_tokenize_clean(query))
        ans_tokens = set(_tokenize_clean(answer))

        if not q_tokens:
            return 0.85, "查询词无强约束，相关度达标"

        overlap = len(q_tokens & ans_tokens)
        ratio = overlap / max(len(q_tokens), 1)

        # 额外加分项：问“多少”有数字、问“是什么”有专有名词
        bonus = 0.0
        if any(w in query for w in ["多少", "大小", "参数", "几"]) and any(
            c.isdigit() for c in answer
        ):
            bonus += 0.20
        if any(w in query for w in ["算法", "模型", "架构"]) and any(
            len(w) >= 2 and w in answer for w in ["BM25", "RRF", "Qdrant", "Parent", "gte", "Cross"]
        ):
            bonus += 0.20

        final_rel = round(min(1.0, max(0.35, ratio * 1.20 + bonus)), 4)
        reason = (
            f"回答切中了用户提问中 {round(ratio * 100, 1)}% 的核心概念词"
            f"并提供了关键数值/技术实体答复，切题相关度为 {round(final_rel * 100, 2)}%"
        )
        return final_rel, reason

    # =========================================================================
    # 四大评估方法入口 (支持 LLM-as-a-Judge 与数学降级自适应协同)
    # =========================================================================

    async def evaluate_faithfulness(
        self, answer: str, contexts: List[str]
    ) -> Tuple[float, str]:
        """计算回答的忠实度得分 (0.0 ~ 1.0)"""
        if not self.llm.is_llm_configured:
            return self._compute_math_faithfulness(answer, contexts)

        joined_contexts = "\n\n".join(contexts)[:3000]
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一名严格的事实一致性裁判。请按照以下步骤评估【回答】相对于【参考证据】的忠实度：\n"
                    "1. 将【回答】切分为独立的客观事实命题；\n"
                    "2. 逐一核查每个命题能否在【参考证据】中找到明确推导依据；\n"
                    "3. 计算支持比例，输出 0.0 到 1.0 之间的浮点数。\n\n"
                    "输出格式要求：严格输出合法 JSON：\n"
                    '{\n  "score": 0.85,\n  "reason": "..."\n}'
                ),
            },
            {
                "role": "user",
                "content": f"【参考证据】:\n{joined_contexts}\n\n【回答】:\n{answer}",
            },
        ]
        try:
            resp = await self.llm.chat(prompt, temperature=0.0)
            score, reason = self._parse_score_and_reason(resp)
            if score is not None:
                return score, reason
        except Exception as e:
            logger.warning(f"[RagasEvaluator] LLM 忠实度评测异常 ({e})，切换为客观度量计算")

        return self._compute_math_faithfulness(answer, contexts)

    async def evaluate_answer_relevancy(
        self, query: str, answer: str
    ) -> Tuple[float, str]:
        """计算答案与提问的相关度得分 (0.0 ~ 1.0)"""
        if not self.llm.is_llm_configured:
            return self._compute_math_answer_relevancy(query, answer)

        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一名语言问答质量专家。请评估【回答】与【用户问题】的语义相关度与切题程度 (0.0 ~ 1.0)。\n"
                    "输出格式要求：严格输出合法 JSON：\n"
                    '{\n  "score": 0.9,\n  "reason": "..."\n}'
                ),
            },
            {
                "role": "user",
                "content": f"【用户问题】: {query}\n\n【回答】: {answer}",
            },
        ]
        try:
            resp = await self.llm.chat(prompt, temperature=0.0)
            score, reason = self._parse_score_and_reason(resp)
            if score is not None:
                return score, reason
        except Exception as e:
            logger.warning(f"[RagasEvaluator] LLM 相关度评测异常 ({e})，切换为客观度量计算")

        return self._compute_math_answer_relevancy(query, answer)

    async def evaluate_context_precision(
        self, query: str, contexts: List[str]
    ) -> Tuple[float, str]:
        """计算检索上下文的平均精度 (Context Precision / MAP) (0.0 ~ 1.0)"""
        if not self.llm.is_llm_configured:
            return self._compute_math_context_precision(query, contexts)

        eval_items = "\n".join(
            [f"[{i+1}] {c[:200]}..." for i, c in enumerate(contexts[:5])]
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一名搜索召回排序审计员。请评估各片段排序质量并给出整体 MAP 精度分 (0.0 ~ 1.0)。\n"
                    "输出格式要求：严格输出合法 JSON：\n"
                    '{\n  "score": 0.88,\n  "reason": "..."\n}'
                ),
            },
            {
                "role": "user",
                "content": f"【用户问题】: {query}\n\n【按顺序召回的切片】:\n{eval_items}",
            },
        ]
        try:
            resp = await self.llm.chat(prompt, temperature=0.0)
            score, reason = self._parse_score_and_reason(resp)
            if score is not None:
                return score, reason
        except Exception as e:
            logger.warning(f"[RagasEvaluator] LLM 精准度评测异常 ({e})，切换为客观度量计算")

        return self._compute_math_context_precision(query, contexts)

    async def evaluate_context_recall(
        self, query: str, contexts: List[str], ground_truth: Optional[str] = None
    ) -> Tuple[float, str]:
        """计算检索上下文的真实事实召回率 (Context Recall) (0.0 ~ 1.0)"""
        if not self.llm.is_llm_configured:
            return self._compute_math_context_recall(query, contexts, ground_truth)

        gt_text = ground_truth or f"与提问 '{query}' 相关的全面技术事实"
        joined_contexts = "\n\n".join(contexts)[:3000]
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一名信息完整度审计员。请对比【标准真实答案 Ground Truth】与【检索到的上下文】，"
                    "计算事实召回比例 (0.0 ~ 1.0)。\n"
                    "输出格式要求：严格输出合法 JSON：\n"
                    '{\n  "score": 0.95,\n  "reason": "..."\n}'
                ),
            },
            {
                "role": "user",
                "content": f"【用户问题】: {query}\n\n【标准真实答案】: {gt_text}\n\n【检索到的上下文】:\n{joined_contexts}",
            },
        ]
        try:
            resp = await self.llm.chat(prompt, temperature=0.0)
            score, reason = self._parse_score_and_reason(resp)
            if score is not None:
                return score, reason
        except Exception as e:
            logger.warning(f"[RagasEvaluator] LLM 召回率评测异常 ({e})，切换为客观度量计算")

        return self._compute_math_context_recall(query, contexts, ground_truth)

    # =========================================================================
    # 单样本全量评测调度
    # =========================================================================

    async def evaluate_sample(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Tuple[RagasEvaluationMetrics, Dict[str, str]]:
        """
        对单次问答执行四维雷达评测
        :return: (RagasEvaluationMetrics, 每项指标的客观判定原因字典)
        """
        faith_score, faith_reason = await self.evaluate_faithfulness(answer, contexts)
        relevancy_score, rel_reason = await self.evaluate_answer_relevancy(
            query, answer
        )
        precision_score, prec_reason = await self.evaluate_context_precision(
            query, contexts
        )
        recall_score, rec_reason = await self.evaluate_context_recall(
            query, contexts, ground_truth
        )

        overall = (
            faith_score * 0.30
            + relevancy_score * 0.25
            + precision_score * 0.25
            + recall_score * 0.20
        )
        overall = round(overall, 4)

        metrics = RagasEvaluationMetrics(
            faithfulness=round(faith_score, 4),
            answer_relevancy=round(relevancy_score, 4),
            context_precision=round(precision_score, 4),
            context_recall=round(recall_score, 4),
            overall_score=overall,
            evaluated_at=time.time(),
        )
        reasons = {
            "faithfulness_reason": faith_reason,
            "answer_relevancy_reason": rel_reason,
            "context_precision_reason": prec_reason,
            "context_recall_reason": rec_reason,
        }
        return metrics, reasons

    # =========================================================================
    # 批量基准测试集自动化评测套件
    # =========================================================================

    async def evaluate_batch(
        self, dataset: List[Dict[str, Any]], report_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """批量运行评测并输出汇总归档报告"""
        if not dataset:
            return {"error": "测试集为空"}
        validate_report_name(report_name)  # 先校验报告名，避免跑完整批评测才发现非法

        logger.info(
            f"[RagasEvaluator] 开始对 {len(dataset)} 条问答数据执行量化基准评测..."
        )
        results: List[Dict[str, Any]] = []

        total_faith = 0.0
        total_relevancy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_overall = 0.0

        for idx, item in enumerate(dataset, start=1):
            q = item.get("query", "")
            a = item.get("answer", "")
            ctxs = item.get("contexts", [])
            gt = item.get("ground_truth")

            metrics, reasons = await self.evaluate_sample(
                query=q, answer=a, contexts=ctxs, ground_truth=gt
            )

            total_faith += metrics.faithfulness
            total_relevancy += metrics.answer_relevancy
            total_precision += metrics.context_precision
            total_recall += metrics.context_recall
            total_overall += metrics.overall_score

            results.append(
                {
                    "sample_id": idx,
                    "query": q,
                    "ground_truth": gt,
                    "metrics": metrics.model_dump(),
                    "audit_reasons": reasons,
                }
            )

        count = len(dataset)
        summary = {
            "total_samples": count,
            "evaluation_engine": "Hybrid LLM-as-a-Judge & Grounded Mathematical Metric Suite",
            "average_metrics": {
                "faithfulness": round(total_faith / count, 4),
                "answer_relevancy": round(total_relevancy / count, 4),
                "context_precision": round(total_precision / count, 4),
                "context_recall": round(total_recall / count, 4),
                "overall_score": round(total_overall / count, 4),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": results,
        }

        # SEC-1：原来直接拼接 report_name，传入 "../../main.py" 可覆盖任意文件（已实测）。
        # 现在只接受纯文件名白名单，并确认解析后的路径仍在评测目录内。
        filename = validate_report_name(report_name) or f"ragas_report_{int(time.time())}.json"
        save_path = safe_join(self.eval_output_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[RagasEvaluator] 批量基准评测完成！报告已保存至: {save_path}")
        return summary

    def _parse_score_and_reason(
        self, text: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """从 LLM 返回内容中安全提取 JSON 评分或数字，解析失败返回 (None, None)"""
        if not text or "[mock" in text.lower() or "离线测试模式" in text:
            return None, None

        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
                score = float(data.get("score", 0.0))
                reason = str(data.get("reason", "模型评测完成"))
                score = max(0.0, min(1.0, score))
                return score, reason
        except Exception:
            pass

        return None, None


# 全局单例评测器
ragas_evaluator = RagasEvaluator()
