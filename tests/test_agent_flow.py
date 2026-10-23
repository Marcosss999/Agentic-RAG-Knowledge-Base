"""
Self-RAG 状态机分支测试（缺陷追踪 AGENT-1 ~ AGENT-8、ENG-1）

原测试只检查"返回里有某个字段"，从来没有让 LLM 返回 irrelevant / FAIL，最核心的自省分支完全没被测到。
这里用一个可编排的假 LLM，逐条覆盖状态图里的每一条边。
"""

import sys
import asyncio
import unittest
from pathlib import Path
from unittest import mock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import app.agent.self_rag_graph as graph
from app.agent.self_rag_graph import parse_json_object, run_self_rag, summarize_quality
from app.agent.streaming import event_sink
from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.rag.bm25_index import BM25Index
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import VectorStoreManager, InMemoryVectorStore

DOC_A = "差旅报销制度：员工出差回来后需在 30 天内提交发票原件，经部门主管审批后由财务打款。" * 3
DOC_B = "年假制度：员工入职满一年享有 5 天带薪年假，需提前 3 个工作日在系统中申请。" * 3


class FakeLLM:
    """按提示词类型分派的假 LLM。每类可配置一个返回值序列，或抛出异常。"""

    def __init__(self, grade=None, audit=None, rewrite=None, answer=None, disambiguate=None):
        self.script = {"grade": list(grade or []), "audit": list(audit or []),
                       "rewrite": list(rewrite or []), "answer": list(answer or []),
                       "disambiguate": list(disambiguate or [])}
        self.calls = []        # (kind, messages)

    @staticmethod
    def kind_of(messages):
        system = messages[0]["content"]
        if "资料审核员" in system:
            return "grade"
        if "事实核查员" in system:
            return "audit"
        if "检索优化专家" in system:
            return "rewrite"
        if "查询重写专家" in system:
            return "disambiguate"
        return "answer"

    def _next(self, kind, messages):
        self.calls.append((kind, messages))
        seq = self.script[kind]
        value = seq.pop(0) if len(seq) > 1 else (seq[0] if seq else None)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return {"answer": "根据资料，差旅报销需在 30 天内提交发票原件 [1]。",
                    "disambiguate": messages[-1]["content"], "rewrite": "报销 发票",
                    "grade": '{"grade": "relevant", "reason": "资料覆盖问题"}',
                    "audit": '{"verdict": "pass", "unsupported_claims": []}'}[kind]
        return value

    async def chat(self, messages, temperature=None, **kw):
        return self._next(self.kind_of(messages), messages)

    async def chat_stream(self, messages, temperature=None, **kw):
        text = self._next(self.kind_of(messages), messages)
        for i in range(0, len(text), 3):
            yield text[i:i + 3]

    def kinds(self):
        return [k for k, _ in self.calls]


class AgentFlowTestBase(unittest.IsolatedAsyncioTestCase):
    docs = (("d_a", "差旅.txt", DOC_A), ("d_b", "年假.txt", DOC_B))
    llm_configured = True

    async def asyncSetUp(self):
        self.retriever = HybridRetriever(vector_store=VectorStoreManager(store=InMemoryVectorStore()), bm25=BM25Index())
        for doc_id, name, text in self.docs:
            await self.retriever.index_document(text, doc_id, name)
        self.external_calls = []

        async def fake_external(q):
            self.external_calls.append(q)
            return "外部百科摘要：这是一段足够长的外部公开资料内容，用于测试兜底分支。"

        for p in (
            mock.patch.object(graph, "hybrid_retriever", self.retriever),
            mock.patch.object(graph, "_fetch_external_web_knowledge", fake_external),
            mock.patch.object(llm_provider, "is_llm_configured", self.llm_configured),
        ):
            p.start()
            self.addCleanup(p.stop)

    def use_llm(self, fake: FakeLLM):
        for name in ("chat", "chat_stream"):
            p = mock.patch.object(llm_provider, name, getattr(fake, name))
            p.start()
            self.addCleanup(p.stop)
        import app.agent.query_transform as qt
        p = mock.patch.object(qt.query_transformer, "llm", llm_provider)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def steps(self, state):
        return [s["step"] for s in state["steps_timeline"]]


class TestJsonParsing(unittest.TestCase):
    def test_parse_variants(self):
        self.assertEqual(parse_json_object('{"grade": "relevant"}')["grade"], "relevant")
        self.assertEqual(parse_json_object('```json\n{"verdict": "fail"}\n```')["verdict"], "fail")
        self.assertEqual(parse_json_object('结论如下：{"grade":"irrelevant","reason":"无关"} 以上')["grade"], "irrelevant")
        self.assertIsNone(parse_json_object("**FAIL** 回答中有捏造"))
        self.assertIsNone(parse_json_object(""))


class TestGates(AgentFlowTestBase):
    async def test_happy_path(self):
        fake = self.use_llm(FakeLLM())
        st = await run_self_rag("差旅报销要多久内提交发票？")
        self.assertEqual(self.steps(st), ["transform_query", "retrieval", "grade_documents", "generate", "hallucination_check"])
        q = summarize_quality(st)
        self.assertEqual((q["retrieval_grade"], q["audit_status"], q["fallback_used"]), ("relevant", "pass", False))
        self.assertTrue(st["citations"])

    async def test_irrelevant_then_rewrite_then_relevant(self):
        fake = self.use_llm(FakeLLM(grade=['{"grade":"irrelevant","reason":"无关"}', '{"grade":"relevant","reason":"ok"}']))
        st = await run_self_rag("差旅报销有什么规定？", max_retries=2)
        self.assertEqual(self.steps(st).count("rewrite_query"), 1)
        self.assertEqual(st["retrieval_retries"], 1)
        self.assertEqual(st["generation_retries"], 0)
        self.assertEqual(st["transformed_query"], "报销 发票")

    async def test_rewrites_exhausted_go_to_fallback_and_clear_citations(self):
        """AGENT-5：兜底时清空过期引用；外部兜底默认关闭，不向第三方发送查询"""
        fake = self.use_llm(FakeLLM(grade=['{"grade":"irrelevant","reason":"无关"}']))
        st = await run_self_rag("公司股票期权怎么行权？", max_retries=2)
        self.assertEqual(self.steps(st).count("rewrite_query"), 2)
        self.assertIn("web_fallback", self.steps(st))
        self.assertEqual(st["citations"], [])
        self.assertEqual(self.external_calls, [])
        q = summarize_quality(st)
        self.assertTrue(q["fallback_used"])
        self.assertFalse(q["external_query_sent"])
        self.assertEqual(q["audit_status"], "skipped")
        answer_prompt = [m for k, m in fake.calls if k == "answer"][-1]
        self.assertIn("知识库中未找到相关资料", answer_prompt[0]["content"])

    async def test_external_fallback_uses_user_question_when_enabled(self):
        """AGENT-4：开启外部兜底时，发出去的是用户的问题，而不是 Mock 文本或改写词"""
        self.use_llm(FakeLLM(grade=['{"grade":"irrelevant","reason":"无关"}'], rewrite=["期权 行权 流程"]))
        with mock.patch.object(settings.rag, "WEB_FALLBACK_ENABLED", True):
            st = await run_self_rag("公司股票期权怎么行权？", max_retries=1)
        self.assertEqual(self.external_calls, ["公司股票期权怎么行权？"])
        self.assertTrue(summarize_quality(st)["external_query_sent"])

    async def test_unparsable_grade_is_uncertain_not_relevant(self):
        """AGENT-1 回归：原实现解析不出等级就默认 relevant"""
        self.use_llm(FakeLLM(grade=["我觉得还行吧"]))
        st = await run_self_rag("差旅报销要多久内提交发票？")
        self.assertEqual(st["retrieval_grade"], "uncertain")
        self.assertIn("generate", self.steps(st))

    async def test_grade_call_failure_is_uncertain(self):
        self.use_llm(FakeLLM(grade=[TimeoutError("llm timeout")]))
        st = await run_self_rag("差旅报销要多久内提交发票？")
        self.assertEqual(st["retrieval_grade"], "uncertain")
        self.assertIn("调用失败", st["grade_reason"])

    async def test_audit_fail_then_regenerate_then_pass(self):
        fake = self.use_llm(FakeLLM(
            audit=['{"verdict":"fail","unsupported_claims":["报销需 CEO 签字"]}', '{"verdict":"pass","unsupported_claims":[]}'],
            answer=["需在 30 天内提交，并由 CEO 签字 [1]。", "需在 30 天内提交发票原件 [1]。"]))
        st = await run_self_rag("差旅报销要多久内提交发票？")
        self.assertEqual(self.steps(st).count("generate"), 2)
        self.assertEqual(st["audit_status"], "pass")
        self.assertEqual(len(st["answer_attempts"]), 2)
        self.assertEqual(st["generation_retries"], 1)
        # 第二次生成的提示词里带有批评意见
        second_prompt = [m for k, m in fake.calls if k == "answer"][1]
        self.assertIn("报销需 CEO 签字", second_prompt[0]["content"])

    async def test_audit_keeps_failing_is_flagged_not_hidden(self):
        """重新生成次数用完后仍不通过：结束，但 audit_status=fail 会返回给用户（AGENT-6）"""
        self.use_llm(FakeLLM(audit=['{"verdict":"fail","unsupported_claims":["编造"]}']))
        st = await run_self_rag("差旅报销要多久内提交发票？")
        self.assertEqual(self.steps(st).count("generate"), 1 + settings.rag.GENERATION_MAX_RETRIES)
        self.assertEqual(summarize_quality(st)["audit_status"], "fail")
        self.assertFalse(st["hallucination_passed"])

    async def test_separate_retry_budgets(self):
        """AGENT-3 回归：检索改写用掉 2 次后，重新生成的机会不受影响"""
        self.use_llm(FakeLLM(
            grade=['{"grade":"insufficient","reason":"缺"}', '{"grade":"insufficient","reason":"缺"}', '{"grade":"relevant","reason":"ok"}'],
            audit=['{"verdict":"fail","unsupported_claims":["x"]}', '{"verdict":"pass"}']))
        st = await run_self_rag("差旅报销要多久内提交发票？", max_retries=2)
        self.assertEqual(st["retrieval_retries"], 2)
        self.assertEqual(st["generation_retries"], 1)
        self.assertEqual(st["audit_status"], "pass")

    async def test_markdown_wrapped_fail_is_not_treated_as_pass(self):
        """AGENT-1 回归：原实现只认以 FAIL 开头，"**FAIL**" 会被当成通过"""
        self.use_llm(FakeLLM(audit=["**FAIL** 回答中有资料没有的内容"]))
        st = await run_self_rag("差旅报销要多久内提交发票？")
        self.assertEqual(st["audit_status"], "uncertain")
        self.assertFalse(st["hallucination_passed"])

    async def test_gates_see_full_context(self):
        """AGENT-2 回归：原来闸门只看前 800 / 1000 字，看不到排在后面的资料"""
        fake = self.use_llm(FakeLLM())
        # 构造足够长的资料，保证上下文明显超过原来的 800 / 1000 字截断点
        for i, topic in enumerate(["差旅报销", "年假申请", "加班调休"]):
            text = "".join(f"{topic}制度第{j}条：员工办理{topic}时需要按流程提交材料并经主管审批。" for j in range(40))
            await self.retriever.index_document(text, f"long_{i}", f"{topic}.txt")
        st = await run_self_rag("差旅报销、年假申请和加班调休分别怎么规定？")
        ctx = st["formatted_context"]
        self.assertGreater(len(ctx), 1200)
        grade_prompt = [m for k, m in fake.calls if k == "grade"][0][1]["content"]
        audit_prompt = [m for k, m in fake.calls if k == "audit"][0][1]["content"]
        self.assertIn(ctx, grade_prompt)
        self.assertIn(ctx, audit_prompt)

    async def test_multi_turn_resolved_query_used_in_generation(self):
        """AGENT-7：第二轮问"它要多久内提交"时，生成阶段使用消解后的完整问题"""
        fake = self.use_llm(FakeLLM(disambiguate=["差旅报销的发票需要在多久内提交？"]))
        history = [{"role": "user", "content": "差旅报销需要什么材料？"},
                   {"role": "assistant", "content": "需要发票原件 [1]。"}]
        st = await graph.run_self_rag("它要多久内提交？", chat_history=history)
        self.assertEqual(st["resolved_query"], "差旅报销的发票需要在多久内提交？")
        answer_prompt = [m for k, m in fake.calls if k == "answer"][0][1]["content"]
        self.assertIn("差旅报销的发票需要在多久内提交？", answer_prompt)
        self.assertIn("它要多久内提交？", answer_prompt)


class TestAnswerLanguage(AgentFlowTestBase):
    async def test_prompt_requires_same_language(self):
        """AGENT-9：评测中发现英文问题得到中文回答、或明明资料写了却先说"资料中未说明"（p14、p26）"""
        fake = self.use_llm(FakeLLM())
        await run_self_rag("How long do I have to submit travel reimbursement?")
        system = [m for k, m in fake.calls if k == "answer"][0][0]["content"]
        self.assertIn("相同的语言", system)
        self.assertIn("不要先说未说明再引用", system)


class TestStreaming(AgentFlowTestBase):
    async def test_tokens_streamed_and_regenerate_event(self):
        """AGENT-8：生成节点边收 token 边推送；审计不通过时推送 regenerate 事件"""
        self.use_llm(FakeLLM(
            audit=['{"verdict":"fail","unsupported_claims":["编造"]}', '{"verdict":"pass"}'],
            answer=["第一版回答内容", "第二版回答内容"]))
        q: asyncio.Queue = asyncio.Queue()
        token = event_sink.set(q)
        try:
            st = await run_self_rag("差旅报销要多久内提交发票？")
        finally:
            event_sink.reset(token)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        types = [e["type"] for e in events]
        self.assertIn("token", types)
        self.assertIn("regenerate", types)
        self.assertEqual("".join(e["content"] for e in events if e["type"] == "token" and e["attempt"] == 1), "第一版回答内容")
        self.assertEqual([e["status"] for e in events if e["type"] == "audit"], ["fail", "pass"])
        self.assertLess(types.index("token"), types.index("regenerate"))


class TestEmptyKnowledgeBase(AgentFlowTestBase):
    docs = ()

    async def test_empty_kb_short_circuits_without_rewrite(self):
        """AGENT-4 回归：知识库为空时原实现会改写重试两轮（离线时还把 Mock 文本当检索词）"""
        fake = self.use_llm(FakeLLM())
        st = await run_self_rag("年假怎么申请？", max_retries=2)
        self.assertNotIn("rewrite_query", self.steps(st))
        self.assertNotIn("rewrite", fake.kinds())
        self.assertNotIn("grade", fake.kinds())       # 没有资料就不用让 LLM 评级
        self.assertTrue(summarize_quality(st)["kb_empty"])
        self.assertEqual(self.external_calls, [])


class TestOfflineMode(AgentFlowTestBase):
    llm_configured = False

    async def test_offline_skips_gates_honestly(self):
        """没有对话模型时：不做改写、不做评级/审计，但如实标注为 unverified"""
        st = await run_self_rag("差旅报销要多久内提交发票？", max_retries=2)
        self.assertNotIn("rewrite_query", self.steps(st))
        q = summarize_quality(st)
        self.assertEqual((q["retrieval_grade"], q["audit_status"]), ("unverified", "unverified"))


class TestQueryExpansionSwitches(AgentFlowTestBase):
    async def test_multi_query_and_hyde_reach_retriever(self):
        self.use_llm(FakeLLM())
        import app.agent.query_transform as qt
        captured = {}
        original = self.retriever.retrieve

        async def spy(**kwargs):
            captured.update(kwargs)
            return await original(**kwargs)

        with mock.patch.object(settings.rag, "ENABLE_MULTI_QUERY", True), \
             mock.patch.object(settings.rag, "ENABLE_HYDE", True), \
             mock.patch.object(qt.query_transformer, "generate_multi_queries", mock.AsyncMock(return_value=["报销时限", "发票提交期限"])), \
             mock.patch.object(qt.query_transformer, "generate_hyde_passage", mock.AsyncMock(return_value="员工应在规定期限内提交发票。")), \
             mock.patch.object(self.retriever, "retrieve", spy):
            await run_self_rag("差旅报销要多久内提交发票？")
        self.assertEqual(captured["extra_queries"], ["报销时限", "发票提交期限"])
        self.assertEqual(captured["dense_text"], "员工应在规定期限内提交发票。")

    async def test_switches_off_by_default(self):
        """不受本机 .env 影响：清掉相关环境变量后检查默认值"""
        import os
        from app.core.config import RAGHyperParams
        keys = ("ENABLE_MULTI_QUERY", "ENABLE_HYDE", "WEB_FALLBACK_ENABLED")
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            defaults = RAGHyperParams()
        self.assertFalse(defaults.ENABLE_MULTI_QUERY)
        self.assertFalse(defaults.ENABLE_HYDE)
        self.assertFalse(defaults.WEB_FALLBACK_ENABLED)


if __name__ == "__main__":
    unittest.main()
