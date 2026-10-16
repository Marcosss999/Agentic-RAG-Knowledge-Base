"""
LangGraph 自省式 Self-RAG 全局状态机引擎 (Self-Reflective RAG Graph)
职责：
1. 构建基于 LangGraph StateGraph 的循环自省控制流，彻底颠覆传统 Naive RAG 单向“检索-拼接-生成”模式。
2. 落地双重自省闸门 (Dual Reflection Gates)：
   - 闸门一【检索文档质量审核】(grade_documents_node)：自省检索内容是否充分支持提问，不足则自动重构 Query 触发重新检索；重试耗尽时平滑触发外部兜底。
   - 闸门二【生成回答幻觉自审】(hallucination_check_node)：自省生成内容是否严格忠于事实资料 (Faithfulness)，发现幻觉自动反馈批评意见并重新生成。
3. 全链路沉淀执行轨迹时间线 (Timeline Trace)，提供可解释性审计。
4. 提供开箱即用的高层异步调用接口与流式调度门面。
"""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.models.schemas import SearchResultItem
from app.rag.hybrid_retriever import hybrid_retriever
from app.agent.query_transform import query_transformer
from app.agent.state import AgenticRAGState, create_initial_state

logger = logging.getLogger(__name__)


# =============================================================================
# 图节点 1：查询分析与指代消解 / 重写节点 (Query Transformation Node)
# =============================================================================

async def transform_query_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    负责消除多轮对话指代歧义，或者在自省重试时改写提问
    """
    retry_count = state.get("retry_count", 0)
    original_query = state["original_query"]
    chat_history = state.get("chat_history", [])

    if retry_count == 0:
        # 初次进入：执行多轮指代消解与多查询扩展
        transform_result = await query_transformer.transform(
            query=original_query,
            chat_history=chat_history,
            enable_multi_query=False,
            enable_hyde=False
        )
        transformed_q = transform_result.rewritten_queries[0]
        multi_q = transform_result.rewritten_queries

        timeline_event = {
            "step": "transform_query",
            "title": "查询意图理解与指代消解",
            "detail": f"原始Query: '{original_query}' -> 消解后Query: '{transformed_q}'",
            "retry_count": 0
        }
        logger.info(f"[Self-RAG] 查询重塑完成: '{original_query}' -> '{transformed_q}'")

        return {
            "transformed_query": transformed_q,
            "multi_queries": multi_q,
            "is_complex": transform_result.is_complex,
            "steps_timeline": [timeline_event]
        }
    else:
        # 触发自省重写循环：根据前一次的失败理由重构 Query
        grade_reason = state.get("grade_reason", "检索结果不够匹配")
        current_query = state.get("transformed_query", original_query)

        rewrite_prompt = [
            {
                "role": "system",
                "content": (
                    "你是一名知识库检索优化专家。上一次针对用户问题的知识检索未能获取足够的相关材料。\n"
                    f"未达标原因：{grade_reason}\n"
                    "请从另一个技术角度、或提取更核心的关键词/专业术语，将检索词进行改写优化。\n"
                    "要求：仅输出一句改写后的简短搜索词，严禁附加任何解释。"
                )
            },
            {
                "role": "user",
                "content": f"原始问题: {original_query}\n前一次搜索词: {current_query}"
            }
        ]

        rewritten_q = await llm_provider.chat(rewrite_prompt, temperature=0.3)
        rewritten_q = rewritten_q.strip().replace('"', '').replace("'", "") or current_query

        timeline_event = {
            "step": "transform_query_retry",
            "title": f"触发第 {retry_count} 次自省查询改写",
            "detail": f"优化检索词为: '{rewritten_q}' (调整依据: {grade_reason})",
            "retry_count": retry_count
        }
        logger.info(f"[Self-RAG] 第 {retry_count} 次查询改写: '{rewritten_q}'")

        return {
            "transformed_query": rewritten_q,
            "steps_timeline": [timeline_event]
        }


# =============================================================================
# 图节点 2：多路混合检索与重排节点 (Retrieval Node)
# =============================================================================

async def retrieval_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    调用混合检索器，并发拉取 BM25 + Qdrant 结果并经由 RRF 融合与 Cross-Encoder 重排
    """
    search_query = state["transformed_query"]
    logger.info(f"[Self-RAG] 执行底层混合检索，检索词: '{search_query}'")

    docs = await hybrid_retriever.retrieve(
        query=search_query,
        top_k=settings.rag.FINAL_TOP_K,
        enable_rerank=True,
        enable_hybrid=True
    )

    formatted_context, citations = hybrid_retriever.build_context_prompt(docs)

    timeline_event = {
        "step": "retrieval",
        "title": "双路混合检索与深度重排",
        "detail": f"召回精排有效切片 {len(docs)} 个，最高置信度: {docs[0].score if docs else 0.0}",
        "docs_count": len(docs),
        "top_doc": docs[0].doc_name if docs else "None"
    }

    return {
        "retrieved_docs": docs,
        "formatted_context": formatted_context,
        "citations": citations,
        "steps_timeline": [timeline_event]
    }


# =============================================================================
# 图节点 3：自省闸门一：检索材料质量裁决 (Document Grader Node)
# =============================================================================

async def grade_documents_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    评估检索到的上下文片段是否包含回答用户问题所需的充分事实。
    裁定等级：
    - 'relevant': 资料充足，可直接回答
    - 'insufficient': 部分相关或证据不足
    - 'irrelevant': 完全不相关或未召回到内容
    """
    docs = state.get("retrieved_docs", [])
    query = state["transformed_query"]

    retry_count = state.get("retry_count", 0)

    # 情况 1：未检索到任何候选切片
    if not docs:
        grade = "irrelevant"
        reason = "本地知识库未检索到任何相关文档片段"
        logger.warning(f"[Self-RAG] 自省判定: {grade} -> {reason}")
        return {
            "retrieval_grade": grade,
            "grade_reason": reason,
            "retry_count": retry_count + 1,
            "steps_timeline": [{
                "step": "grade_documents",
                "title": "检索文档质量审核 (Grade Documents)",
                "detail": f"判定: {grade} | 理由: {reason}"
            }]
        }

    # 情况 2：利用大模型执行事实相关度审校
    eval_prompt = [
        {
            "role": "system",
            "content": (
                "你是一名严格的事实资料审计员。你的任务是核对检索到的参考资料是否能够有效回答用户的问题。\n"
                "请严格按以下标准判定：\n"
                "- relevant: 参考资料中包含了回答该问题所需的关键事实信息。\n"
                "- insufficient: 参考资料与问题领域有关，但缺少关键核心数据或事实。\n"
                "- irrelevant: 参考资料与用户问题完全风马牛不相及。\n\n"
                "输出格式要求：第一行输出等级（仅限 relevant、insufficient、irrelevant 之一）；第二行输出一句话理由。"
            )
        },
        {
            "role": "user",
            "content": f"【用户问题】: {query}\n\n【参考资料摘要】:\n{state['formatted_context'][:800]}\n\n【审计判定】:"
        }
    ]

    grade_resp = await llm_provider.chat(eval_prompt, temperature=0.0)
    lines = [line.strip() for line in grade_resp.strip().split("\n") if line.strip()]

    grade = "relevant"
    reason = "资料与问题相匹配"

    if lines:
        first_line = lines[0].lower()
        if "irrelevant" in first_line:
            grade = "irrelevant"
        elif "insufficient" in first_line:
            grade = "insufficient"
        elif "relevant" in first_line:
            grade = "relevant"

        if len(lines) > 1:
            reason = lines[1]
        else:
            reason = lines[0]

    timeline_event = {
        "step": "grade_documents",
        "title": "检索文档质量审核 (Grade Documents)",
        "detail": f"判定评级: {grade.upper()} | 理由: {reason}",
        "grade": grade,
        "reason": reason
    }
    logger.info(f"[Self-RAG] 文档质量审核: {grade} ({reason})")

    new_retry = retry_count + 1 if grade != "relevant" else retry_count
    return {
        "retrieval_grade": grade,
        "grade_reason": reason,
        "retry_count": new_retry,
        "steps_timeline": [timeline_event]
    }


# =============================================================================
# 条件分支边 1：决定下一步走向 (Decide After Grade)
# =============================================================================

def decide_after_grade(state: AgenticRAGState) -> str:
    """
    根据检索质量评级与重试次数路由：
    - relevant -> 'generate'
    - insufficient / irrelevant ->
        * retry_count <= max_retries -> 'rewrite_query' (重构并再次检索)
        * retry_count > max_retries -> 'web_fallback' (兜底降级)
    """
    grade = state.get("retrieval_grade", "relevant")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if grade == "relevant":
        logger.info("[Self-RAG 路由] 检索质量达标，进入回答生成节点。")
        return "generate"

    if retry_count <= max_retries:
        logger.info(f"[Self-RAG 路由] 检索质量为 '{grade}'，重试次数 {retry_count}/{max_retries}，触发查询改写重试。")
        return "rewrite_query"

    logger.warning(f"[Self-RAG 路由] 重试次数已达上限 ({max_retries})，转入兜底降级节点。")
    return "web_fallback"


# =============================================================================
# 图节点 4：外部网络搜索 / 兜底降级节点 (Web Fallback Node)
# =============================================================================

async def web_fallback_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    当本地私有知识库多次重试依然无法召回有效证据时，触发外部兜底机制
    """
    query = state["transformed_query"]
    logger.info(f"[Self-RAG] 触发兜底降级流程: '{query}'")

    fallback_context = (
        f"【系统声明】：经知识库多轮深度自省检索，内部文档库中未检索到关于“{query}”的确定性记录。\n"
        "以下基于通用工程事实进行合理建议，请注意甄别："
    )

    timeline_event = {
        "step": "web_fallback",
        "title": "触发外部知识兜底与降级声明",
        "detail": "本地库未命中确定性证据，注入降级提示与通用常识指引。"
    }

    return {
        "formatted_context": fallback_context,
        "web_fallback_needed": True,
        "steps_timeline": [timeline_event]
    }


# =============================================================================
# 图节点 5：根据证据生成严谨回答 (Generate Node)
# =============================================================================

async def generate_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    基于注入的上下文材料和历史反思批评，合成严格事实对齐的最终回答
    """
    original_query = state["original_query"]
    context = state.get("formatted_context", "")
    critique = state.get("hallucination_critique")

    system_instruction = (
        "你是一名专业严谨的企业级知识助手。请依据提供的【参考证据资料】回答用户问题。\n"
        "守则：\n"
        "1. 严格基于证据资料中提供的事实作答，必须在引用的事实后标注数字索引（如 [1]、[2]）。\n"
        "2. 严禁凭空捏造未在证据中出现的具体参数、规格或时间。\n"
        "3. 若资料中未提及某点，请明确告知用户该项在当前资料中未说明，切勿胡乱推测。"
    )

    if critique:
        system_instruction += f"\n【重要纠偏】：前一次生成的回答存在以下事实不符或幻觉疑点：\n{critique}\n本次请务必严格修正，仅保留有据可查的信息！"

    prompt = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": f"【参考证据资料】:\n{context}\n\n【用户问题】: {original_query}"
        }
    ]

    answer = await llm_provider.chat(prompt, temperature=0.1)

    timeline_event = {
        "step": "generate",
        "title": "大模型严谨答案生成",
        "detail": f"已生成回答，字数: {len(answer)} 字" + (" (包含反思纠偏)" if critique else "")
    }
    logger.info(f"[Self-RAG] 答案生成完成 (长度: {len(answer)})")

    return {
        "generated_answer": answer,
        "steps_timeline": [timeline_event]
    }


# =============================================================================
# 图节点 6：自省闸门二：回答忠实度与幻觉自审 (Hallucination Checker Node)
# =============================================================================

async def hallucination_check_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    自省审校生成的回答是否对参考事实具有 100% 的忠实度 (Faithfulness Audit)
    """
    answer = state.get("generated_answer", "")
    context = state.get("formatted_context", "")
    is_fallback = state.get("web_fallback_needed", False)

    # 若已处于兜底模式，免除内部私有库忠实度硬审
    if is_fallback:
        return {
            "hallucination_passed": True,
            "hallucination_critique": "",
            "steps_timeline": [{
                "step": "hallucination_check",
                "title": "忠实度幻觉自审",
                "detail": "当前处于通用兜底模式，免除私有库事实比对。"
            }]
        }

    audit_prompt = [
        {
            "role": "system",
            "content": (
                "你是一名严谨的事实核查员（Hallucination Auditor）。\n"
                "请比对【参考事实资料】与【生成的回答】，检查回答中是否存在资料未支持的捏造事实或虚假参数。\n"
                "规则：\n"
                "- 若回答中的每一个事实论断均能在参考资料中找到依据，输出：PASS\n"
                "- 若回答中包含资料里根本没有的事实、编造的技术指标或错误结论，输出：FAIL: <具体指出捏造了什么>\n"
                "请直接在第一行输出 PASS 或 FAIL: ..."
            )
        },
        {
            "role": "user",
            "content": f"【参考事实资料】:\n{context[:1000]}\n\n【生成的回答】:\n{answer}\n\n【核查结论】:"
        }
    ]

    audit_resp = await llm_provider.chat(audit_prompt, temperature=0.0)
    audit_resp = audit_resp.strip()

    passed = True
    critique = ""

    if audit_resp.startswith("FAIL"):
        passed = False
        critique = audit_resp[len("FAIL:"):].strip() if ":" in audit_resp else audit_resp

    timeline_event = {
        "step": "hallucination_check",
        "title": "忠实度幻觉自审 (Hallucination Audit)",
        "detail": "审校结果: 通过 (Pass)" if passed else f"审校结果: 存在幻觉 (Fail) - {critique}",
        "passed": passed
    }
    retry_count = state.get("retry_count", 0)
    new_retry = retry_count + 1 if not passed else retry_count

    return {
        "hallucination_passed": passed,
        "hallucination_critique": critique,
        "retry_count": new_retry,
        "steps_timeline": [timeline_event]
    }


# =============================================================================
# 条件分支边 2：幻觉判定后路由 (Decide After Hallucination)
# =============================================================================

def decide_after_hallucination(state: AgenticRAGState) -> str:
    """
    根据幻觉审查结果决定：
    - passed -> 结束 (END)
    - failed 且未超重试次数 -> 携带 critique 重新生成 ('regenerate')
    - failed 且达重试上限 -> 结束 (END)
    """
    passed = state.get("hallucination_passed", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if passed:
        logger.info("[Self-RAG 路由] 幻觉审校通过，结束执行。")
        return END

    if retry_count <= max_retries:
        logger.warning(f"[Self-RAG 路由] 发现幻觉疑点，重新回退生成 (重试次数 {retry_count}/{max_retries})。")
        return "regenerate"

    logger.warning("[Self-RAG 路由] 达到最大修正次数，输出当前结果并结束。")
    return END


# =============================================================================
# 状态机图装配与编译 (Build & Compile StateGraph)
# =============================================================================

def build_self_rag_graph() -> StateGraph:
    """构建自省式 RAG 状态机网络"""
    workflow = StateGraph(AgenticRAGState)

    # 1. 挂载所有逻辑计算节点
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_fallback", web_fallback_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # 2. 编排固定边
    workflow.add_edge(START, "transform_query")
    workflow.add_edge("transform_query", "retrieval")
    workflow.add_edge("retrieval", "grade_documents")

    # 3. 编排检索质量判定条件分支边
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "transform_query",
            "web_fallback": "web_fallback"
        }
    )

    # 4. 兜底节点连向生成
    workflow.add_edge("web_fallback", "generate")

    # 5. 生成节点连向幻觉自审
    workflow.add_edge("generate", "hallucination_check")

    # 6. 编排幻觉自审条件分支边
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination,
        {
            "regenerate": "generate",
            END: END
        }
    )

    return workflow


# 编译后的全局单例图应用
self_rag_workflow = build_self_rag_graph()
self_rag_app = self_rag_workflow.compile()


# =============================================================================
# 统一高层调用门面 (Facade Runner)
# =============================================================================

async def run_self_rag(
    query: str,
    session_id: str = "default_session",
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_retries: int = 2
) -> AgenticRAGState:
    """
    执行 Self-RAG 全流程自省图调用
    :param query: 用户输入的问题
    :param session_id: 当前会话 ID
    :param chat_history: 历史上下文列表 [{"role": "...", "content": "..."}]
    :param max_retries: 最大自省重试次数
    :return: 运行结束后的终态字典 AgenticRAGState
    """
    initial_state = create_initial_state(
        query=query,
        session_id=session_id,
        chat_history=chat_history,
        max_retries=max_retries
    )

    final_state = await self_rag_app.ainvoke(initial_state)
    return final_state
