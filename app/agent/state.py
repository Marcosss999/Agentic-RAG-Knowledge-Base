"""
LangGraph 状态定义 (Agentic RAG State)
职责：
1. 定义状态图中所有节点之间传递的数据通道。
2. 定义自省决策变量：检索评级、审计结论、两套独立的重试计数、兜底标记。
3. steps_timeline 使用 Annotated[List, operator.add] 累加器，多个节点追加时间线事件不会互相覆盖。
"""

import operator
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from app.models.schemas import SearchResultItem, CitationReference


class AgenticRAGState(TypedDict):
    # 1. 会话与输入
    session_id: str
    original_query: str
    chat_history: List[Dict[str, str]]

    # 2. 查询理解
    resolved_query: str                    # 结合对话历史消解指代后的独立问题（生成阶段使用，AGENT-7）
    transformed_query: str                 # 当前用于检索的查询（自省改写会更新它）
    multi_queries: List[str]               # 多查询扩展（ENABLE_MULTI_QUERY 开启时）
    hyde_passage: Optional[str]            # HyDE 假设性回答（ENABLE_HYDE 开启时）
    is_complex: bool

    # 3. 检索结果
    retrieved_docs: List[SearchResultItem]
    formatted_context: str
    citations: List[CitationReference]
    kb_empty: bool                         # 知识库为空：直接短路，不做无意义的改写重试（AGENT-4）

    # 4. 闸门一：检索评级
    # relevant / insufficient / irrelevant：LLM 给出的结论
    # uncertain：LLM 调用失败或输出无法解析（不再默认当作 relevant，AGENT-1）
    # unverified：离线模式没有 LLM，未做评级
    retrieval_grade: str
    grade_reason: str

    # 5. 两套独立的重试计数（AGENT-3：原来共用一个 retry_count，检索改写会挤占重新生成的机会）
    retrieval_retries: int                 # 已做的检索改写次数
    max_retries: int                       # 检索改写上限
    generation_retries: int                # 因审计不通过而重新生成的次数
    max_generation_retries: int            # 重新生成上限
    retry_count: int                       # = retrieval_retries + generation_retries，仅用于展示与兼容

    # 6. 兜底
    web_fallback_needed: bool              # 是否进入了"知识库无答案"的兜底分支
    web_fallback_external: bool            # 兜底时是否真的向外部发送了查询

    # 7. 生成与闸门二：幻觉审计
    generated_answer: str
    answer_attempts: List[str]             # 每一次生成的回答（重新生成时保留上一版，便于前端展示与排查）
    # pass / fail：LLM 给出的结论；uncertain：审计调用失败或无法解析；
    # skipped：兜底模式不做私有库比对；unverified：离线模式没有 LLM
    audit_status: str
    hallucination_passed: bool             # audit_status == "pass"（兼容旧字段）
    hallucination_critique: str

    # 8. 时间线（累加器通道）
    steps_timeline: Annotated[List[Dict[str, Any]], operator.add]

    # 9. 错误
    error_message: Optional[str]


def create_initial_state(
    query: str,
    session_id: str = "default_session",
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_retries: int = 2,
    max_generation_retries: Optional[int] = None,
) -> AgenticRAGState:
    """创建初始状态"""
    from app.core.config import settings
    q = query.strip()
    return {
        "session_id": session_id,
        "original_query": q,
        "chat_history": chat_history or [],
        "resolved_query": q,
        "transformed_query": q,
        "multi_queries": [],
        "hyde_passage": None,
        "is_complex": False,
        "retrieved_docs": [],
        "formatted_context": "",
        "citations": [],
        "kb_empty": False,
        "retrieval_grade": "init",
        "grade_reason": "",
        "retrieval_retries": 0,
        "max_retries": max_retries,
        "generation_retries": 0,
        "max_generation_retries": (settings.rag.GENERATION_MAX_RETRIES
                                   if max_generation_retries is None else max_generation_retries),
        "retry_count": 0,
        "web_fallback_needed": False,
        "web_fallback_external": False,
        "generated_answer": "",
        "answer_attempts": [],
        "audit_status": "init",
        "hallucination_passed": True,
        "hallucination_critique": "",
        "steps_timeline": [],
        "error_message": None,
    }
