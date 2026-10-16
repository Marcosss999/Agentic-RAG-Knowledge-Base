"""
LangGraph 自省状态机全局契约 (Agentic RAG State Definition)
职责：
1. 作为状态图 (StateGraph) 中所有节点间传递的“唯一定义与单一事实来源 (Single Source of Truth)”。
2. 规范定义自省决策变量（检索评价、重试计数器、幻觉审计标志、回退搜索控制）。
3. 利用 Annotated[List, operator.add] 状态累加器通道，原生支持节点执行时间线 (Timeline Trace) 的有序累加，不丢步。
"""

import operator
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from app.models.schemas import SearchResultItem, CitationReference


class AgenticRAGState(TypedDict):
    """
    自省式 RAG 状态契约 (Self-RAG State Channels)
    每一个键值都是 LangGraph 图调度中流动的数据通道
    """

    # 1. 基础会话与入参通道
    session_id: str
    original_query: str
    chat_history: List[Dict[str, str]]

    # 2. 查询理解与重塑通道
    transformed_query: str                 # 指代消解后的清晰检索Query
    multi_queries: List[str]               # 多查询扩展列表 (用于并发多路拉取)
    is_complex: bool                       # 是否属于复合多跳问题

    # 3. 混合检索与重排结果通道
    retrieved_docs: List[SearchResultItem] # 经过 RRF 融合与重排后的精选片段列表
    formatted_context: str                 # 格式化后注入给大模型的母块全文段落
    citations: List[CitationReference]     # 结构化引用证据列表 (前端高亮渲染)

    # 4. Agent 自省判定与控制信号通道 (Reflection & Control Flags)
    retrieval_grade: str                   # 检索质量评价: "relevant"(达标) / "insufficient"(部分相关) / "irrelevant"(不相关)
    grade_reason: str                      # 质量评级判定依据
    retry_count: int                       # 循环重试/改写计数器 (严控上限，防止无限死循环)
    max_retries: int                       # 允许最大重写检索次数 (默认 2 次)
    web_fallback_needed: bool              # 是否触发外部搜索引擎兜底降级

    # 5. 生成与幻觉自审通道 (Generation & Hallucination Audit)
    generated_answer: str                  # 大模型当前生成的回答候选
    hallucination_passed: bool             # 忠实度 (Faithfulness) 审核是否通过: True/False
    hallucination_critique: str            # 发现的幻觉/捏造疑点说明 (用于指导重写生成)

    # 6. 可观测性与执行轨迹时间线 (利用 operator.add 支持多节点无冲突增量追加)
    steps_timeline: Annotated[List[Dict[str, Any]], operator.add]

    # 7. 异常与错误控制通道
    error_message: Optional[str]


def create_initial_state(
    query: str,
    session_id: str = "default_session",
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_retries: int = 2
) -> AgenticRAGState:
    """创建并初始化标准状态字典的辅助构建函数"""
    return {
        "session_id": session_id,
        "original_query": query.strip(),
        "chat_history": chat_history or [],
        "transformed_query": query.strip(),
        "multi_queries": [query.strip()],
        "is_complex": False,
        "retrieved_docs": [],
        "formatted_context": "",
        "citations": [],
        "retrieval_grade": "init",
        "grade_reason": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "web_fallback_needed": False,
        "generated_answer": "",
        "hallucination_passed": True,
        "hallucination_critique": "",
        "steps_timeline": [],
        "error_message": None
    }
