"""
节点内事件推送通道 (Streaming Sink)

AGENT-8：原来的"流式"是整张图跑完后，再把完整答案每 4 个字切一段假装逐字输出。
真流式需要在 generate 节点内部，一边接收 LLM 的 token，一边推给 SSE 接口。

做法：用 ContextVar 保存一个 asyncio.Queue。
- SSE 接口在启动图之前设置它；asyncio 创建任务时会复制当前上下文，所以图里的节点能拿到同一个队列；
- 同步接口 (/chat/query) 不设置，节点发现没有队列就只生成、不推送。
这个方案不依赖某个特定版本 LangGraph 的流式 API，换框架也能用。
"""

import asyncio
from contextvars import ContextVar
from typing import Any, Dict, Optional

event_sink: ContextVar[Optional[asyncio.Queue]] = ContextVar("rag_event_sink", default=None)


async def emit(event: Dict[str, Any]) -> None:
    """有订阅者时推送事件，没有时什么都不做"""
    sink = event_sink.get()
    if sink is not None:
        await sink.put(event)


def streaming_enabled() -> bool:
    return event_sink.get() is not None
