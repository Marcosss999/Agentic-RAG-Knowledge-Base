"""
后台入库任务 (CONC-2)

原来上传接口要等"解析 → 切分 → 向量化 → 建索引"全部做完才返回。大文件（例如 343 个子块的论文，
在线向量化约 35 次请求）会让浏览器一直转圈，代理也可能因超时断开连接。

现在支持后台模式：接口先做同步校验（类型、大小、doc_id），通过后立即返回任务 ID（HTTP 202），
入库在后台进行；前端轮询 GET /documents/tasks/{task_id} 显示阶段与进度。

说明：任务状态只保存在内存里（重启后丢失），最多保留最近 100 个；单进程部署够用，多进程部署需要
换成 Redis 等外部存储，并配合任务队列（如 Celery / RQ）。
"""

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

MAX_TASKS = 100


class IngestTaskManager:
    def __init__(self):
        self._tasks: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._handles: Dict[str, asyncio.Task] = {}

    def submit(self, label: str, job: Callable[[Callable[[str, int, int], None]], Awaitable[Dict[str, Any]]]) -> str:
        """job 接收一个进度回调 progress(stage, done, total)，返回入库结果"""
        task_id = uuid.uuid4().hex[:12]
        info = {"task_id": task_id, "label": label, "status": "queued", "stage": "排队中",
                "done": 0, "total": 0, "result": None, "error": None,
                "created_at": time.time(), "finished_at": None}
        self._tasks[task_id] = info
        while len(self._tasks) > MAX_TASKS:
            old_id, old = self._tasks.popitem(last=False)
            self._handles.pop(old_id, None)

        def progress(stage: str, done: int, total: int) -> None:
            info.update(stage=stage, done=done, total=total)

        async def runner():
            info["status"] = "running"
            try:
                info["result"] = await job(progress)
                info["status"] = "done"
                info["stage"] = "完成"
            except Exception as e:  # 失败原因交给前端展示
                info["status"] = "failed"
                info["error"] = str(e) or type(e).__name__
                logger.error(f"[IngestTask] {label} 入库失败: {e}", exc_info=True)
            finally:
                info["finished_at"] = time.time()
                self._handles.pop(task_id, None)

        self._handles[task_id] = asyncio.get_running_loop().create_task(runner())
        return task_id

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        info = self._tasks.get(task_id)
        return dict(info) if info else None

    async def wait(self, task_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """测试用：等待任务结束"""
        handle = self._handles.get(task_id)
        if handle is not None:
            await asyncio.wait_for(asyncio.shield(handle), timeout)
        return self.get(task_id)


ingest_tasks = IngestTaskManager()
