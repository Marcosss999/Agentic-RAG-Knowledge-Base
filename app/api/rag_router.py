"""
FastAPI 业务路由层 (API Router)

接口：
1. 文档管理：上传 (/documents/upload)、列表 (GET /documents)、删除 (DELETE /documents/{doc_id})。
2. 同步问答 (/chat/query)：调度 LangGraph Self-RAG 状态机，返回答案、引用、耗时与执行时间线。
3. 流式问答 (/chat/stream)：SSE 推送节点事件，并逐 token 推送回答（真流式）。
4. 评测 (/evaluation/run)、状态 (/status)、模型配置 (/settings) 与连通性测试 (/settings/test)。

安全措施（缺陷追踪 SEC-1 ~ SEC-6）：
- 管理类接口（修改配置、删除文档、触发评测、连通性测试）需要管理权限：
  配置了 ADMIN_TOKEN 时必须带 X-Admin-Token 请求头；未配置时只接受本机且来源可信的请求。
- 上传：doc_id 白名单、文件类型白名单、大小上限、服务端生成存储文件名。
- 问答：问题长度与对话历史上限、按 IP 限流、整次请求总超时。
"""

import asyncio
import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings, save_env_settings
from app.core.llm_provider import llm_provider
from app.core.security import (
    SecurityError, RateLimiter, check_admin_access, validate_base_url, validate_report_name,
)
from app.models.schemas import ChatQueryResponse
from app.agent.state import create_initial_state
from app.agent.self_rag_graph import run_self_rag, self_rag_app, summarize_quality, GRAPH_CONFIG
from app.agent.streaming import event_sink
from app.rag.knowledge_base import knowledge_base, DocumentNotFound, DuplicateContentError
from app.rag.loaders import DocumentParseError
from app.rag.ingest_tasks import ingest_tasks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["Enterprise Agentic RAG Platform"])

chat_limiter = RateLimiter(settings.security.RATE_LIMIT_CHAT_PER_MIN)
upload_limiter = RateLimiter(settings.security.RATE_LIMIT_UPLOAD_PER_MIN)

SSE_HEARTBEAT_SECONDS = 10.0


# =============================================================================
# 通用辅助
# =============================================================================

def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _require_admin(request: Request) -> None:
    """管理接口访问控制（SEC-4）"""
    try:
        check_admin_access(
            client_host=request.client.host if request.client else None,
            origin=request.headers.get("origin"),
            provided_token=request.headers.get("x-admin-token"),
            admin_token=settings.security.ADMIN_TOKEN,
            allowed_origins=settings.security.CORS_ORIGINS,
        )
    except SecurityError as e:
        code = status.HTTP_401_UNAUTHORIZED if str(e) == "invalid_token" else status.HTTP_403_FORBIDDEN
        messages = {
            "invalid_token": "需要有效的管理令牌（请求头 X-Admin-Token）",
            "admin_local_only": "未配置 ADMIN_TOKEN 时，管理接口只允许本机访问",
            "forbidden_origin": "请求来源不在允许列表中",
        }
        raise HTTPException(status_code=code, detail=messages.get(str(e), "无权访问"))


def _check_rate(limiter: RateLimiter, request: Request) -> None:
    if not limiter.allow(_client_ip(request)):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="请求过于频繁，请稍后再试")


# =============================================================================
# 请求与响应模型
# =============================================================================

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    parent_chunks: int = Field(..., description="母块数量")
    child_chunks: int = Field(..., description="子块数量（写入向量库与 BM25）")
    replaced: bool = Field(default=False, description="是否替换了同 doc_id 的旧版本")
    message: str


class ChatRequest(BaseModel):
    """问答请求体（SEC-6：长度与历史上限）"""
    query: str = Field(..., min_length=1, max_length=settings.security.MAX_QUERY_CHARS, description="用户提问")
    session_id: Optional[str] = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}",
                                      max_length=64, description="会话ID")
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="多轮对话历史")
    max_retries: int = Field(default=2, ge=0, le=5, description="自省最大改写重试次数")

    @field_validator("chat_history")
    @classmethod
    def _limit_history(cls, v):
        """只保留 user/assistant 消息；超出条数或总字数时丢弃最早的消息"""
        if not v:
            return []
        cleaned = [
            {"role": m.get("role"), "content": str(m.get("content", ""))}
            for m in v if m.get("role") in ("user", "assistant")
        ][-settings.security.MAX_HISTORY_MESSAGES:]
        total, kept = 0, []
        for m in reversed(cleaned):
            total += len(m["content"])
            if total > settings.security.MAX_HISTORY_CHARS:
                break
            kept.append(m)
        return list(reversed(kept))


class EvaluationRunRequest(BaseModel):
    mode: str = Field(default="retrieval", pattern="^(retrieval|full)$",
                      description="retrieval：只评检索（快）；full：再逐题真跑问答并由判官打分（慢）")
    include_llm_configs: bool = Field(default=False, description="检索评测是否包含多查询扩展 / HyDE（需调用对话模型）")
    limit: Optional[int] = Field(default=None, ge=1, le=200, description="端到端评测只跑前 N 题")
    report_name: Optional[str] = Field(default=None, description="报告文件名（不含扩展名也可），如 my_report.json")


# =============================================================================
# 接口 1：文档管理
# =============================================================================

@router.post("/documents/upload", response_model=DocumentUploadResponse, summary="上传文档并建立混合索引")
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="待入库的文档 (.txt, .md, .json, .csv, .pdf)"),
    doc_id: Optional[str] = Form(None, description="文档唯一标识（缺省自动生成；已存在则替换旧版本）"),
    background: bool = Query(False, description="true：校验通过后立即返回任务 ID（202），在后台入库（CONC-2）"),
):
    _check_rate(upload_limiter, request)
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未检测到有效的文件名")

    max_bytes = int(settings.security.MAX_UPLOAD_MB * 1024 * 1024)
    content = await file.read(max_bytes + 1)  # 最多读上限+1 字节，超出即拒绝，不把超大文件整个读进内存
    if len(content) > max_bytes:
        raise HTTPException(status_code=413,
                            detail=f"文件超过大小上限 {settings.security.MAX_UPLOAD_MB} MB")

    if background:
        try:
            display_name, _, checked_id = knowledge_base.validate_upload(content, file.filename, doc_id or None)
        except SecurityError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        async def job(progress):
            return await knowledge_base.add_document(content, file.filename, checked_id, progress=progress)

        task_id = ingest_tasks.submit(display_name, job)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"task_id": task_id, "doc_id": checked_id, "filename": display_name,
                                     "message": "已开始后台入库，可通过 /documents/tasks/{task_id} 查询进度"})

    try:
        rec = await knowledge_base.add_document(content, file.filename, doc_id or None)
    except DuplicateContentError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"相同内容的文档已存在（doc_id={e.existing_doc_id}）")
    except (SecurityError, DocumentParseError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"[API] 文档索引构建失败: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="索引构建失败，请查看服务端日志")

    return DocumentUploadResponse(
        doc_id=rec["doc_id"],
        filename=rec["doc_name"],
        parent_chunks=rec["parent_chunks"],
        child_chunks=rec["child_chunks"],
        replaced=rec["replaced"],
        message="已替换旧版本并重建索引。" if rec["replaced"] else "文档已建立父子分块与 BM25 + 向量双路索引。",
    )


@router.get("/documents/tasks/{task_id}", summary="查询后台入库任务的进度")
async def get_ingest_task(task_id: str):
    info = ingest_tasks.get(task_id)
    if info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在或已过期")
    return info


@router.get("/documents", summary="列出已入库的文档")
async def list_documents():
    return {"documents": knowledge_base.list_documents()}


@router.delete("/documents/{doc_id}", summary="删除文档（同时清理 BM25、向量库、母块与原文件）")
async def delete_document(doc_id: str, request: Request):
    _require_admin(request)
    try:
        result = await knowledge_base.delete_document(doc_id)
    except SecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DocumentNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"文档不存在: {doc_id}")
    return {"status": "deleted", **result}


# =============================================================================
# 接口 2：同步问答
# =============================================================================

@router.post("/chat/query", response_model=ChatQueryResponse, summary="同步自省问答")
async def chat_query(req: ChatRequest, request: Request):
    _check_rate(chat_limiter, request)
    start_time = time.time()
    try:
        final_state = await asyncio.wait_for(
            run_self_rag(
                query=req.query,
                session_id=req.session_id,
                chat_history=req.chat_history,
                max_retries=req.max_retries,
            ),
            timeout=settings.security.REQUEST_TOTAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail=f"问答超过 {settings.security.REQUEST_TOTAL_TIMEOUT:.0f} 秒未完成")
    except Exception as e:
        logger.error(f"[API] 问答执行失败: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Agent 推理异常，请查看服务端日志")

    return ChatQueryResponse(
        session_id=req.session_id,
        query=req.query,
        answer=final_state.get("generated_answer", "未能生成有效回答"),
        citations=final_state.get("citations", []),
        latency_ms=round((time.time() - start_time) * 1000, 2),
        steps_timeline=final_state.get("steps_timeline", []),
        quality=summarize_quality(final_state),
    )


# =============================================================================
# 接口 3：SSE 流式问答
# =============================================================================

def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", summary="SSE 流式问答与节点事件推送")
async def chat_stream(req: ChatRequest, request: Request):
    """
    事件类型：
    - step：某个节点完成（时间线）
    - token：回答的增量片段（真流式，生成节点边收边推；attempt 表示第几版回答）
    - regenerate：上一版回答未通过事实核查，开始生成修正版
    - audit：事实核查结论（pass / fail / uncertain / skipped / unverified）
    - citations：引用卡片；done：结束，附带质量摘要 quality；error：出错
    另外每隔 SSE_HEARTBEAT_SECONDS 秒发送一行 SSE 注释作为心跳，防止代理或浏览器因长时间无数据而断开。
    """
    _check_rate(chat_limiter, request)

    async def sse_event_generator():
        start_time = time.time()
        deadline = start_time + settings.security.REQUEST_TOTAL_TIMEOUT
        initial_state = create_initial_state(
            query=req.query,
            session_id=req.session_id,
            chat_history=req.chat_history,
            max_retries=req.max_retries,
        )
        queue: asyncio.Queue = asyncio.Queue()
        _END = object()

        async def produce():
            try:
                async for node_event in self_rag_app.astream(initial_state, config=GRAPH_CONFIG):
                    await queue.put({"type": "__node__", "data": node_event})
            except Exception as err:  # 交给消费者统一转成 error 事件
                await queue.put(err)
            finally:
                await queue.put(_END)

        # 节点内部（生成、审计）通过 event_sink 把 token 等事件推到同一个队列；
        # create_task 会复制当前上下文，所以必须先 set 再创建任务（AGENT-8）
        sink_token = event_sink.set(queue)
        producer = asyncio.create_task(produce())
        event_sink.reset(sink_token)

        final_state: Dict[str, Any] = dict(initial_state)
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=min(SSE_HEARTBEAT_SECONDS, remaining))
                except asyncio.TimeoutError:
                    if time.time() >= deadline:
                        raise
                    yield ": keep-alive\n\n"
                    continue
                if item is _END:
                    break
                if isinstance(item, Exception):
                    raise item
                if item.get("type") != "__node__":
                    yield _sse(item)          # token / regenerate / audit 直接转发
                    continue
                for node_name, node_output in item["data"].items():
                    node_output = node_output or {}
                    for k, v in node_output.items():   # 合并出终态（时间线是累加通道）
                        final_state[k] = final_state.get(k, []) + v if k == "steps_timeline" else v
                    if node_output.get("steps_timeline"):
                        last_step = node_output["steps_timeline"][-1]
                        yield _sse({
                            "type": "step", "node": node_name,
                            "title": last_step.get("title", node_name),
                            "detail": last_step.get("detail", ""),
                            "timestamp": time.time(),
                        })

            citations = [c.model_dump() for c in final_state.get("citations") or []]
            if citations:
                yield _sse({"type": "citations", "data": citations})
            yield _sse({"type": "done", "session_id": req.session_id,
                        "latency_ms": round((time.time() - start_time) * 1000, 2),
                        "quality": summarize_quality(final_state)})

        except asyncio.TimeoutError:
            yield _sse({"type": "error",
                        "message": f"问答超过 {settings.security.REQUEST_TOTAL_TIMEOUT:.0f} 秒未完成"})
        except Exception as err:
            logger.error(f"[API SSE] 流式问答异常: {err}", exc_info=True)
            yield _sse({"type": "error", "message": "Agent 推理异常，请查看服务端日志"})
        finally:
            if not producer.done():
                producer.cancel()  # 客户端断开或超时时，停止后台仍在运行的图

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# 接口 4：评测
# =============================================================================

@router.post("/evaluation/run", summary="运行真实评测（检索指标 / 端到端）")
async def run_evaluation(req: EvaluationRunRequest, request: Request):
    """
    EVAL-1：原来这里评测的是写死在代码里的 2 条"回答 + 上下文"，完全不经过本系统，
    同样的输入两次跑出 91.9% 和 99.4%。现在改为用 eval_sets/kb_eval_v1.json 的题目真实运行检索
    （以及可选的完整问答），报告同时保存到 data/eval/。
    完整的消融实验与多次判官请用命令行：python -m experiments.rag_eval --mode full
    """
    _require_admin(request)  # 评测会写文件并产生模型调用费用
    from app.eval.pipeline import run_evaluation as run_pipeline, save_report
    try:
        name = req.report_name
        if name:
            validate_report_name(name if name.endswith(".json") else f"{name}.json")
    except SecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        report = await run_pipeline(mode=req.mode, include_llm_configs=req.include_llm_configs, limit=req.limit)
        stem = (name[:-5] if name and name.endswith(".json") else name) or f"rag_eval_{req.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        save_report(report, settings.storage.EVAL_DATASET_DIR, stem)
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"[API] 评测任务运行失败: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="评测执行失败，请查看服务端日志")


@router.get("/evaluation/latest", summary="读取最近一次评测报告（页面打开时展示）")
async def latest_evaluation():
    eval_dir = settings.storage.EVAL_DATASET_DIR
    files = sorted(eval_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True) if eval_dir.exists() else []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "retrieval" in data:  # 只认新版评测报告，旧版手写样本的报告忽略
            return {"report": data, "file": f.name}
    return {"report": None}


# =============================================================================
# 接口 5：状态
# =============================================================================

@router.get("/status", summary="知识库与检索引擎状态")
async def get_rag_status():
    kb = knowledge_base.stats()
    eval_dir = settings.storage.EVAL_DATASET_DIR
    return {
        "status": "online",
        "documents_indexed": kb["documents"],
        "parent_documents_indexed": kb["parent_chunks"],
        "child_chunks_indexed": kb["child_chunks"],
        "raw_uploaded_files": kb["documents"],
        "eval_reports_archived": len(list(eval_dir.glob("*.json"))) if eval_dir.exists() else 0,
        "vector_db_type": settings.storage.VECTOR_DB_TYPE,
        "embedding_signature": kb["embedding_signature"],
        "active_models": {
            "chat_model": settings.llm.DEFAULT_MODEL,
            "embedding_model": settings.embedding.MODEL_NAME,
            "reranker_model": settings.reranker.MODEL_NAME,
        },
        "hyperparams": {
            "parent_chunk_size": settings.rag.PARENT_CHUNK_SIZE,
            "child_chunk_size": settings.rag.CHILD_CHUNK_SIZE,
            "dense_top_k": settings.rag.DENSE_TOP_K,
            "sparse_top_k": settings.rag.SPARSE_TOP_K,
            "rrf_k": settings.rag.RRF_K,
            "final_top_k": settings.rag.FINAL_TOP_K,
        },
    }


# =============================================================================
# 接口 6：模型配置与连通性测试
# =============================================================================

def _mask_api_key(key: str) -> str:
    """脱敏展示 API 密钥"""
    if not key or "your_api_key" in key:
        return ""
    if len(key) <= 8:
        return "******"
    return f"{key[:3]}****{key[-4:]}"


def _key_source(own_key: str, inherited: bool) -> str:
    if _mask_api_key(own_key):
        return "own"
    return "inherited" if inherited else "none"


class ModelSettingsPayload(BaseModel):
    llm_api_key: Optional[str] = Field(default=None, max_length=256)
    llm_base_url: Optional[str] = Field(default=None, max_length=512)
    llm_default_model: Optional[str] = Field(default=None, max_length=128)
    llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    embedding_api_key: Optional[str] = Field(default=None, max_length=256)
    embedding_base_url: Optional[str] = Field(default=None, max_length=512)
    embedding_model_name: Optional[str] = Field(default=None, max_length=128)
    reranker_api_key: Optional[str] = Field(default=None, max_length=256)
    reranker_base_url: Optional[str] = Field(default=None, max_length=512)
    reranker_model_name: Optional[str] = Field(default=None, max_length=128)


@router.get("/settings", summary="获取模型配置（密钥脱敏）")
async def get_settings():
    return {
        "llm": {
            "api_key_configured": llm_provider.is_llm_configured,
            "api_key_masked": _mask_api_key(settings.llm.API_KEY),
            "base_url": settings.llm.BASE_URL,
            "model": settings.llm.DEFAULT_MODEL,
            "temperature": settings.llm.TEMPERATURE,
        },
        "embedding": {
            "api_key_configured": llm_provider.is_embed_configured,
            "api_key_masked": _mask_api_key(settings.embedding.API_KEY),
            # own：使用自己的 Key；inherited：与对话模型同一服务商，复用其 Key；none：未配置，使用离线模式
            "api_key_source": _key_source(settings.embedding.API_KEY, llm_provider.embed_key_inherited),
            "base_url": settings.embedding.BASE_URL,
            "model": settings.embedding.MODEL_NAME,
            "dimension": settings.embedding.DIMENSION,
        },
        "reranker": {
            "api_key_configured": llm_provider.is_rerank_configured,
            "api_key_masked": _mask_api_key(settings.reranker.API_KEY),
            "api_key_source": _key_source(settings.reranker.API_KEY, llm_provider.rerank_key_inherited),
            "base_url": settings.reranker.BASE_URL,
            "model": settings.reranker.MODEL_NAME,
        },
        "admin_token_required": bool(settings.security.ADMIN_TOKEN),
    }


@router.post("/settings", summary="更新模型配置并持久化到 .env（需要管理权限）")
async def update_settings(payload: ModelSettingsPayload, request: Request):
    _require_admin(request)

    updates: Dict[str, str] = {}
    try:
        for field_name, env_key in (
            ("llm_base_url", "LLM_BASE_URL"),
            ("embedding_base_url", "EMBEDDING_BASE_URL"),
            ("reranker_base_url", "RERANKER_BASE_URL"),
        ):
            value = getattr(payload, field_name)
            if value is not None and value.strip():
                # SEC-4：网关地址必须是 https（本机地址除外），防止把密钥发到明文或任意地址
                updates[env_key] = validate_base_url(value)
    except SecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    for field_name, env_key in (
        ("llm_api_key", "LLM_API_KEY"), ("embedding_api_key", "EMBEDDING_API_KEY"),
        ("reranker_api_key", "RERANKER_API_KEY"), ("llm_default_model", "LLM_DEFAULT_MODEL"),
        ("embedding_model_name", "EMBEDDING_MODEL_NAME"), ("reranker_model_name", "RERANKER_MODEL_NAME"),
    ):
        value = getattr(payload, field_name)
        if value is not None and value.strip():
            updates[env_key] = value.strip()
    if payload.llm_temperature is not None:
        updates["LLM_TEMPERATURE"] = str(payload.llm_temperature)

    reindexed = False
    if updates:
        try:
            save_env_settings(updates)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        llm_provider.reload_config()
        # API-2：向量模型变化后，旧向量与新查询不在同一空间，必须重建索引
        reindexed = await knowledge_base.ensure_embedding_consistency()

    return {
        "status": "success",
        "message": f"已更新 {len(updates)} 项配置" + ("，向量模型已变化，索引已全部重建" if reindexed else ""),
        "reindexed": reindexed,
        "settings": await get_settings(),
    }


@router.post("/settings/test", summary="测试模型网关连通性（需要管理权限）")
async def test_connectivity(request: Request):
    _require_admin(request)
    results = await llm_provider.test_connectivity()
    return {"status": "completed", "tested_at": time.time(), "components": results}
