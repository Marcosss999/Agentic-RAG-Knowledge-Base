"""
FastAPI 企业级知识库与 Agentic RAG 业务路由层 (API Router)
职责：
1. 暴露标准文档上传与异步索引接口 (/documents/upload)：
   - 支持多格式文本/文档上传，驱动 Parent-Child 父子多尺度切分与 BM25 + Qdrant 双路索引持久化。
2. 暴露标准同步问答接口 (/chat/query)：
   - 调度 LangGraph 自省状态机，返回严谨答案、证据溯源卡片 (Citations)、端到端耗时与推理时间线。
3. 暴露打字机流式 SSE 问答接口 (/chat/stream)：
   - 基于 Server-Sent Events (SSE) 协议，实时流式推送 Agent 节点流转事件、状态审计判定与最终 Token 逐字流。
4. 暴露自动化 Ragas 基准评测接口 (/evaluation/run)：
   - 触发 LLM-as-a-Judge 四维黄金指标（忠实度、相关度、精准度、召回率）批量量化评测，归档 JSON 报告。
5. 暴露知识库状态监控探针 (/status)：
   - 实时感知已索引母块实体、存储目录状态与底层检索健康度。
"""

import os
import time
import json
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.schemas import ChatQueryResponse, CitationReference
from app.rag.hybrid_retriever import hybrid_retriever
from app.agent.state import create_initial_state
from app.agent.self_rag_graph import run_self_rag, self_rag_app
from app.eval.ragas_evaluator import ragas_evaluator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["Enterprise Agentic RAG Platform"])


# =============================================================================
# API 请求与响应契约模型
# =============================================================================

class DocumentUploadResponse(BaseModel):
    """文档上传解析响应结构"""
    doc_id: str
    filename: str
    parent_chunks: int = Field(..., description="生成的全局背景母块数量")
    child_chunks: int = Field(..., description="存入向量库与倒排索引的微粒度子块数量")
    message: str


class ChatRequest(BaseModel):
    """问答交互请求体"""
    query: str = Field(..., min_length=1, description="用户提问内容")
    session_id: Optional[str] = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}", description="会话ID")
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="多轮对话历史列表")
    max_retries: int = Field(default=2, ge=0, le=5, description="自省最大改写重试次数")


class EvaluationRunRequest(BaseModel):
    """评测触发请求体"""
    dataset: Optional[List[Dict[str, Any]]] = Field(default=None, description="自定义待评测问答集合")
    report_name: Optional[str] = Field(default=None, description="指定输出的报告文件名")


# =============================================================================
# 接口 1：文档上传与父子双路索引构建 (Upload & Index Pipeline)
# =============================================================================

@router.post("/documents/upload", response_model=DocumentUploadResponse, summary="上传文档并建立多级混合索引")
async def upload_document(
    file: UploadFile = File(..., description="待解析入库的文档 (.txt, .md, .json, .csv)"),
    doc_id: Optional[str] = Form(None, description="文档唯一标识 (缺省时自动生成)")
):
    """
    接收私有知识文档，异步执行：
    1. 保存原始文件到本地数据存储目录；
    2. 执行 ParentChildSplitter 多尺度分块；
    3. 写入 Okapi BM25 稀疏倒排索引与 Qdrant 密集向量存储。
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未检测到有效的文件名")

    target_doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"
    upload_path = settings.storage.UPLOAD_DIR / f"{target_doc_id}_{file.filename}"

    try:
        content_bytes = await file.read()
        # 尝试常用字符集解码
        try:
            raw_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = content_bytes.decode("gbk", errors="ignore")

        if not raw_text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="上传文件内容为空")

        # 写入物理磁盘归档
        with open(upload_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # 驱动 HybridRetriever 构建母块与子块双路索引
        parent_cnt, child_cnt = await hybrid_retriever.index_document(
            raw_text=raw_text,
            doc_id=target_doc_id,
            doc_name=file.filename,
            metadata={"source_path": str(upload_path), "file_size": len(content_bytes)}
        )

        return DocumentUploadResponse(
            doc_id=target_doc_id,
            filename=file.filename,
            parent_chunks=parent_cnt,
            child_chunks=child_cnt,
            message="文档已成功建立父子双层索引及 BM25 + Qdrant 混合召回通道。"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] 文档索引构建失败: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"索引构建失败: {str(e)}")


# =============================================================================
# 接口 2：同步自省问答 (Non-Streaming Agentic Q&A)
# =============================================================================

@router.post("/chat/query", response_model=ChatQueryResponse, summary="标准同步自省问答交互")
async def chat_query(req: ChatRequest):
    """
    端到端触发 Self-RAG 自省状态机：
    1. 查询意图分析与指代消解 (Query Transformation)；
    2. BM25 + 向量并发召回与 RRF/Cross-Encoder 重排；
    3. 闸门一：检索材料相关度自检与自适应改写；
    4. 严格事实合成回答；
    5. 闸门二：回答忠实度 (Faithfulness) 幻觉审计；
    6. 返回标准答案、高亮证据引用卡片与全链路执行时间线。
    """
    start_time = time.time()
    try:
        final_state = await run_self_rag(
            query=req.query,
            session_id=req.session_id,
            chat_history=req.chat_history,
            max_retries=req.max_retries
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return ChatQueryResponse(
            session_id=req.session_id,
            query=req.query,
            answer=final_state.get("generated_answer", "未能生成有效回答"),
            citations=final_state.get("citations", []),
            latency_ms=latency_ms,
            steps_timeline=final_state.get("steps_timeline", [])
        )

    except Exception as e:
        logger.error(f"[API] 问答执行失败: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent 推理异常: {str(e)}")


# =============================================================================
# 接口 3：打字机流式 SSE 问答 (Server-Sent Events Streaming Q&A)
# =============================================================================

@router.post("/chat/stream", summary="SSE 打字机流式问答与状态机可解释性事件推送")
async def chat_stream(req: ChatRequest):
    """
    采用 Server-Sent Events (SSE) 协议推流：
    - 阶段 1：实时推送状态机节点流转事件 (type: 'step')，如正在消解指代、正在召回、正在质检；
    - 阶段 2：逐字/逐词打字机吐出模型 Token (type: 'token')，秒级优化首字响应延迟 (TTFT)；
    - 阶段 3：下发证据引用卡片 (type: 'citations') 与执行完毕包 (type: 'done')。
    """
    async def sse_event_generator():
        start_time = time.time()
        initial_state = create_initial_state(
            query=req.query,
            session_id=req.session_id,
            chat_history=req.chat_history,
            max_retries=req.max_retries
        )

        final_answer = ""
        citations = []

        try:
            # 1. 订阅 LangGraph 自省状态机异步节点流
            async for node_event in self_rag_app.astream(initial_state):
                for node_name, node_output in node_event.items():
                    # 捕获并向前端推送当前节点执行情况
                    if "steps_timeline" in node_output and node_output["steps_timeline"]:
                        last_step = node_output["steps_timeline"][-1]
                        step_payload = {
                            "type": "step",
                            "node": node_name,
                            "title": last_step.get("title", node_name),
                            "detail": last_step.get("detail", ""),
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(step_payload, ensure_ascii=False)}\n\n"

                    # 捕获证据引用
                    if "citations" in node_output and node_output["citations"]:
                        citations = [c.model_dump() for c in node_output["citations"]]

                    # 捕获生成结果
                    if "generated_answer" in node_output:
                        final_answer = node_output["generated_answer"]

            # 2. 模拟打字机平滑逐字吐出 Token (Token-level Streaming)
            # 在没有实时底层大模型 WebSocket 流时，提供优雅的平滑打字机体验
            if final_answer:
                chunk_size = 4
                for i in range(0, len(final_answer), chunk_size):
                    chunk = final_answer[i: i + chunk_size]
                    token_payload = {
                        "type": "token",
                        "content": chunk
                    }
                    yield f"data: {json.dumps(token_payload, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.015)

            # 3. 推送引用卡片列表
            if citations:
                citations_payload = {
                    "type": "citations",
                    "data": citations
                }
                yield f"data: {json.dumps(citations_payload, ensure_ascii=False)}\n\n"

            # 4. 传输完成结束信号
            latency_ms = round((time.time() - start_time) * 1000, 2)
            done_payload = {
                "type": "done",
                "session_id": req.session_id,
                "latency_ms": latency_ms
            }
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

        except Exception as err:
            logger.error(f"[API SSE] 流式事件推送异常: {err}", exc_info=True)
            error_payload = {"type": "error", "message": str(err)}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" # 兼容 Nginx 代理禁用响应缓冲
        }
    )


# =============================================================================
# 接口 4：Ragas 四维自动化基准评测运行 (Run Benchmark Evaluation)
# =============================================================================

@router.post("/evaluation/run", summary="在线触发 Ragas 四维黄金指标量化评测")
async def run_evaluation(req: EvaluationRunRequest):
    """
    触发 LLM-as-a-Judge 自动化基准评测：
    - 若传入自定义测试集，则针对传入集合打分；
    - 缺省时加载内置黄金问答对测试集，输出综合量化报表并自动落盘归档至 data/eval/ 目录。
    """
    default_dataset = [
        {
            "query": "系统的父子分块切分大小分别是多少？",
            "answer": "母块尺寸为1500字符保留完整全局上下文，子块尺寸为300字符用于向量高维检索。",
            "contexts": ["系统采用ParentChildSplitter，母块大小1500字，子块大小300字并建立Qdrant向量。"],
            "ground_truth": "母块1500字，子块300字。"
        },
        {
            "query": "检索融合阶段采用了什么算法抹平量纲差异？",
            "answer": "检索融合阶段采用了 Reciprocal Rank Fusion (RRF，平滑常数 k=60) 互逆排名融合算法。",
            "contexts": ["利用RRF互逆排名算法对BM25与向量名次进行无量纲合并，常数取60。"],
            "ground_truth": "RRF互逆排名融合算法(k=60)。"
        }
    ]

    target_dataset = req.dataset or default_dataset

    try:
        report = await ragas_evaluator.evaluate_batch(
            dataset=target_dataset,
            report_name=req.report_name
        )
        return {
            "status": "success",
            "report_summary": report
        }
    except Exception as e:
        logger.error(f"[API] 评测任务运行失败: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"评测执行失败: {str(e)}")


# =============================================================================
# 接口 5：知识库与检索状态监控 (Knowledge Base Health & Status)
# =============================================================================

@router.get("/status", summary="查询知识库与向量引擎健康状态")
async def get_rag_status():
    """返回当前已注册母块实体数量、存储目录路径及运行模式"""
    parent_count = len(hybrid_retriever.parent_docs_store)
    upload_files_count = len(list(settings.storage.UPLOAD_DIR.glob("*"))) if settings.storage.UPLOAD_DIR.exists() else 0
    eval_reports_count = len(list(settings.storage.EVAL_DATASET_DIR.glob("*.json"))) if settings.storage.EVAL_DATASET_DIR.exists() else 0

    return {
        "status": "online",
        "parent_documents_indexed": parent_count,
        "raw_uploaded_files": upload_files_count,
        "eval_reports_archived": eval_reports_count,
        "vector_db_type": settings.storage.VECTOR_DB_TYPE,
        "active_models": {
            "chat_model": settings.llm.DEFAULT_MODEL,
            "embedding_model": settings.embedding.MODEL_NAME,
            "reranker_model": settings.reranker.MODEL_NAME
        },
        "hyperparams": {
            "parent_chunk_size": settings.rag.PARENT_CHUNK_SIZE,
            "child_chunk_size": settings.rag.CHILD_CHUNK_SIZE,
            "dense_top_k": settings.rag.DENSE_TOP_K,
            "sparse_top_k": settings.rag.SPARSE_TOP_K,
            "rrf_k": settings.rag.RRF_K,
            "final_top_k": settings.rag.FINAL_TOP_K
        }
    }
