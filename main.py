"""
企业级 Agentic RAG 智能知识库平台应用主入口 (Application Entrypoint)
职责：
1. 装配 FastAPI 顶层应用实例与现代化 Lifespan 生命周期管理器 (启动预热、目录检测、优雅停机)。
2. 配置生产级 CORS 跨域资源共享中间件，无缝适配 Vue/React/Web 前端生态。
3. 挂载 API V1 核心业务路由 (/api/v1/rag)，集成文档索引、同步问答、SSE 打字机推流与 Ragas 自动化评测。
4. 暴露根路径导航与 /health 高可用健康检查探针，适配 Kubernetes/Docker 容器存活检测。
5. 集成 Uvicorn 异步 ASGI 服务器快速启动引导。
"""

import os
import time
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.rag_router import router as rag_router

# 配置全局日志输出格式
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 现代生命周期上下文管理器 (Lifespan Context Manager)
    负责服务启动时的依赖预热与服务关闭时的资源释放
    """
    # 1. 启动阶段 (Startup)
    logger.info("=" * 60)
    logger.info(f"正在启动 {settings.PROJECT_NAME} 服务...")
    logger.info("=" * 60)

    # 确保运行时目录完备
    settings.ensure_directories()
    logger.info(f"[Startup] 上传目录已就绪: {settings.storage.UPLOAD_DIR}")
    logger.info(f"[Startup] 评测归档目录已就绪: {settings.storage.EVAL_DATASET_DIR}")
    logger.info(f"[Startup] 向量库配置模式: {settings.storage.VECTOR_DB_TYPE}")
    logger.info(f"[Startup] 基础对话模型: {settings.llm.DEFAULT_MODEL}")
    logger.info(f"[Startup] 密集向量模型: {settings.embedding.MODEL_NAME}")
    logger.info(f"[Startup] 交叉重排模型: {settings.reranker.MODEL_NAME}")

    start_time = time.time()
    yield  # 服务运行中，处理外部请求

    # 2. 关闭阶段 (Shutdown)
    uptime = round(time.time() - start_time, 2)
    logger.info("=" * 60)
    logger.info(f"{settings.PROJECT_NAME} 服务正在优雅停机 (运行总时长: {uptime}s)...")
    logger.info("=" * 60)


# 初始化 FastAPI 应用实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description=(
        "基于 LangGraph 自省状态机与混合检索（BM25 + Qdrant + RRF + Cross-Encoder + Self-RAG）"
        "的企业级 Agentic RAG 智能知识库平台。"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 配置生产级跨域访问控制 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许跨域源，生产环境可收敛至具体前端域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有标头
)

# 挂载业务路由
app.include_router(rag_router, prefix=settings.API_V1_STR)


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =============================================================================
# 全局探针与基础导航端点
# =============================================================================

@app.get("/", summary="平台现代化 Web 控制台与导航")
async def root():
    """欢迎页面与现代化 Web 前端入口 (优先返回 index.html)"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return {
        "project": settings.PROJECT_NAME,
        "version": "1.0.0",
        "description": "Enterprise Agentic RAG Platform with Self-Reflection & Hybrid Search",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_v1_prefix": settings.API_V1_STR,
        "endpoints": {
            "upload_document": f"{settings.API_V1_STR}/rag/documents/upload",
            "chat_query": f"{settings.API_V1_STR}/rag/chat/query",
            "chat_stream_sse": f"{settings.API_V1_STR}/rag/chat/stream",
            "run_evaluation": f"{settings.API_V1_STR}/rag/evaluation/run",
            "system_status": f"{settings.API_V1_STR}/rag/status"
        }
    }


@app.get("/health", summary="服务存活与健康检查探针")
async def health_check():
    """Kubernetes / 容器健康存活性探针 (Liveness & Readiness Probe)"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "timestamp": time.time(),
            "service": settings.PROJECT_NAME
        }
    )


# =============================================================================
# 独立脚本启动入口
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1
    )
