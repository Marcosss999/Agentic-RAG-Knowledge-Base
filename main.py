"""
Agentic RAG 知识库平台应用入口 (Application Entrypoint)
职责：
1. 装配 FastAPI 顶层应用实例与现代化 Lifespan 生命周期管理器 (启动预热、目录检测、优雅停机)。
2. 配置生产级 CORS 跨域资源共享中间件，无缝适配 Vue/React/Web 前端生态。
3. 挂载 API V1 核心业务路由 (/api/v1/rag)，集成文档索引、同步问答、SSE 打字机推流与 Ragas 自动化评测。
4. 暴露根路径导航与 /health 高可用健康检查探针，适配 Kubernetes/Docker 容器存活检测。
5. 集成 Uvicorn 异步 ASGI 服务器快速启动引导。
"""

import asyncio
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

    # 预热分词器（jieba 首次加载约 0.5~1 秒，放到线程池，避免阻塞事件循环）
    try:
        import jieba  # noqa: F401
        await asyncio.to_thread(jieba.initialize)
    except Exception:
        pass

    # 按文档清单重建全部索引（DATA-2 / DATA-6：上传与重启共用同一抽取函数，doc_id 保持不变）
    try:
        from app.rag.knowledge_base import knowledge_base
        result = await knowledge_base.rebuild_from_disk()
        logger.info(f"[Startup] 知识库重建完成: {result}")
    except Exception as e:
        logger.error(f"[Startup] 知识库重建失败: {e}", exc_info=True)

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
        "的 Agentic RAG 知识库平台。"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 跨域访问控制 (SEC-4)
# 原配置是 allow_origins=["*"] + allow_credentials=True：任何网页都能从你的浏览器调用本机接口。
# 现在只允许白名单来源（默认本机前端），不携带凭据；管理接口另有令牌/本机校验。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Token"],
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
    # 默认只监听本机 (127.0.0.1)；需要局域网访问时设置环境变量 HOST=0.0.0.0，并务必配置 ADMIN_TOKEN
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=settings.DEBUG,
        workers=1
    )
