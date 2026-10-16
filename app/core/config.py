"""
全局配置管理模块 (Core Configuration)
职责：
1. 统一管理大模型、向量模型、重排模型、数据库及 RAG 核心超参数。
2. 支持优先读取系统环境变量或 .env 配置文件，支持默认生产级超参数回退。
3. 提供强类型校验，避免运行时因缺参导致静默失败。
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# 定位项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"

def _load_env_file(filepath: Path):
    """简易轻量 .env 解析器，避免强依赖额外包"""
    if not filepath.exists():
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            # 系统环境变量优先级高于 .env 文件
            if key not in os.environ:
                os.environ[key] = val

_load_env_file(ENV_FILE)


class LLMSettings(BaseModel):
    """对话大模型配置"""
    API_KEY: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    BASE_URL: str = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    DEFAULT_MODEL: str = Field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "qwen-plus"))
    TEMPERATURE: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    REQUEST_TIMEOUT: int = Field(default=60, description="请求超时时间(秒)")


class EmbeddingSettings(BaseModel):
    """文本向量化模型配置"""
    API_KEY: str = Field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "")))
    BASE_URL: str = Field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")))
    MODEL_NAME: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4"))
    DIMENSION: int = Field(default=1024, description="向量维度 (text-embedding-v4 为 1024 维)")


class RerankerSettings(BaseModel):
    """交叉重排模型配置 (Cross-Encoder)"""
    API_KEY: str = Field(default_factory=lambda: os.getenv("RERANKER_API_KEY", os.getenv("LLM_API_KEY", "")))
    BASE_URL: str = Field(default_factory=lambda: os.getenv("RERANKER_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"))
    MODEL_NAME: str = Field(default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", "gte-rerank-v2"))
    TOP_N: int = Field(default=5, description="重排后保留的最相关片段数量")
    MIN_SCORE_THRESHOLD: float = Field(default=0.15, description="重排相关度过滤硬阈值")


class RAGHyperParams(BaseModel):
    """RAG 检索与切分超参数"""
    # 父子分块大小与重叠
    PARENT_CHUNK_SIZE: int = Field(default=1500, description="父文档块大小(字)")
    PARENT_CHUNK_OVERLAP: int = Field(default=150, description="父文档块重叠(字)")
    CHILD_CHUNK_SIZE: int = Field(default=300, description="子文档块大小(字)")
    CHILD_CHUNK_OVERLAP: int = Field(default=50, description="子文档块重叠(字)")
    
    # 多路召回候选规模
    DENSE_TOP_K: int = Field(default=20, description="向量检索初筛数量")
    SPARSE_TOP_K: int = Field(default=20, description="BM25关键词检索初筛数量")
    RRF_K: int = Field(default=60, description="RRF名次平滑常数")
    FINAL_TOP_K: int = Field(default=5, description="最终送入LLM上下文的片段上限")


class StorageSettings(BaseModel):
    """存储与向量库配置"""
    UPLOAD_DIR: Path = ROOT_DIR / "data" / "uploads"
    EVAL_DATASET_DIR: Path = ROOT_DIR / "data" / "eval"
    
    # 向量库选型: qdrant / pgvector / memory
    VECTOR_DB_TYPE: str = Field(default_factory=lambda: os.getenv("VECTOR_DB_TYPE", "qdrant"))
    QDRANT_HOST: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    QDRANT_PORT: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    COLLECTION_NAME: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "enterprise_kb"))


class AppSettings(BaseModel):
    """应用总体配置入口"""
    PROJECT_NAME: str = "Enterprise-Agentic-RAG"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    
    # 子模块配置组合
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    rag: RAGHyperParams = Field(default_factory=RAGHyperParams)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    def ensure_directories(self):
        """确保运行时必要目录已创建"""
        self.storage.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.storage.EVAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)


# 全局单例
settings = AppSettings()
settings.ensure_directories()
