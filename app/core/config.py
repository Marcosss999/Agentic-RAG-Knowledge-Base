"""
全局配置管理模块 (Core Configuration)
职责：
1. 统一管理大模型、向量模型、重排模型、数据库、安全限额及 RAG 核心超参数。
2. 支持优先读取系统环境变量或 .env 配置文件，未配置时回退到默认值。
3. 提供强类型校验，避免运行时因缺参导致静默失败。
"""

import os
from pathlib import Path
from typing import List
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


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: str) -> List[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


# 允许通过 /settings 接口写入 .env 的键（白名单），防止借接口写入任意配置
WRITABLE_ENV_KEYS = {
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_DEFAULT_MODEL", "LLM_TEMPERATURE",
    "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL_NAME",
    "RERANKER_API_KEY", "RERANKER_BASE_URL", "RERANKER_MODEL_NAME",
}


def save_env_settings(updates: dict) -> None:
    """持久化保存大模型与检索配置到 .env 文件，并同步更新当前运行时环境变量与配置对象。

    安全约束 (SEC-4 / ENG-6)：
    - 只允许写入 WRITABLE_ENV_KEYS 中的键；
    - 值中不允许出现换行符等控制字符，否则可借此向 .env 注入额外的配置行。
    """
    target_keys = {}
    for k, v in updates.items():
        if v is None:
            continue
        if k not in WRITABLE_ENV_KEYS:
            raise ValueError(f"不允许写入的配置项: {k}")
        val_str = str(v).strip()
        if any(ch in val_str for ch in ("\n", "\r", "\x00")):
            raise ValueError(f"配置项 {k} 的值包含非法控制字符")
        target_keys[k] = val_str

    for k, v in target_keys.items():
        os.environ[k] = v

    # 同步更新当前 settings 实例属性
    if "LLM_API_KEY" in target_keys:
        settings.llm.API_KEY = target_keys["LLM_API_KEY"]
    if "LLM_BASE_URL" in target_keys:
        settings.llm.BASE_URL = target_keys["LLM_BASE_URL"]
    if "LLM_DEFAULT_MODEL" in target_keys:
        settings.llm.DEFAULT_MODEL = target_keys["LLM_DEFAULT_MODEL"]
    if "LLM_TEMPERATURE" in target_keys:
        settings.llm.TEMPERATURE = float(target_keys["LLM_TEMPERATURE"])

    if "EMBEDDING_API_KEY" in target_keys:
        settings.embedding.API_KEY = target_keys["EMBEDDING_API_KEY"]
    if "EMBEDDING_BASE_URL" in target_keys:
        settings.embedding.BASE_URL = target_keys["EMBEDDING_BASE_URL"]
    if "EMBEDDING_MODEL_NAME" in target_keys:
        settings.embedding.MODEL_NAME = target_keys["EMBEDDING_MODEL_NAME"]

    if "RERANKER_API_KEY" in target_keys:
        settings.reranker.API_KEY = target_keys["RERANKER_API_KEY"]
    if "RERANKER_BASE_URL" in target_keys:
        settings.reranker.BASE_URL = target_keys["RERANKER_BASE_URL"]
    if "RERANKER_MODEL_NAME" in target_keys:
        settings.reranker.MODEL_NAME = target_keys["RERANKER_MODEL_NAME"]

    # 重写写入 .env 文件
    env_lines = []
    keys_written = set()

    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k, _ = stripped.split("=", 1)
                    k = k.strip()
                    if k in target_keys:
                        env_lines.append(f"{k}={target_keys[k]}\n")
                        keys_written.add(k)
                        continue
                env_lines.append(line if line.endswith("\n") else line + "\n")

    for k, v in target_keys.items():
        if k not in keys_written:
            env_lines.append(f"{k}={v}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(env_lines)


class LLMSettings(BaseModel):
    """对话大模型配置"""
    API_KEY: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    BASE_URL: str = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    DEFAULT_MODEL: str = Field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "qwen-plus"))
    TEMPERATURE: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    REQUEST_TIMEOUT: int = Field(default=60, description="单次请求超时时间(秒)")
    MAX_RETRIES: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")), description="SDK 内置的失败重试次数(指数退避)")


class EmbeddingSettings(BaseModel):
    """文本向量化模型配置"""
    # 只读取本项自己的 Key；留空时是否复用对话模型的 Key 由 llm_provider 按"是否同一服务商"判断 (API-6)
    API_KEY: str = Field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    # 不再默认沿用 LLM_BASE_URL：对话模型常用 DeepSeek 等没有向量接口的服务商 (API-6)
    BASE_URL: str = Field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    MODEL_NAME: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4"))
    DIMENSION: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1024")),
                           description="离线哈希向量的维度；在线模式以模型实际返回的维度为准")
    # API-1：百炼 text-embedding-v4 每次请求最多 10 行，超过会报错
    BATCH_SIZE: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "10")), description="单次 Embedding 请求的最大条数")
    MAX_CONCURRENCY: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "4")), description="同时在途的 Embedding 请求数上限")
    # DATA-7：在线向量写入本地缓存，重启重建索引时不再重复调用接口
    CACHE_ENABLED: bool = Field(default_factory=lambda: _env_bool("EMBEDDING_CACHE_ENABLED", "true"))


class RerankerSettings(BaseModel):
    """交叉重排模型配置 (Cross-Encoder)"""
    API_KEY: str = Field(default_factory=lambda: os.getenv("RERANKER_API_KEY", ""))
    BASE_URL: str = Field(default_factory=lambda: os.getenv("RERANKER_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"))
    MODEL_NAME: str = Field(default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", "gte-rerank-v2"))
    TOP_N: int = Field(default=5, description="重排后保留的最相关片段数量")
    # API-4：该阈值尚未标定（离线打分器与在线模型的分数尺度不同），先做成可配置项，标定见缺陷追踪 API-4 / EVAL-2
    MIN_SCORE_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("RERANKER_MIN_SCORE", "0.15")), description="重排相关度过滤阈值(未标定)")
    REQUEST_TIMEOUT: float = Field(default=15.0, description="重排接口超时(秒)")


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
    # RET-3：RRF 中向量路与 BM25 路的权重。默认值沿用原实现，是否调整以评测结果为准（见 eval_sets/、experiments/rag_eval.py）
    DENSE_WEIGHT: float = Field(default_factory=lambda: float(os.getenv("RAG_DENSE_WEIGHT", "1.0")))
    SPARSE_WEIGHT: float = Field(default_factory=lambda: float(os.getenv("RAG_SPARSE_WEIGHT", "0.8")))
    FINAL_TOP_K: int = Field(default=5, description="最终送入LLM上下文的片段上限")
    # DATA-4：上下文预算，母块去重后按此上限截断，避免同一母块重复占满窗口
    MAX_CONTEXT_CHARS: int = Field(default_factory=lambda: int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000")), description="送入 LLM 的上下文总字数上限")

    # Agent 流程开关（第 5 组）
    # AGENT-4：外部百科兜底默认关闭——私有知识库里的问题默认不应发往第三方
    WEB_FALLBACK_ENABLED: bool = Field(default_factory=lambda: _env_bool("WEB_FALLBACK_ENABLED", "false"))
    # AGENT-7：多查询扩展与 HyDE 接入但默认关闭，是否开启由第 6 组评测数据决定
    ENABLE_MULTI_QUERY: bool = Field(default_factory=lambda: _env_bool("ENABLE_MULTI_QUERY", "false"))
    ENABLE_HYDE: bool = Field(default_factory=lambda: _env_bool("ENABLE_HYDE", "false"))
    # AGENT-3：幻觉审计不通过时最多重新生成的次数（与检索改写次数分开计算）
    GENERATION_MAX_RETRIES: int = Field(default_factory=lambda: int(os.getenv("GENERATION_MAX_RETRIES", "1")))


class StorageSettings(BaseModel):
    """存储与向量库配置"""
    UPLOAD_DIR: Path = ROOT_DIR / "data" / "uploads"
    EVAL_DATASET_DIR: Path = ROOT_DIR / "data" / "eval"
    # DATA-6：文档清单，记录 doc_id ↔ 原文件名 ↔ 存储文件 ↔ 内容哈希 ↔ 所用向量模型
    MANIFEST_PATH: Path = ROOT_DIR / "data" / "manifest.json"
    EMBEDDING_CACHE_PATH: Path = ROOT_DIR / "data" / "embedding_cache.sqlite"
    ALLOWED_UPLOAD_SUFFIXES: List[str] = Field(default=[".txt", ".md", ".json", ".csv", ".pdf"])

    # 向量库选型: qdrant / memory
    VECTOR_DB_TYPE: str = Field(default_factory=lambda: os.getenv("VECTOR_DB_TYPE", "qdrant"))
    QDRANT_HOST: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    QDRANT_PORT: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    COLLECTION_NAME: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "enterprise_kb"))


class SecuritySettings(BaseModel):
    """安全与限额配置 (SEC-4 / SEC-6)"""
    # 管理令牌：设置后，修改配置等管理接口必须带请求头 X-Admin-Token；
    # 未设置时，管理接口只接受来自本机(127.0.0.1/::1)且来源(Origin)在白名单内的请求。
    ADMIN_TOKEN: str = Field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: _env_list(
        "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"))
    MAX_UPLOAD_MB: float = Field(default_factory=lambda: float(os.getenv("MAX_UPLOAD_MB", "20")))
    MAX_QUERY_CHARS: int = Field(default_factory=lambda: int(os.getenv("MAX_QUERY_CHARS", "2000")))
    MAX_HISTORY_MESSAGES: int = Field(default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "20")))
    MAX_HISTORY_CHARS: int = Field(default_factory=lambda: int(os.getenv("MAX_HISTORY_CHARS", "8000")))
    # 每个客户端 IP 每分钟允许的请求数，0 表示不限
    RATE_LIMIT_CHAT_PER_MIN: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_CHAT_PER_MIN", "30")))
    RATE_LIMIT_UPLOAD_PER_MIN: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_UPLOAD_PER_MIN", "10")))
    # 整次问答的总超时(秒)，防止多次 LLM 串行调用叠加出几分钟的等待
    REQUEST_TOTAL_TIMEOUT: float = Field(default_factory=lambda: float(os.getenv("REQUEST_TOTAL_TIMEOUT", "120")))


class AppSettings(BaseModel):
    """应用总体配置入口"""
    PROJECT_NAME: str = "Enterprise-Agentic-RAG"
    API_V1_STR: str = "/api/v1"
    # ENG-6：默认关闭调试模式（关闭热重载与 DEBUG 日志），开发时在 .env 中设置 DEBUG=true
    DEBUG: bool = Field(default_factory=lambda: _env_bool("DEBUG", "false"))

    # 子模块配置组合
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    rag: RAGHyperParams = Field(default_factory=RAGHyperParams)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    def ensure_directories(self):
        """确保运行时必要目录已创建"""
        self.storage.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.storage.EVAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)


# 全局单例
settings = AppSettings()
settings.ensure_directories()
