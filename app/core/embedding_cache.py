"""
磁盘 Embedding 缓存 (DATA-7)

问题：索引放在内存里，服务每次重启都要按清单重建；使用在线向量模型时，等于每次重启都把整个知识库重新
发给 Embedding 接口计算一遍（343 个子块的论文约 35 次请求），既慢又花钱。

做法：把"文本 → 向量"的结果存进本地 SQLite。键 = sha256(向量模型标识 + 文本)，所以：
- 同一段文字、同一个模型，只会计算一次；重启后 BM25 和母块从原文重新切分（纯 CPU，很快），向量直接读缓存；
- 换了向量模型，标识不同，自然不会误用旧向量（与 API-2 的签名机制一致）。

为什么不直接持久化整个索引：BM25、母块表、向量库三份结构各自落盘，还要保证彼此一致，复杂度高；
而它们里面唯一"贵"的只有向量，缓存向量就解决了主要问题。代价是重启时仍需重新切分与构建 BM25。
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, dim INTEGER, v BLOB)")
        self._conn.commit()

    @staticmethod
    def key(signature: str, text: str) -> str:
        return hashlib.sha256(f"{signature}\n{text}".encode("utf-8")).hexdigest()

    def get_many(self, signature: str, texts: List[str]) -> Dict[int, List[float]]:
        """返回 {输入下标: 向量}，只包含命中的条目"""
        keys = [self.key(signature, t) for t in texts]
        found: Dict[str, List[float]] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(chunk))})"
                for k, blob in self._conn.execute(q, chunk):
                    found[k] = np.frombuffer(blob, dtype=np.float32).tolist()
        return {i: found[k] for i, k in enumerate(keys) if k in found}

    def put_many(self, signature: str, texts: List[str], vectors: List[List[float]]) -> None:
        rows = [(self.key(signature, t), len(v), np.asarray(v, dtype=np.float32).tobytes())
                for t, v in zip(texts, vectors)]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO emb (k, dim, v) VALUES (?, ?, ?)", rows)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM emb").fetchone()[0]


_cache: Optional[EmbeddingCache] = None
_cache_lock = threading.Lock()


def get_cache() -> Optional[EmbeddingCache]:
    """按配置返回全局缓存；关闭时返回 None"""
    global _cache
    from app.core.config import settings
    if not settings.embedding.CACHE_ENABLED:
        return None
    with _cache_lock:
        if _cache is None or _cache.path != Path(settings.storage.EMBEDDING_CACHE_PATH):
            if _cache is not None:
                _cache.close()
            _cache = EmbeddingCache(settings.storage.EMBEDDING_CACHE_PATH)
        return _cache
