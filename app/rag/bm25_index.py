"""
BM25 稀疏倒排索引与关键词检索模块 (BM25 Sparse Index)
职责：
1. 解决密集向量检索（Dense Vector）对专有名词、产品型号、错误码（如 GTE-3000、CVE-2024-1234）不敏感的致命缺陷。
2. 实现 Okapi BM25 算法（文档长度归一化与词频饱和度控制）。
3. 具备自适应分词机制：优先调用 jieba 精确分词；未安装时自动平滑回退至中英文混合分词器。
4. 返回标准化的 SearchResultItem 实体，保留原始 BM25 分值以供后续 RRF 融合。
"""

import math
import re
import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict
from app.core.config import settings
from app.models.schemas import ChildChunk, SearchResultItem, RetrievalSource

logger = logging.getLogger(__name__)

# 尝试导入 jieba 中文分词库
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.info("[BM25Index] 未检测到 jieba 分词库，自动启用内置自适应正则分词器。")


class Tokenizer:
    """中英文混合分词处理器"""

    # 标点符号与无意义停用字符过滤正则
    CLEAN_PATTERN = re.compile(r"[\s\.,!?;:\"'()\[\]{}《》【】、，。！？；：“”‘’]+")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        if not text:
            return []

        text = text.lower().strip()

        if JIEBA_AVAILABLE:
            # 采用搜索引擎分词模式，对长词进行粒度切分 (如把 GTE3000 切出 gte 和 3000)
            tokens = [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
            return [t for t in tokens if not cls.CLEAN_PATTERN.fullmatch(t)]

        # 内置分词备选方案：提取连续的英文/数字/下划线单词，中文按单字与双字滑动提取
        tokens: List[str] = []
        # 1. 抽取英文字词、型号 (如 gte-3000, api_v1, 1024)
        words = re.findall(r"[a-z0-9_\-]+", text)
        tokens.extend(words)

        # 2. 提取中文字符
        chinese_chars = re.findall(r"[\u4e00-\u9fa5]", text)
        tokens.extend(chinese_chars)
        # 补充双字词组合 (Bi-gram) 提升中文短语召回率
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])

        return tokens


@dataclass(frozen=True)
class _BM25Snapshot:
    """一次完整、不可变的索引快照。检索只读快照；写入时构建新快照后整体替换引用。"""
    chunks: Dict[str, ChildChunk] = field(default_factory=dict)        # chunk_id -> 子块实体
    doc_lengths: Dict[str, int] = field(default_factory=dict)          # chunk_id -> token 数
    inverted_index: Dict[str, Dict[str, int]] = field(default_factory=dict)  # token -> {chunk_id: tf}
    idf: Dict[str, float] = field(default_factory=dict)
    avgdl: float = 0.0

    @property
    def corpus_size(self) -> int:
        return len(self.chunks)


class BM25Index:
    """
    Okapi BM25 倒排索引

    超参数：
    - k1 (1.2 ~ 2.0): 词频饱和度。词频越高得分增长越慢。
    - b (0.75): 文档长度归一化强度。惩罚长文本因词多而虚高的得分。

    并发设计 (CONC-1)：
    原实现在事件循环里原地修改字典，同时检索在 `asyncio.to_thread` 的线程里遍历同一批字典，
    实测并发时出现 `RuntimeError: dictionary changed size during iteration`。
    现在采用"写时复制 + 原子替换"：
    - 检索开始时取一次 `self._snap` 引用，之后只读这份不可变快照，读操作完全不需要加锁；
    - 写操作在锁内基于当前数据构建一份全新的快照，最后一步才替换引用（Python 中引用赋值是原子的）。
    代价是每次写入都要重建倒排表（O(总 token 数)），对中小规模知识库可以接受；规模变大时应换成
    Elasticsearch / OpenSearch 等专门的倒排引擎。

    正确性 (DATA-3)：
    原实现增量写入时 avgdl 只用新批次的长度计算，且替换同一 chunk 时旧词条残留。
    现在每次都从"全部 chunk 的 token 缓存"重建，avgdl 与倒排表天然与全量重建一致。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._snap = _BM25Snapshot()
        self._tokens: Dict[str, List[str]] = {}   # chunk_id -> 分词结果缓存（只在写锁内修改）
        self._write_lock = threading.Lock()

    # ---------------- 只读视图（兼容旧代码与测试） ----------------
    @property
    def corpus_size(self) -> int:
        return self._snap.corpus_size

    @property
    def avgdl(self) -> float:
        return self._snap.avgdl

    @property
    def doc_lengths(self) -> Dict[str, int]:
        return self._snap.doc_lengths

    @property
    def inverted_index(self) -> Dict[str, Dict[str, int]]:
        return self._snap.inverted_index

    @property
    def doc_id_to_chunk(self) -> Dict[str, ChildChunk]:
        return self._snap.chunks

    # ---------------- 写操作 ----------------
    def build_index(self, chunks: List[ChildChunk]):
        """增量写入（同 chunk_id 视为替换）。保留旧方法名以兼容调用方。"""
        self.add_chunks(chunks)

    def add_chunks(self, chunks: List[ChildChunk]) -> None:
        if not chunks:
            return
        with self._write_lock:
            new_chunks = dict(self._snap.chunks)
            for c in chunks:
                new_chunks[c.chunk_id] = c
                self._tokens[c.chunk_id] = Tokenizer.tokenize(c.content)
            self._rebuild(new_chunks)

    def remove_chunks(self, chunk_ids: List[str]) -> int:
        with self._write_lock:
            new_chunks = dict(self._snap.chunks)
            removed = 0
            for cid in chunk_ids:
                if new_chunks.pop(cid, None) is not None:
                    removed += 1
                self._tokens.pop(cid, None)
            if removed:
                self._rebuild(new_chunks)
            return removed

    def remove_document(self, doc_id: str) -> int:
        """删除某个文档的全部子块，返回删除数量"""
        ids = [cid for cid, c in self._snap.chunks.items() if c.doc_id == doc_id]
        return self.remove_chunks(ids)

    def clear(self):
        with self._write_lock:
            self._tokens.clear()
            self._snap = _BM25Snapshot()

    def _rebuild(self, chunks: Dict[str, ChildChunk]) -> None:
        """基于 token 缓存构建新快照并原子替换（调用方须持有写锁）"""
        inverted: Dict[str, Dict[str, int]] = {}
        doc_lengths: Dict[str, int] = {}
        total = 0
        for cid in chunks:
            tokens = self._tokens.get(cid, [])
            doc_lengths[cid] = len(tokens)
            total += len(tokens)
            for token, freq in Counter(tokens).items():
                inverted.setdefault(token, {})[cid] = freq
        n = len(chunks)
        # 平滑 IDF：ln((N - df + 0.5) / (df + 0.5) + 1)，+1 保证高频词权重不为负
        idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in inverted.items()}
        self._snap = _BM25Snapshot(
            chunks=chunks, doc_lengths=doc_lengths, inverted_index=inverted,
            idf=idf, avgdl=(total / n if n else 0.0),
        )
        logger.info(f"[BM25Index] 索引已重建，切片数: {n}, 词项数: {len(inverted)}")

    # ---------------- 读操作 ----------------
    def search(self, query: str, top_k: int = None) -> List[SearchResultItem]:
        """执行 BM25 关键词检索，按得分降序返回"""
        limit = top_k or settings.rag.SPARSE_TOP_K
        snap = self._snap  # 只取一次引用，之后整个检索都基于这份不可变快照
        query_tokens = Tokenizer.tokenize(query)
        if not query_tokens or snap.corpus_size == 0:
            return []

        scores: Dict[str, float] = Counter()
        avgdl = snap.avgdl or 1.0
        for token in query_tokens:
            postings = snap.inverted_index.get(token)
            if not postings:
                continue
            idf_val = snap.idf.get(token, 0.0)
            for cid, tf in postings.items():
                doc_len = snap.doc_lengths.get(cid, 0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                scores[cid] += idf_val * (numerator / (denominator or 1.0))

        results: List[SearchResultItem] = []
        for rank, (cid, bm25_score) in enumerate(scores.most_common(limit), start=1):
            chunk = snap.chunks[cid]
            results.append(SearchResultItem(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_name,
                parent_id=chunk.parent_id,
                content=chunk.content,
                score=round(float(bm25_score), 4),
                source=RetrievalSource.SPARSE_BM25,
                rank=rank,
                metadata={**chunk.metadata, "raw_bm25_score": round(float(bm25_score), 4)}
            ))
        return results
