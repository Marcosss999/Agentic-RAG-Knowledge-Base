"""
BM25 稀疏倒排索引与关键词检索模块 (BM25 Sparse Index)
职责：
1. 解决密集向量检索（Dense Vector）对专有名词、产品型号、错误码（如 GTE-3000、CVE-2024-1234）不敏感的致命缺陷。
2. 实现工业级 Okapi BM25 算法（文档长度归一化与词频饱和度控制）。
3. 具备自适应分词机制：优先调用 jieba 精确分词；未安装时自动平滑回退至中英文混合分词器。
4. 返回标准化的 SearchResultItem 实体，保留原始 BM25 分值以供后续 RRF 融合。
"""

import math
import re
import logging
from collections import Counter
from typing import List, Dict, Any, Optional
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


class BM25Index:
    """
    工业级 Okapi BM25 倒排索引引擎
    超参数说明：
    - k1 (1.2 ~ 2.0): 词频饱和度参数。控制词频增加对得分提升的衰减速度。
    - b (0.75): 文档长度归一化惩罚参数。惩罚过长文本因词数多而带来的虚高得分。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # 核心索引存储结构
        self.corpus_size: int = 0                  # 文档/切片总数 N
        self.avgdl: float = 0.0                    # 平均文档长度 (Token 数)
        self.doc_lengths: Dict[str, int] = {}      # chunk_id -> 该块包含的 token 数量
        self.doc_id_to_chunk: Dict[str, ChildChunk] = {} # chunk_id -> 原始切片实体

        # 倒排索引表: token -> {chunk_id: tf (该词在该文档出现的频次)}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # 词频统计: token -> df (包含该词的文档总数)
        self.df: Dict[str, int] = {}
        # 预计算 IDF: token -> idf
        self.idf: Dict[str, float] = {}

    def build_index(self, chunks: List[ChildChunk]):
        """
        全量或增量构建 BM25 倒排索引
        :param chunks: 子切片实体列表
        """
        if not chunks:
            return

        total_length = 0

        for chunk in chunks:
            cid = chunk.chunk_id
            self.doc_id_to_chunk[cid] = chunk

            # 分词
            tokens = Tokenizer.tokenize(chunk.content)
            doc_len = len(tokens)
            self.doc_lengths[cid] = doc_len
            total_length += doc_len

            # 统计当前文档内部的词频 (TF)
            tf_counter = Counter(tokens)
            for token, freq in tf_counter.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][cid] = freq

        self.corpus_size = len(self.doc_id_to_chunk)
        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # 重新计算所有词的 IDF (逆文档频率)
        self._calculate_idf()
        logger.info(f"[BM25Index] 索引构建完成，收录切片数: {self.corpus_size}, 词项数: {len(self.inverted_index)}")

    def _calculate_idf(self):
        """
        计算平滑 IDF：
        IDF = ln( (N - df + 0.5) / (df + 0.5) + 1 )
        加 1 保证极端高频词的 IDF 值始终大于 0，避免出现负权重。
        """
        N = self.corpus_size
        for token, postings in self.inverted_index.items():
            df_val = len(postings)
            self.df[token] = df_val
            # Lucene / Okapi 平滑公式
            self.idf[token] = math.log((N - df_val + 0.5) / (df_val + 0.5) + 1.0)

    def search(self, query: str, top_k: int = None) -> List[SearchResultItem]:
        """
        执行 BM25 关键词检索打分
        :param query: 搜索问题或实体词
        :param top_k: 期望返回的最优结果数
        :return: 按照 BM25 相关度得分从高到低排序的标准结果列表
        """
        limit = top_k or settings.rag.SPARSE_TOP_K
        query_tokens = Tokenizer.tokenize(query)

        if not query_tokens or self.corpus_size == 0:
            return []

        # 累加每个文档对查询词的 BM25 贡献值
        scores: Dict[str, float] = Counter()

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf_val = self.idf.get(token, 0.0)
            postings = self.inverted_index[token]

            for cid, tf in postings.items():
                doc_len = self.doc_lengths.get(cid, 0)
                # BM25 分数公式
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                scores[cid] += idf_val * (numerator / (denominator or 1.0))

        # 筛选最高分结果
        best_matches = scores.most_common(limit)

        results: List[SearchResultItem] = []
        for rank, (cid, bm25_score) in enumerate(best_matches, start=1):
            chunk = self.doc_id_to_chunk[cid]
            item = SearchResultItem(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_name,
                parent_id=chunk.parent_id,
                content=chunk.content,
                score=round(float(bm25_score), 4),
                source=RetrievalSource.SPARSE_BM25,
                rank=rank,
                metadata={
                    **chunk.metadata,
                    "raw_bm25_score": round(float(bm25_score), 4)
                }
            )
            results.append(item)

        return results

    def clear(self):
        """清空索引"""
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths.clear()
        self.doc_id_to_chunk.clear()
        self.inverted_index.clear()
        self.df.clear()
        self.idf.clear()
