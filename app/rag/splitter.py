"""
父子分块切分器与滑动窗口算法 (Parent-Child Text Splitter)
职责：
1. 解决 RAG 领域经典的“检索精准度”与“上下文连贯性”的固有矛盾。
2. 遵循中英文自然句读分词优先级（段落 -> 换行 -> 句号/问号 -> 空格），避免机械切断语义。
3. 实现母块（Parent Chunk）与子块（Child Chunk）的双层构建与双向 ID 绑定映射。
"""

import bisect
import re
import uuid
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.models.schemas import ParentChunk, ChildChunk


PAGE_MARK = re.compile(r"--- \[第 (\d+) 页\] ---")


def _page_at(page_marks, pos: int):
    """返回 pos 所在的页码：位置之前最近的一个页码标记；在第一个标记之前则返回 None"""
    idx = bisect.bisect_right([p for p, _ in page_marks], pos) - 1
    return page_marks[idx][1] if idx >= 0 else None


class RecursiveTextSplitter:
    """
    自研纯 Python 递归字符分块器
    优先按照段落、句子标点递归拆分，保证自然语义完整
    """

    DEFAULT_SEPARATORS = [
        "\n\n",      # 优先双换行（段落边界）
        "\n",        # 单换行
        "。", "！", "？", # 中文句读标点
        ". ", "! ", "? ", # 英文标点
        "；", ";",    # 分号
        " ",         # 空格
        ""           # 兜底按字切分
    ]

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        """将长文本递归切分成重叠的片段"""
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        # 寻找当前文本中存在的最高优先级分隔符
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # 按选定的分隔符切分
        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                # 依然超长，则用更低优先级的符号继续向下递归
                if new_separators:
                    sub_splits = self._split(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # 已经没有更低符号，硬切
                    for idx in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[idx: idx + self.chunk_size])

        # 滑动窗口拼接与重叠处理 (Overlap Stitching)
        current_chunk = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if separator else 0)
            if current_len + piece_len > self.chunk_size:
                if current_chunk:
                    merged = separator.join(current_chunk) if separator else "".join(current_chunk)
                    final_chunks.append(merged)
                    # 保留重叠部分
                    while current_chunk and current_len > self.chunk_overlap:
                        removed = current_chunk.pop(0)
                        current_len -= len(removed) + (len(separator) if separator else 0)

            current_chunk.append(piece)
            current_len += piece_len

        if current_chunk:
            merged = separator.join(current_chunk) if separator else "".join(current_chunk)
            final_chunks.append(merged)

        return [c.strip() for c in final_chunks if c.strip()]


class ParentChildSplitter:
    """
    父子分块管理器 (Parent-Child Chunking)
    双塔检索架构的解药：
    - 母块 (Parent Chunk)：大颗粒度 (~1500字)，留存完整的叙事背景与因果推导，供 LLM 生成高质量答案。
    - 子块 (Child Chunk)：小颗粒度 (~300字)，语义密度极高，专门提供给向量库与 BM25 做高灵敏度检索命中。
    """

    def __init__(
        self,
        parent_chunk_size: int = None,
        parent_chunk_overlap: int = None,
        child_chunk_size: int = None,
        child_chunk_overlap: int = None
    ):
        # 允许外部注入，未注入时回退至全局配置参数。
        # DATA-5：原来用 `x or default`，传入 0 时 `0 or 150` 得到 150，overlap=0 永远无法生效。
        def pick(value, default):
            return default if value is None else value

        self.p_size = pick(parent_chunk_size, settings.rag.PARENT_CHUNK_SIZE)
        self.p_overlap = pick(parent_chunk_overlap, settings.rag.PARENT_CHUNK_OVERLAP)
        self.c_size = pick(child_chunk_size, settings.rag.CHILD_CHUNK_SIZE)
        self.c_overlap = pick(child_chunk_overlap, settings.rag.CHILD_CHUNK_OVERLAP)

        self.parent_splitter = RecursiveTextSplitter(
            chunk_size=self.p_size,
            chunk_overlap=self.p_overlap
        )
        self.child_splitter = RecursiveTextSplitter(
            chunk_size=self.c_size,
            chunk_overlap=self.c_overlap
        )

    def process_document(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        metadata: Dict[str, Any] = None
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        核心分块与双向绑定映射入口
        :param raw_text: 原始全文字符串
        :param doc_id: 业务文档唯一标识
        :param doc_name: 文档名称
        :param metadata: 附加元数据 (如作者、发布时间、部门权限等)
        :return: (母块实体列表, 子块实体列表)
        """
        base_meta = metadata or {}
        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []

        # 阶段一：先切出大粒度母块
        raw_parent_texts = self.parent_splitter.split_text(raw_text)

        # ENG-7：PDF 抽取文本带有 "--- [第 n 页] ---" 标记，记下每个标记的位置，用来给子块标注页码
        page_marks = [(m.start(), int(m.group(1))) for m in PAGE_MARK.finditer(raw_text)]
        parent_cursor = 0

        global_child_index = 0

        for p_idx, p_text in enumerate(raw_parent_texts):
            p_start = raw_text.find(p_text[:60], parent_cursor) if page_marks else -1
            if p_start >= 0:
                parent_cursor = p_start + 1
            child_cursor = 0
            parent_id = f"parent_{doc_id}_{p_idx}_{uuid.uuid4().hex[:8]}"
            child_ids_in_this_parent: List[str] = []

            # 阶段二：针对当前母块，细切出高密度子块
            raw_child_texts = self.child_splitter.split_text(p_text)

            for c_text in raw_child_texts:
                page = None
                if p_start >= 0:
                    c_off = p_text.find(c_text[:40], child_cursor)
                    if c_off >= 0:
                        child_cursor = c_off + 1
                        page = _page_at(page_marks, p_start + c_off + len(c_text) // 2)
                child_id = f"child_{doc_id}_{global_child_index}_{uuid.uuid4().hex[:8]}"
                child_ids_in_this_parent.append(child_id)

                child_entity = ChildChunk(
                    chunk_id=child_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    parent_id=parent_id,
                    content=c_text,
                    chunk_index=global_child_index,
                    metadata={
                        **base_meta,
                        "parent_id": parent_id,
                        "parent_index": p_idx,
                        "char_count": len(c_text),
                        **({"page": page} if page is not None else {}),
                    }
                )
                child_chunks.append(child_entity)
                global_child_index += 1

            # 阶段三：生成母块实体，记录其包含的子块列表
            parent_entity = ParentChunk(
                chunk_id=parent_id,
                doc_id=doc_id,
                doc_name=doc_name,
                content=p_text,
                chunk_index=p_idx,
                child_ids=child_ids_in_this_parent,
                metadata={
                    **base_meta,
                    "children_count": len(child_ids_in_this_parent),
                    "char_count": len(p_text)
                }
            )
            parent_chunks.append(parent_entity)

        return parent_chunks, child_chunks
