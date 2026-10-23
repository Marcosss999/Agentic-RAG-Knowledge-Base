"""
文档文本抽取 (Document Loaders)
DATA-2：上传与重启预热原本各写了一套抽取逻辑，PDF 两边结果不一致（一边带页码标记、一边不带）。
现在两条路径都调用这里的同一个函数，保证同一份文件无论何时入库，切出来的块完全相同。
"""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentParseError(ValueError):
    """文件无法解析为文本"""


def extract_text(data: bytes, filename: str) -> str:
    """把原始文件字节转换为纯文本。

    - PDF：逐页抽取，每页前加 `--- [第 n 页] ---` 标记，跳过无文字的页（扫描件需另做 OCR）；
    - CSV：每一行改写成"列名：值"的形式，保证切块后每块都带表头含义；
    - 其他：优先 UTF-8，失败再按 GBK 解码（兼容 Windows 中文记事本）。
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = []
            for p_idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(f"--- [第 {p_idx} 页] ---\n{text}")
            return "\n\n".join(pages)
        except Exception as e:  # pypdf 的异常类型较多，这里统一转换
            raise DocumentParseError(f"PDF 解析失败: {e}") from e

    text = _decode(data)
    if suffix == ".csv":
        return _csv_to_text(text)
    return text


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _csv_to_text(text: str) -> str:
    """ENG-7：CSV 按纯文本切分时，表头只在第一块里，后面的块只剩一串数字、看不出含义。
    这里把每一行改写成"列名：值；列名：值"的一行文字，每行自带表头，切到哪一块都能读懂。"""
    import csv
    import io as _io
    rows = list(csv.reader(_io.StringIO(text)))
    if len(rows) < 2:
        return text
    header = [h.strip() or f"列{i + 1}" for i, h in enumerate(rows[0])]
    lines = []
    for n, row in enumerate(rows[1:], start=1):
        if not any(cell.strip() for cell in row):
            continue
        pairs = [f"{header[i] if i < len(header) else f'列{i + 1}'}：{cell.strip()}"
                 for i, cell in enumerate(row) if cell.strip()]
        lines.append(f"第{n}行｜" + "；".join(pairs))
    return "\n".join(lines)
