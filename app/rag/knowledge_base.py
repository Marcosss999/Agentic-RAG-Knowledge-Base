"""
知识库文档管理 (Knowledge Base Service)

一篇文档同时存在四个地方：磁盘原文件、BM25 倒排表、向量库、母块表。本模块以 doc_id 为主键，
负责让这四处同步增、删、改，并用文档清单 (manifest) 记录它们之间的对应关系。

对应缺陷追踪：
- DATA-1：重复上传先删后写；新增删除与列表功能；不同 doc_id 上传相同内容时拒绝（内容哈希去重）。
- DATA-2：原始文件按字节原样保存，上传与重启预热共用同一个文本抽取函数。
- DATA-6：重启后按清单重建，doc_id 保持不变（原实现用文件名主干当 doc_id，重启后 ID 会变）。
- SEC-2：doc_id 白名单校验；磁盘文件名由服务端生成（{doc_id}{后缀}），用户提供的文件名只存进元数据。
- API-2：清单记录建索引时所用的向量模型标识；模型变化后全部重建，避免新旧向量混在一个索引里。
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.core.security import SecurityError, safe_join, sanitize_filename, validate_doc_id
from app.rag.hybrid_retriever import HybridRetriever, hybrid_retriever, _PerLoopLock
from app.rag.loaders import extract_text

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


class DocumentNotFound(KeyError):
    pass


class DuplicateContentError(ValueError):
    def __init__(self, existing_doc_id: str):
        super().__init__(f"相同内容的文档已存在: {existing_doc_id}")
        self.existing_doc_id = existing_doc_id


class KnowledgeBase:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        upload_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.retriever = retriever or hybrid_retriever
        self.upload_dir = Path(upload_dir or settings.storage.UPLOAD_DIR)
        self.manifest_path = Path(manifest_path or settings.storage.MANIFEST_PATH)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._lock = _PerLoopLock()
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------ 清单
    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("documents"), dict):
                    return data
            except Exception as e:
                logger.error(f"[KnowledgeBase] 清单文件损坏，将按上传目录重新登记: {e}")
        return {"version": MANIFEST_VERSION, "embedding_signature": None, "documents": {}}

    def _save_manifest(self) -> None:
        """原子写入：先写临时文件再替换，避免写到一半崩溃导致清单损坏"""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    @property
    def documents(self) -> Dict[str, Dict[str, Any]]:
        return self._manifest["documents"]

    def list_documents(self) -> List[Dict[str, Any]]:
        docs = [{"doc_id": k, **{f: v[f] for f in (
            "doc_name", "size_bytes", "parent_chunks", "child_chunks", "created_at", "updated_at") if f in v}}
            for k, v in self.documents.items()]
        return sorted(docs, key=lambda d: d.get("updated_at", 0), reverse=True)

    # ------------------------------------------------------------------ 新增 / 替换
    def validate_upload(self, data: bytes, filename: str, doc_id: Optional[str] = None):
        """入库前的同步校验（文件名、类型、大小、doc_id）。后台入库时先在请求内调用它，校验失败直接返回 400。"""
        display_name = sanitize_filename(filename)
        suffix = Path(display_name).suffix.lower()
        if suffix not in settings.storage.ALLOWED_UPLOAD_SUFFIXES:
            raise SecurityError(f"不支持的文件类型: {suffix or '(无后缀)'}，仅支持 {settings.storage.ALLOWED_UPLOAD_SUFFIXES}")
        max_bytes = int(settings.security.MAX_UPLOAD_MB * 1024 * 1024)
        if len(data) > max_bytes:
            raise SecurityError(f"文件超过大小上限 {settings.security.MAX_UPLOAD_MB} MB")
        doc_id = validate_doc_id(doc_id) if doc_id else f"doc_{uuid.uuid4().hex[:12]}"
        return display_name, suffix, doc_id

    async def add_document(self, data: bytes, filename: str, doc_id: Optional[str] = None,
                           progress=None) -> Dict[str, Any]:
        """校验并入库一份文件；doc_id 已存在时整体替换。返回该文档的清单记录。
        :param progress: 可选回调 progress(stage, done, total)，用于后台入库显示进度"""
        display_name, suffix, doc_id = self.validate_upload(data, filename, doc_id)
        report = progress or (lambda stage, done, total: None)
        sha256 = hashlib.sha256(data).hexdigest()

        async with self._lock.get():
            for other_id, rec in self.documents.items():
                if other_id != doc_id and rec.get("sha256") == sha256:
                    raise DuplicateContentError(other_id)

            report("解析文本", 0, 1)
            raw_text = await asyncio.to_thread(extract_text, data, display_name)
            if not raw_text.strip():
                raise ValueError("文件中没有可抽取的文本（扫描版 PDF 需要先做 OCR）")

            stored_name = f"{doc_id}{suffix}"
            target = safe_join(self.upload_dir, stored_name)

            # 先建索引（最可能失败），成功后再落盘与更新清单；失败时旧版本不受影响
            parent_cnt, child_cnt = await self.retriever.index_document(
                raw_text=raw_text, doc_id=doc_id, doc_name=display_name,
                metadata={"doc_name": display_name, "file_size": len(data)},
                progress=lambda done, total: report("向量化", done, total),
            )
            report("写入索引", 1, 1)

            old = self.documents.get(doc_id)
            if old and old.get("stored_file") and old["stored_file"] != stored_name:
                self._remove_file(old["stored_file"])
            await asyncio.to_thread(target.write_bytes, data)

            now = time.time()
            record = {
                "doc_name": display_name,
                "stored_file": stored_name,
                "sha256": sha256,
                "size_bytes": len(data),
                "parent_chunks": parent_cnt,
                "child_chunks": child_cnt,
                "created_at": old.get("created_at", now) if old else now,
                "updated_at": now,
            }
            self.documents[doc_id] = record
            self._manifest["embedding_signature"] = llm_provider.embedding_signature
            self._save_manifest()
            return {"doc_id": doc_id, **record, "replaced": old is not None}

    # ------------------------------------------------------------------ 删除
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        validate_doc_id(doc_id)
        async with self._lock.get():
            rec = self.documents.get(doc_id)
            if rec is None:
                raise DocumentNotFound(doc_id)
            removed = await self.retriever.delete_document(doc_id)
            self._remove_file(rec.get("stored_file"))
            del self.documents[doc_id]
            self._save_manifest()
            return {"doc_id": doc_id, **removed}

    def _remove_file(self, stored_name: Optional[str]) -> None:
        if not stored_name:
            return
        try:
            safe_join(self.upload_dir, stored_name).unlink(missing_ok=True)
        except (SecurityError, OSError) as e:
            logger.warning(f"[KnowledgeBase] 删除文件 {stored_name} 失败: {e}")

    # ------------------------------------------------------------------ 启动重建 / 模型变更
    async def rebuild_from_disk(self) -> Dict[str, int]:
        """服务启动时调用：按清单重建全部索引；上传目录中未登记的旧文件会被补登记。"""
        async with self._lock.get():
            await self.retriever.clear()
            self._register_legacy_files()
            ok, failed = 0, 0
            for doc_id, rec in list(self.documents.items()):
                try:
                    path = safe_join(self.upload_dir, rec["stored_file"])
                    data = await asyncio.to_thread(path.read_bytes)
                    raw_text = await asyncio.to_thread(extract_text, data, rec["doc_name"])
                    p, c = await self.retriever.index_document(
                        raw_text=raw_text, doc_id=doc_id, doc_name=rec["doc_name"],
                        metadata={"doc_name": rec["doc_name"], "file_size": len(data)},
                    )
                    rec["parent_chunks"], rec["child_chunks"] = p, c
                    ok += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"[KnowledgeBase] 重建文档 {doc_id} 失败: {e}")
            self._manifest["embedding_signature"] = llm_provider.embedding_signature
            self._save_manifest()
            logger.info(f"[KnowledgeBase] 启动重建完成：成功 {ok} 篇，失败 {failed} 篇")
            return {"indexed": ok, "failed": failed}

    def _register_legacy_files(self) -> None:
        """把上传目录里清单之外的文件登记进清单（兼容引入清单之前上传的文件）"""
        known = {rec.get("stored_file") for rec in self.documents.values()}
        for f in sorted(self.upload_dir.iterdir()):
            if not f.is_file() or f.name in known:
                continue
            if f.suffix.lower() not in settings.storage.ALLOWED_UPLOAD_SUFFIXES:
                continue
            base = re.sub(r"[^A-Za-z0-9_\-]", "_", f.stem)[:56] or "doc"
            doc_id, n = base, 1
            while doc_id in self.documents:
                n += 1
                doc_id = f"{base}_{n}"
            data = f.read_bytes()
            now = f.stat().st_mtime
            self.documents[doc_id] = {
                "doc_name": f.name, "stored_file": f.name,
                "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                "parent_chunks": 0, "child_chunks": 0, "created_at": now, "updated_at": now,
            }
            logger.info(f"[KnowledgeBase] 补登记历史文件 {f.name} -> doc_id={doc_id}")

    async def ensure_embedding_consistency(self) -> bool:
        """向量模型变化后重建全部索引。返回是否发生了重建。"""
        current = llm_provider.embedding_signature
        if self._manifest.get("embedding_signature") == current:
            return False
        logger.warning(f"[KnowledgeBase] 向量模型已变化 "
                       f"({self._manifest.get('embedding_signature')} -> {current})，重建全部索引")
        await self.rebuild_from_disk()
        return True

    def stats(self) -> Dict[str, Any]:
        return {
            "documents": len(self.documents),
            "parent_chunks": len(self.retriever.parent_docs_store),
            "child_chunks": self.retriever.bm25.corpus_size,
            "embedding_signature": self._manifest.get("embedding_signature"),
        }


# 全局单例
knowledge_base = KnowledgeBase()
