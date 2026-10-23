"""
安全工具集 (Security Utilities)
对应缺陷追踪 SEC-1 / SEC-2 / SEC-4 / SEC-6。

核心原则：凡是来自外部的输入（请求参数、上传文件名、LLM 输出、访问 localhost 的浏览器）都不可信。
- 路径：白名单校验 + 解析后确认仍在允许目录内（防路径穿越）。
- 管理接口：令牌或"本机 + 可信来源"双重校验（防恶意网页借浏览器调用本机服务）。
- 资源：限流 + 大小上限（防 DoS 与成本失控）。
"""

import re
import time
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Iterable
from urllib.parse import urlparse

DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
REPORT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,80}\.json$")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w\-. ]", re.UNICODE)
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class SecurityError(ValueError):
    """输入未通过安全校验（调用方通常转换为 HTTP 400/401/403）"""


def validate_doc_id(doc_id: str) -> str:
    """doc_id 只允许字母、数字、下划线、连字符，长度 1~64"""
    if not doc_id or not DOC_ID_PATTERN.fullmatch(doc_id):
        raise SecurityError("doc_id 只能包含字母、数字、下划线和连字符，长度 1~64")
    return doc_id


def sanitize_filename(filename: str, max_len: int = 120) -> str:
    """去掉路径成分与危险字符，只保留展示用的安全文件名（原名仅存元数据，不参与拼路径）"""
    name = Path(str(filename).replace("\\", "/")).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    if not name:
        raise SecurityError("无效的文件名")
    if len(name) > max_len:
        stem, dot, suffix = name.rpartition(".")
        name = (stem[: max_len - len(suffix) - 1] + "." + suffix) if dot else name[:max_len]
    return name


def safe_join(base_dir: Path, name: str) -> Path:
    """把 name 拼到 base_dir 下，并确认解析后的路径没有逃出 base_dir"""
    base = Path(base_dir).resolve()
    target = (base / name).resolve()
    if target.parent != base:
        raise SecurityError(f"非法路径: {name}")
    return target


def validate_report_name(name: Optional[str]) -> Optional[str]:
    """评测报告名：只允许 `字母数字_-.json`，不允许任何目录成分"""
    if name is None:
        return None
    if not REPORT_NAME_PATTERN.fullmatch(name):
        raise SecurityError("report_name 只能是形如 my_report.json 的文件名（字母、数字、下划线、连字符）")
    return name


def validate_base_url(url: str) -> str:
    """模型网关地址：必须是 https；只有本机地址（如本地部署的模型）允许 http"""
    parsed = urlparse(url.strip())
    if parsed.scheme == "https" and parsed.netloc:
        return url.strip()
    if parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS:
        return url.strip()
    raise SecurityError("BASE_URL 必须是 https 地址（本机地址可用 http）")


def origin_allowed(origin: Optional[str], allowed: Iterable[str]) -> bool:
    """没有 Origin 头（非浏览器发起，如 curl/脚本）视为允许；有则必须在白名单内"""
    if not origin:
        return True
    return origin.rstrip("/") in {o.rstrip("/") for o in allowed}


def check_admin_access(
    client_host: Optional[str],
    origin: Optional[str],
    provided_token: Optional[str],
    admin_token: str,
    allowed_origins: Iterable[str],
) -> None:
    """管理接口访问控制。

    - 配置了 ADMIN_TOKEN：必须带正确的 X-Admin-Token；
    - 未配置：只允许本机请求，且浏览器来源必须在 CORS 白名单内。
      第二条专门防御"你开着服务时访问恶意网页，网页用你的浏览器调本机接口"这种攻击。
    """
    import hmac

    if not origin_allowed(origin, allowed_origins):
        raise SecurityError("forbidden_origin")
    if admin_token:
        if not provided_token or not hmac.compare_digest(provided_token, admin_token):
            raise SecurityError("invalid_token")
        return
    if client_host not in LOCAL_HOSTS:
        raise SecurityError("admin_local_only")


class RateLimiter:
    """按 key（通常是客户端 IP）的滑动窗口限流器，线程安全。limit<=0 表示不限流。"""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True
