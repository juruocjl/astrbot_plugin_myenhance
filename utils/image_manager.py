from __future__ import annotations

import hashlib
import re
from pathlib import Path
from collections import OrderedDict
from typing import Any

import httpx

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow unavailable fallback
    Image = None


class ImageManager:
    """管理图片 ID 生成与基于 ID 的 LRU 缓存。"""

    def __init__(self, image_store_dir: Path, id_length: int = 16, download_timeout: float = 15.0):
        self.image_store_dir = Path(image_store_dir)
        self.image_store_dir.mkdir(parents=True, exist_ok=True)
        self.id_length = max(8, int(id_length))
        self.download_timeout = max(1.0, float(download_timeout))

    def build_image_id(self, image_bytes: bytes) -> str:
        if not image_bytes:
            return ""
        digest = hashlib.sha256(image_bytes).hexdigest()
        return digest[: self.id_length]

    def build_legacy_image_id(self, image_source: str) -> str:
        normalized = str(image_source or "").strip()
        if not normalized:
            return ""
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"legacy_{digest[: self.id_length]}"

    def build_keyword(self, content: str, max_len: int = 32) -> str:
        normalized = re.sub(r"\s+", " ", str(content or "").strip())
        if not normalized:
            return ""
        keyword = normalized[:max_len]
        return keyword.rstrip("，,;；:：。.!?！？ ")

    def get_entry(
        self,
        image_id: str,
        image_lru: OrderedDict[str, Any],
    ) -> dict[str, str] | None:
        key = str(image_id or "").strip()
        if not key:
            return None
        raw_value = image_lru.get(key)
        if raw_value is None:
            return None

        entry = self._coerce_entry(raw_value)
        image_lru[key] = entry
        image_lru.move_to_end(key)
        return entry

    def touch_image(
        self,
        image_source: str,
        image_lru: OrderedDict[str, Any],
        max_size: int,
    ) -> str:
        # 保留同步接口的兼容性；同步接口不再负责下载。
        source = str(image_source or "").strip()
        if not source:
            return ""

        found_id = self._find_image_id_by_url(source, image_lru)
        if not found_id:
            return ""
        self.get_entry(found_id, image_lru)
        self._trim(image_lru, max_size)
        return found_id

    async def ensure_image(
        self,
        image_source: str,
        image_lru: OrderedDict[str, Any],
        max_size: int,
    ) -> str:
        source = str(image_source or "").strip()
        if not source:
            return ""

        found_id = self._find_image_id_by_url(source, image_lru)
        if found_id:
            self.get_entry(found_id, image_lru)
            self._trim(image_lru, max_size)
            return found_id

        image_bytes, content_type = await self._download_image(source)
        if not image_bytes:
            return ""

        image_id = self.build_image_id(image_bytes)
        if not image_id:
            return ""

        entry = self._coerce_entry(image_lru.get(image_id))
        if not entry.get("url"):
            entry["url"] = source

        extension = self._guess_extension(image_bytes, content_type, source)
        image_file = self.image_store_dir / f"{image_id}{extension}"
        if not image_file.exists():
            image_file.write_bytes(image_bytes)
        entry["local_path"] = str(image_file)
        thumb_path = self._build_thumbnail(image_file, image_id)
        if thumb_path:
            entry["thumb_path"] = str(thumb_path)

        image_lru[image_id] = entry
        image_lru.move_to_end(image_id)
        self._trim(image_lru, max_size)
        return image_id

    def set_description(
        self,
        image_id: str,
        image_lru: OrderedDict[str, Any],
        keyword: str,
        content: str,
        max_size: int,
    ) -> dict[str, str] | None:
        key = str(image_id or "").strip()
        if not key:
            return None

        entry = self._coerce_entry(image_lru.get(key))
        normalized_content = re.sub(r"\s+", " ", str(content or "").strip())
        if not normalized_content:
            return None

        normalized_keyword = re.sub(r"\s+", " ", str(keyword or "").strip())
        if not normalized_keyword:
            normalized_keyword = self.build_keyword(normalized_content)

        entry["keyword"] = normalized_keyword
        entry["content"] = normalized_content

        image_lru[key] = entry
        image_lru.move_to_end(key)
        self._trim(image_lru, max_size)
        return entry

    def build_inject_tag(self, image_id: str, keyword: str = "") -> str:
        normalized_id = str(image_id or "").strip()
        if not normalized_id:
            return "[image]"

        normalized_keyword = re.sub(r"\s+", " ", str(keyword or "").strip())
        if normalized_keyword:
            return f"[image:{normalized_id},{normalized_keyword}]"
        return f"[image:{normalized_id}]"

    def _coerce_entry(self, raw_value: Any) -> dict[str, str]:
        if isinstance(raw_value, dict):
            url = str(raw_value.get("url") or "").strip()
            keyword = re.sub(r"\s+", " ", str(raw_value.get("keyword") or "").strip())
            content = re.sub(r"\s+", " ", str(raw_value.get("content") or "").strip())
            local_path = str(raw_value.get("local_path") or "").strip()
            thumb_path = str(raw_value.get("thumb_path") or "").strip()
            if content and not keyword:
                keyword = self.build_keyword(content)
            return {
                "url": url,
                "keyword": keyword,
                "content": content,
                "local_path": local_path,
                "thumb_path": thumb_path,
            }

        legacy_content = re.sub(r"\s+", " ", str(raw_value or "").strip())
        return {
            "url": "",
            "keyword": self.build_keyword(legacy_content),
            "content": legacy_content,
            "local_path": "",
            "thumb_path": "",
        }

    def _find_image_id_by_url(self, image_source: str, image_lru: OrderedDict[str, Any]) -> str:
        source = str(image_source or "").strip()
        if not source:
            return ""
        for image_id, raw_value in image_lru.items():
            if not isinstance(raw_value, dict):
                continue
            if str(raw_value.get("url") or "").strip() == source:
                return str(image_id or "").strip()
        return ""

    async def _download_image(self, image_source: str) -> tuple[bytes, str]:
        source = str(image_source or "").strip()
        if not source:
            return b"", ""

        if source.lower().startswith(("http://", "https://")):
            try:
                timeout = httpx.Timeout(self.download_timeout)
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.get(source)
                if resp.status_code >= 400:
                    return b"", ""
                return resp.content or b"", str(resp.headers.get("content-type") or "")
            except Exception:
                return b"", ""

        try:
            file_path = Path(source)
            if file_path.exists() and file_path.is_file():
                return file_path.read_bytes(), ""
        except Exception:
            return b"", ""

        return b"", ""

    def _guess_extension(self, image_bytes: bytes, content_type: str, image_source: str) -> str:
        ctype = str(content_type or "").lower().split(";", 1)[0].strip()
        ctype_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }
        if ctype in ctype_map:
            return ctype_map[ctype]

        header = image_bytes[:16]
        if header.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return ".gif"
        if header.startswith(b"RIFF") and b"WEBP" in image_bytes[:32]:
            return ".webp"
        if header.startswith(b"BM"):
            return ".bmp"

        source = str(image_source or "").strip().lower()
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"):
            if source.endswith(ext):
                return ".jpg" if ext == ".jpeg" else ext
        return ".img"

    def _build_thumbnail(self, image_file: Path, image_id: str, size: tuple[int, int] = (256, 256)) -> Path | None:
        if Image is None:
            return None
        try:
            thumb_file = self.image_store_dir / f"{image_id}_thumb.jpg"
            if thumb_file.exists():
                return thumb_file

            with Image.open(image_file) as img:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.thumbnail(size)
                img.save(thumb_file, format="JPEG", quality=85, optimize=True)
            return thumb_file
        except Exception:
            return None

    def _trim(self, image_lru: OrderedDict[str, Any], max_size: int) -> None:
        limit = max(1, int(max_size))
        while len(image_lru) > limit:
            image_lru.popitem(last=False)
