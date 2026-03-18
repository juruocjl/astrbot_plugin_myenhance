from __future__ import annotations

import json
import random
from pathlib import Path

from astrbot.api import logger


class MemeManager:
    """管理 meme 标签到图片 ID 的映射。"""

    def __init__(self, store_file: Path, max_per_tag: int = 200):
        self.store_file = Path(store_file)
        self.max_per_tag = max(1, int(max_per_tag))
        self.memes_by_tag: dict[str, list[str]] = {}
        self._load()

    def add_meme(self, image_id: str, tag: str) -> tuple[bool, str]:
        normalized_id = str(image_id or "").strip()
        normalized_tag = self._normalize_tag(tag)
        if not normalized_id:
            return False, "image_id is empty"
        if not normalized_tag:
            return False, "tag is empty"

        ids = self.memes_by_tag.setdefault(normalized_tag, [])
        if normalized_id in ids:
            return True, "already exists"

        ids.append(normalized_id)
        if len(ids) > self.max_per_tag:
            del ids[: len(ids) - self.max_per_tag]
        self._save()
        return True, "added"

    def get_random_meme_id(self, tag: str) -> str | None:
        normalized_tag = self._normalize_tag(tag)
        if not normalized_tag:
            return None
        ids = self.memes_by_tag.get(normalized_tag) or []
        if not ids:
            return None
        return random.choice(ids)

    def delete_meme(self, image_id: str, tag: str) -> tuple[bool, str]:
        normalized_id = str(image_id or "").strip()
        normalized_tag = self._normalize_tag(tag)
        if not normalized_id:
            return False, "image_id is empty"
        if not normalized_tag:
            return False, "tag is empty"

        ids = self.memes_by_tag.get(normalized_tag) or []
        if normalized_id not in ids:
            return False, "not found"
        ids = [item for item in ids if item != normalized_id]
        if ids:
            self.memes_by_tag[normalized_tag] = ids
        else:
            self.memes_by_tag.pop(normalized_tag, None)
        self._save()
        return True, "deleted"

    def list_tags(self) -> list[str]:
        return sorted(self.memes_by_tag.keys())

    def _normalize_tag(self, tag: str) -> str:
        return str(tag or "").strip().lower()

    def _load(self) -> None:
        if not self.store_file.exists():
            return
        try:
            data = json.loads(self.store_file.read_text(encoding="utf-8"))
            raw = data.get("memes_by_tag", {}) if isinstance(data, dict) else {}
            if not isinstance(raw, dict):
                return
            loaded: dict[str, list[str]] = {}
            for raw_tag, raw_ids in raw.items():
                tag = self._normalize_tag(raw_tag)
                if not tag or not isinstance(raw_ids, list):
                    continue
                ids = [str(item).strip() for item in raw_ids if str(item).strip()]
                if ids:
                    loaded[tag] = ids[-self.max_per_tag :]
            self.memes_by_tag = loaded
        except Exception as exc:
            logger.warning("[myenhance] failed to load meme store: %s", exc)

    def _save(self) -> None:
        try:
            payload = {"memes_by_tag": self.memes_by_tag}
            self.store_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("[myenhance] failed to save meme store: %s", exc)
