from __future__ import annotations
import json
from pathlib import Path
from collections import deque
from typing import Any, Deque, OrderedDict
from astrbot.api import logger

from .image_manager import ImageManager

class CacheManager:
    """管理持久化缓存状态。"""
    def __init__(self, cache_file: Path, max_history: int, event_cache_size: int, image_url_cache_size: int):
        self.cache_file = cache_file
        self.max_history = max_history
        self.event_cache_size = event_cache_size
        self.image_url_cache_size = image_url_cache_size
        self.image_manager = ImageManager()

    def load_cache_state(self, group_histories, recent_events, image_url_lru: OrderedDict[str, Any]):
        if not self.cache_file.exists():
            return
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[myenhance] failed to load cache state: %s", e)
            return

        try:
            raw_group_histories = data.get("group_histories", {})
            if isinstance(raw_group_histories, dict):
                for group_id, items in raw_group_histories.items():
                    dq: Deque[tuple[float, str, str]] = deque(maxlen=self.max_history)
                    for item in items:
                        if not isinstance(item, list):
                            continue
                        if len(item) == 3:
                            dq.append((float(item[0]), str(item[1]), str(item[2])))
                            continue
                        if len(item) == 2:
                            dq.append((float(item[0]), "unknown", str(item[1])))
                    if dq:
                        group_histories[group_id] = dq

            raw_recent_events = data.get("recent_events", {})
            if isinstance(raw_recent_events, dict):
                for scope_id, items in raw_recent_events.items():
                    dq2: Deque[tuple[str, list[str]]] = deque(maxlen=self.event_cache_size)
                    for item in items:
                        if isinstance(item, list) and len(item) == 2:
                            dq2.append((str(item[0]), item[1]))
                    if dq2:
                        recent_events[scope_id] = dq2

            raw_image_url_lru = data.get("image_url_lru", data.get("image_id_lru", []))
            if isinstance(raw_image_url_lru, list):
                image_url_lru.clear()
                for item in raw_image_url_lru:
                    if isinstance(item, list) and len(item) == 2:
                        raw_key = str(item[0] or "").strip()
                        raw_value = item[1]

                        if isinstance(raw_value, dict):
                            image_id = raw_key
                            url = str(raw_value.get("url") or "").strip()
                            keyword = str(raw_value.get("keyword") or "").strip()
                            content = str(raw_value.get("content") or "").strip()

                            if not image_id and url:
                                image_id = self.image_manager.build_legacy_image_id(url)
                            if content and not keyword:
                                keyword = self.image_manager.build_keyword(content)
                            if not image_id:
                                continue

                            image_url_lru[image_id] = {
                                "url": url,
                                "keyword": keyword,
                                "content": content,
                            }
                            continue

                        legacy_url = raw_key
                        if not legacy_url:
                            continue
                        legacy_content = str(raw_value or "").strip()
                        legacy_id = self.image_manager.build_legacy_image_id(legacy_url)
                        if not legacy_id:
                            continue
                        image_url_lru[legacy_id] = {
                            "url": legacy_url,
                            "keyword": self.image_manager.build_keyword(legacy_content),
                            "content": legacy_content,
                        }
                while len(image_url_lru) > self.image_url_cache_size:
                    image_url_lru.popitem(last=False)
        except Exception as e:
            logger.warning("[myenhance] failed to parse cache state: %s", e)

    def save_cache_state(self, group_histories, recent_events, image_url_lru: OrderedDict[str, Any]):
        try:
            payload = {
                "group_histories": {
                    gid: [[ts, msg_id, ln] for ts, msg_id, ln in history]
                    for gid, history in group_histories.items()
                },
                "recent_events": {
                    sid: [[mid, urls] for mid, urls in entries]
                    for sid, entries in recent_events.items()
                },
                "image_url_lru": [[k, v] for k, v in image_url_lru.items()],
            }
            self.cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("[myenhance] failed to save cache state: %s", e)
