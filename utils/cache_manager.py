from __future__ import annotations
import json
from pathlib import Path
from collections import deque
from typing import Deque, OrderedDict
from astrbot.api import logger

class CacheManager:
    """管理持久化缓存状态。"""
    def __init__(self, cache_file: Path, max_history: int, event_cache_size: int, image_url_cache_size: int):
        self.cache_file = cache_file
        self.max_history = max_history
        self.event_cache_size = event_cache_size
        self.image_url_cache_size = image_url_cache_size

    def load_cache_state(self, group_histories, recent_events, image_url_lru: OrderedDict):
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

            raw_image_url_lru = data.get("image_url_lru", [])
            if isinstance(raw_image_url_lru, list):
                image_url_lru.clear()
                for item in raw_image_url_lru:
                    if isinstance(item, list) and len(item) == 2:
                        image_url_lru[str(item[0])] = str(item[1])
                while len(image_url_lru) > self.image_url_cache_size:
                    image_url_lru.popitem(last=False)
        except Exception as e:
            logger.warning("[myenhance] failed to parse cache state: %s", e)

    def save_cache_state(self, group_histories, recent_events, image_url_lru: OrderedDict):
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
