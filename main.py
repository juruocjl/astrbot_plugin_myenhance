from __future__ import annotations

import asyncio
import copy
import json
import math
import random
import re
import uuid
from collections import OrderedDict, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque
from urllib.parse import urljoin

import httpx

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional dependency
    PILImage = None

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.utils.quoted_message_parser import extract_quoted_message_images

from .utils.cache_manager import CacheManager
from .utils.face_map import load_face_desc_map
from .utils.hybrid_retrieval import hybrid_search
from .utils.image_manager import ImageManager
from .utils.jargon_store import JargonRecord, JargonStore
from .utils.memory_store import MemoryRecord, MemoryStore
from .utils.meme_manager import MemeManager
from .utils.message_utils import extract_image_urls, format_time, get_event_timestamp, normalize_message_text
from .flask_ui import start_flask_app


@register("myenhance", "cjlqwq", "记录群消息并注入到 LLM 请求", "1.9.4")
class MyPlugin(Star):
    MEMORY_CONTEXT_MARKER = "[MYENHANCE_MEMORY_CONTEXT]"
    HISTORY_CONTEXT_MARKER = "[MYENHANCE_HISTORY_CONTEXT]"
    QUOTE_HEAD_RE = re.compile(r'<quote\s+id="([^"]+)"\s*/?>', re.IGNORECASE)
    MENTION_RE = re.compile(r'<mention\s+id="([^"]+)"\s*/?>', re.IGNORECASE)
    IMAGE_RE = re.compile(r'<image\s+id="([^"]+)"\s*/?>', re.IGNORECASE)
    MEME_RE = re.compile(r'<meme\s+tag="([^"]+)"\s*/?>', re.IGNORECASE)
    REFUSE_ONLY_RE = re.compile(r'^\s*<refuse\s*/>\s*$')

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self.config = config or {}
        self._load_config()

        self.group_histories: dict[str, Deque[tuple[float, str, str]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self.recent_events: dict[str, Deque[tuple[str, list[str]]]] = defaultdict(
            lambda: deque(maxlen=self.event_cache_size)
        )
        self.image_url_lru: OrderedDict[str, dict[str, str]] = OrderedDict()
        self.group_history_locks: dict[str, asyncio.Lock] = {}
        self.scope_request_locks: dict[str, asyncio.Lock] = {}
        self.scope_active_reply_blocks: dict[str, int] = defaultdict(int)
        self.scope_provider_origins: dict[str, str] = {}
        self.managed_contexts_by_scope: dict[str, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.context_chain_max_records)
        )
        self.context_summary_task: asyncio.Task | None = None
        self.context_summary_stop_event = asyncio.Event()
        self.face_desc_map = load_face_desc_map()
        plugin_data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        plugin_data_path.mkdir(parents=True, exist_ok=True)
        self.cache_state_file = plugin_data_path / ".myenhance_cache_state.json"
        self.jargon_store_file = plugin_data_path / ".myenhance_jargons.json"
        self.memory_store_file = plugin_data_path / ".myenhance_memories.json"
        self.meme_store_file = plugin_data_path / ".myenhance_memes.json"
        self.managed_contexts_file = plugin_data_path / ".myenhance_contexts.json"
        self.temp_summary_prompt_file = plugin_data_path / ".myenhance_summary_prompt.tmp.txt"
        self.image_manager = ImageManager(plugin_data_path / "images")
        self.cache_manager = CacheManager(
            self.cache_state_file,
            self.max_history,
            self.event_cache_size,
            self.image_url_cache_size,
            image_manager=self.image_manager,
        )
        self.cache_manager.load_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )
        self.jargon_store = JargonStore(self.jargon_store_file, self.jargon_max_records)
        self.memory_store = MemoryStore(self.memory_store_file, self.memory_max_records)
        self.meme_manager = MemeManager(self.meme_store_file)
        self._load_managed_contexts()
        self.summary_prompt_template = self._load_summary_prompt_template()
        self.jargon_prompt_rules = self._load_jargon_prompt_rules()
        self.reply_system_prompt_cn = self._build_reply_system_prompt()
        self._flask_server = None
        self._flask_thread = None
        self.stop_flask = None
        if self.web_port > 0:
            self.stop_flask = start_flask_app(self, self.web_port)
        self._start_context_summary_task_if_possible()

    async def terminate(self) -> None:
        self._save_managed_contexts()
        await self._stop_context_summary_task()
        if self.stop_flask:
            try:
                self.stop_flask()
            except Exception as exc:
                logger.warning("[myenhance] failed to stop Flask UI: %s", exc)
            finally:
                self.stop_flask = None
                self._flask_server = None
                self._flask_thread = None

    def _build_reply_system_prompt(self) -> str:
        fallback = "你正在群聊中进行消息回复。你的整个输出必须是发给群聊的一条回复消息，不要输出额外说明。"
        return self._load_text_asset("reply_system_prompt.txt", fallback)

    def _load_config(self) -> None:
        raw_max_history = self.config.get("max_history", 300)
        try:
            self.max_history = max(1, int(raw_max_history))
        except (TypeError, ValueError):
            self.max_history = 300

        self.active_reply_enable = bool(self.config.get("active_reply_enable", False))
        raw_probability = self.config.get("active_reply_probability", 0.1)
        try:
            self.active_reply_probability = min(max(float(raw_probability), 0.0), 1.0)
        except (TypeError, ValueError):
            self.active_reply_probability = 0.1

        raw_whitelist = self.config.get("active_reply_whitelist", [])
        if isinstance(raw_whitelist, list):
            self.active_reply_whitelist = {str(item) for item in raw_whitelist if str(item).strip()}
        else:
            self.active_reply_whitelist = set()

        raw_event_cache_size = self.config.get("cached_size", 120)
        try:
            self.event_cache_size = max(1, int(raw_event_cache_size))
        except (TypeError, ValueError):
            self.event_cache_size = 120

        self.describe_image_provider_id = str(
            self.config.get("describe_image_provider_id", "") or ""
        ).strip()
        self.describe_image_ask = str(
            self.config.get("describe_image_ask", "请客观描述这张图片中的主要内容，简洁一些。")
            or ""
        ).strip() or "请客观描述这张图片中的主要内容，简洁一些。"
        self.embedding_provider_id = str(
            self.config.get("embedding_provider_id", "") or ""
        ).strip()

        raw_image_url_cache_size = self.config.get("image_url_cache_size", 120)
        try:
            self.image_url_cache_size = max(1, int(raw_image_url_cache_size))
        except (TypeError, ValueError):
            self.image_url_cache_size = 120

        raw_jargon_current_recall_count = self.config.get("jargon_current_recall_count", 3)
        try:
            self.jargon_current_recall_count = max(0, int(raw_jargon_current_recall_count))
        except (TypeError, ValueError):
            self.jargon_current_recall_count = 3

        raw_jargon_total_recall_count = self.config.get(
            "jargon_total_recall_count",
            self.config.get("jargon_recall_count", 5),
        )
        try:
            self.jargon_total_recall_count = max(0, int(raw_jargon_total_recall_count))
        except (TypeError, ValueError):
            self.jargon_total_recall_count = 5
        # Backward compatibility for old code paths.
        self.jargon_recall_count = self.jargon_total_recall_count

        raw_history_inject_count = self.config.get("history_inject_count", 12)
        try:
            self.history_inject_count = max(0, int(raw_history_inject_count))
        except (TypeError, ValueError):
            self.history_inject_count = 12

        raw_context_user_limit = self.config.get("context_user_limit", 12)
        try:
            self.context_user_limit = max(1, int(raw_context_user_limit))
        except (TypeError, ValueError):
            self.context_user_limit = 12

        raw_context_chain_max_records = self.config.get("context_chain_max_records", 120)
        try:
            self.context_chain_max_records = max(10, int(raw_context_chain_max_records))
        except (TypeError, ValueError):
            self.context_chain_max_records = 120

        raw_context_user_keep_after = self.config.get("context_user_keep_after", 4)
        try:
            keep_after = int(raw_context_user_keep_after)
        except (TypeError, ValueError):
            keep_after = 4
        keep_after = max(1, keep_after)
        keep_after = min(keep_after, self.context_user_limit)
        self.context_user_keep_after = keep_after

        raw_context_summary_interval = self.config.get("context_summary_interval", 8)
        try:
            self.context_summary_interval = max(1, int(raw_context_summary_interval))
        except (TypeError, ValueError):
            self.context_summary_interval = 8

        raw_memory_recall_count = self.config.get("memory_recall_count", 3)
        try:
            self.memory_recall_count = max(0, int(raw_memory_recall_count))
        except (TypeError, ValueError):
            self.memory_recall_count = 3

        raw_memory_max_records = self.config.get("memory_max_records", 500)
        try:
            self.memory_max_records = max(1, int(raw_memory_max_records))
        except (TypeError, ValueError):
            self.memory_max_records = 500

        raw_bm25_weight = self.config.get("bm25_weight", 0.55)
        try:
            self.bm25_weight = min(max(float(raw_bm25_weight), 0.0), 1.0)
        except (TypeError, ValueError):
            self.bm25_weight = 0.55
        self.embedding_weight = 1.0 - self.bm25_weight

        raw_jargon_max_records = self.config.get("jargon_max_records", 500)
        try:
            self.jargon_max_records = max(1, int(raw_jargon_max_records))
        except (TypeError, ValueError):
            self.jargon_max_records = 500

        raw_rrf_k = self.config.get("rrf_k", 60)
        try:
            self.rrf_k = max(1, int(raw_rrf_k))
        except (TypeError, ValueError):
            self.rrf_k = 60

        raw_web_port = self.config.get("web_port", 6180)
        try:
            self.web_port = int(raw_web_port)
        except (TypeError, ValueError):
            self.web_port = 6180

        self.mod_api_url = str(self.config.get("mod_api_url") or "").strip()
        self.mod_auth_token = str(self.config.get("mod_auth_token") or "").strip()
        raw_mod_max = self.config.get("mod_max_mute_duration", 0)
        try:
            self.mod_max_mute_duration = max(0, int(raw_mod_max))
        except (TypeError, ValueError):
            self.mod_max_mute_duration = 0

        config_snapshot = {
            "max_history": self.max_history,
            "active_reply_enable": self.active_reply_enable,
            "active_reply_probability": self.active_reply_probability,
            "active_reply_whitelist": sorted(self.active_reply_whitelist),
            "event_cache_size": self.event_cache_size,
            "describe_image_provider_id": self.describe_image_provider_id,
            "describe_image_ask": self.describe_image_ask,
            "embedding_provider_id": self.embedding_provider_id,
            "image_url_cache_size": self.image_url_cache_size,
            "jargon_current_recall_count": self.jargon_current_recall_count,
            "jargon_total_recall_count": self.jargon_total_recall_count,
            "history_inject_count": self.history_inject_count,
            "context_user_limit": self.context_user_limit,
            "context_chain_max_records": self.context_chain_max_records,
            "context_user_keep_after": self.context_user_keep_after,
            "context_summary_interval": self.context_summary_interval,
            "memory_recall_count": self.memory_recall_count,
            "memory_max_records": self.memory_max_records,
            "bm25_weight": self.bm25_weight,
            "embedding_weight": self.embedding_weight,
            "jargon_max_records": self.jargon_max_records,
            "rrf_k": self.rrf_k,
            "web_port": self.web_port,
            "mod_api_url": self.mod_api_url,
            "mod_auth_token": "***" if self.mod_auth_token else "",
            "mod_max_mute_duration": self.mod_max_mute_duration,
        }
        logger.info("[myenhance] loaded config: %s", json.dumps(config_snapshot, ensure_ascii=False))

    def _get_group_lock(self, group_id: str) -> asyncio.Lock:
        lock = self.group_history_locks.get(group_id)
        if lock is None:
            lock = asyncio.Lock()
            self.group_history_locks[group_id] = lock
        return lock

    def _get_scope_request_lock(self, scope_id: str) -> asyncio.Lock:
        lock = self.scope_request_locks.get(scope_id)
        if lock is None:
            lock = asyncio.Lock()
            self.scope_request_locks[scope_id] = lock
        return lock

    def _is_scope_request_locked(self, scope_id: str) -> bool:
        lock = self.scope_request_locks.get(scope_id)
        return bool(lock and lock.locked())

    async def _mark_active_reply_block_start(self, scope_id: str) -> None:
        normalized_scope = str(scope_id or "").strip()
        if not normalized_scope:
            return
        async with self._get_group_lock(normalized_scope):
            self.scope_active_reply_blocks[normalized_scope] += 1

    async def _mark_active_reply_block_end(self, scope_id: str) -> None:
        normalized_scope = str(scope_id or "").strip()
        if not normalized_scope:
            return
        async with self._get_group_lock(normalized_scope):
            current = int(self.scope_active_reply_blocks.get(normalized_scope, 0))
            if current <= 1:
                self.scope_active_reply_blocks.pop(normalized_scope, None)
            else:
                self.scope_active_reply_blocks[normalized_scope] = current - 1

    def _is_active_reply_blocked(self, scope_id: str) -> bool:
        normalized_scope = str(scope_id or "").strip()
        if not normalized_scope:
            return False
        return int(self.scope_active_reply_blocks.get(normalized_scope, 0)) > 0

    def _start_context_summary_task_if_possible(self) -> None:
        if self.context_summary_task and not self.context_summary_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self.context_summary_stop_event.clear()
        self.context_summary_task = asyncio.create_task(self._context_summary_worker())

    async def _stop_context_summary_task(self) -> None:
        task = self.context_summary_task
        self.context_summary_stop_event.set()
        if not task:
            return
        if task.done():
            self.context_summary_task = None
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self.context_summary_task = None

    async def _context_summary_worker(self) -> None:
        logger.info(
            "[myenhance] context summary worker started (interval=%ss)",
            self.context_summary_interval,
        )
        try:
            while not self.context_summary_stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self.context_summary_stop_event.wait(),
                        timeout=float(self.context_summary_interval),
                    )
                except asyncio.TimeoutError:
                    pass

                if self.context_summary_stop_event.is_set():
                    break
                await self._run_context_summary_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[myenhance] context summary worker stopped unexpectedly: %s", exc)

    async def _run_context_summary_cycle(self) -> None:
        if self.context_user_limit <= 0:
            return

        scope_ids = list(self.managed_contexts_by_scope.keys())
        if not scope_ids:
            return

        for scope_id in scope_ids:
            if not scope_id:
                continue

            if self._is_active_reply_blocked(scope_id):
                continue

            await self._mark_active_reply_block_start(scope_id)
            try:
                managed_contexts = await self._get_managed_contexts(scope_id)
                trimmed_count = await self._apply_context_memory_management(
                    scope_id,
                    managed_contexts,
                )
                if trimmed_count > 0:
                    async with self._get_group_lock(scope_id):
                        chain = self.managed_contexts_by_scope.get(scope_id)
                        if not chain:
                            continue
                        remove_count = min(trimmed_count, len(chain))
                        for _ in range(remove_count):
                            chain.popleft()
                        if remove_count > 0:
                            self._save_managed_contexts()
            finally:
                await self._mark_active_reply_block_end(scope_id)

    def _context_to_dict(self, ctx: Any) -> dict[str, Any] | None:
        if isinstance(ctx, dict):
            role = str(ctx.get("role") or "").strip().lower()
            if not role:
                return None
            item = dict(ctx)
            item["role"] = role
            return item

        role = str(getattr(ctx, "role", "") or "").strip().lower()
        if not role:
            return None
        item: dict[str, Any] = {"role": role}
        for field in ("content", "name", "tool_call_id", "tool_calls"):
            if hasattr(ctx, field):
                item[field] = getattr(ctx, field)
        return item

    async def _append_managed_context(self, scope_id: str, context: Any) -> None:
        normalized_scope = str(scope_id or "").strip()
        item = self._context_to_dict(context)
        if not normalized_scope or not item:
            return

        async with self._get_group_lock(normalized_scope):
            chain = self.managed_contexts_by_scope[normalized_scope]
            if chain and chain[-1] == item:
                    return
            chain.append(item)
            self._save_managed_contexts()

    async def _replace_managed_contexts(self, scope_id: str, contexts: list[Any]) -> None:
        normalized_scope = str(scope_id or "").strip()
        if not normalized_scope:
            return
        normalized = [
            item
            for item in (self._context_to_dict(ctx) for ctx in contexts)
            if item is not None
        ]

        async with self._get_group_lock(normalized_scope):
            chain = deque(maxlen=self.context_chain_max_records)
            chain.extend(normalized)
            self.managed_contexts_by_scope[normalized_scope] = chain
            self._save_managed_contexts()

    async def _get_managed_contexts(self, scope_id: str) -> list[dict[str, Any]]:
        normalized_scope = str(scope_id or "").strip()
        if not normalized_scope:
            return []
        async with self._get_group_lock(normalized_scope):
            chain = self.managed_contexts_by_scope.get(normalized_scope)
            if not chain:
                return []
            return [dict(item) for item in chain]

    def _extract_new_context_after_last_user(self, contexts: list[Any] | None) -> list[dict[str, Any]]:
        if not contexts:
            return []

        normalized = [
            item
            for item in (self._context_to_dict(ctx) for ctx in contexts)
            if item is not None
        ]

        if not normalized:
            return []

        last_user_index = -1
        for idx, item in enumerate(normalized):
            if item.get("role") == "user":
                last_user_index = idx

        if last_user_index < 0:
            return normalized
        return normalized[last_user_index + 1 :]

    def _compose_mute_payload(
        self,
        group_id: int,
        user_id: int,
        duration: int,
        reason: str,
    ) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "groupId": group_id,
            "userId": user_id,
            "duration": max(0, duration),
            "reason": reason or "由 MyEnhance 管理工具触发",
        }
        if self.mod_auth_token:
            payload["authToken"] = self.mod_auth_token
        return payload

    async def _send_mute_request(self, payload: dict) -> tuple[bool, str]:
        base_url = str(self.mod_api_url or "").strip()
        if not base_url:
            return False, "未配置 mod_api_url"
        endpoint = urljoin(base_url.rstrip("/") + "/", "/api/mod/mute")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.mod_auth_token:
            headers["Authorization"] = f"Bearer {self.mod_auth_token}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
            status = resp.status_code
            try:
                response = resp.json()
            except ValueError:
                response = {}
            logger.info(f"[myenhance] mute request response: {response}")
            success = bool(response.get("success")) or status == 200
            message = response.get("message") or resp.text or "禁言执行完成"
            return success, message
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            return False, f"禁言调用失败：{exc.response.status_code} {exc}\n{body}"
        except httpx.HTTPError as exc:
            return False, f"禁言调用失败：{exc}"
        except Exception as exc:
            return False, f"禁言调用异常：{exc}"


    @filter.command("ban")
    async def ban_command(
        self,
        event: AstrMessageEvent,
        target: str = "",
        duration: int = 60,
        reason: str = "",
    ):
        """管理员专用 ban 指令，参数自动解析。

        Args:
            target(string): 目标用户 ID 或 mention。
            duration(number): 禁言时长（秒），默认 60。
            reason(string): 禁言理由。
        """
        if not event.is_admin():
            yield event.make_result().message("ban命令仅限管理员使用。")
            return

        normalized_target = str(target or "").strip()
        if not normalized_target:
            yield event.make_result().message("ban命令需要指定被禁言的用户。")
            return
        if not normalized_target.isdigit():
            yield event.make_result().message("ban命令的用户ID必须为数字。")
            return

        try:
            requested_duration = max(0, int(duration))
        except (TypeError, ValueError):
            requested_duration = 60

        reason_text = str(reason or "").strip()
        payload_reason = reason_text or "管理员 ban 命令"

        response = await self.mute_member(
            event,
            user_id=normalized_target,
            duration=requested_duration,
            reason=payload_reason,
        )
        yield event.make_result().message(response)

    def _get_event_scope_id(self, event: AstrMessageEvent) -> str:
        return str(event.get_group_id() or event.unified_msg_origin or "").strip()

    async def _record_line(self, group_id: str, event_ts: float, msg_id: str, line: str) -> None:
        normalized_group_id = str(group_id or "").strip()
        if not normalized_group_id:
            return

        normalized_msg_id = str(msg_id or "unknown").strip() or "unknown"
        async with self._get_group_lock(normalized_group_id):
            self.group_histories[normalized_group_id].append((event_ts, normalized_msg_id, line))
        self.cache_manager.save_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )

    async def _cache_recent_event(self, event: AstrMessageEvent) -> None:
        scope_id = self._get_event_scope_id(event)
        msg_id = str(getattr(event.message_obj, "message_id", "") or "").strip()
        if not scope_id or not msg_id:
            return

        image_ids = await self._touch_image_ids(extract_image_urls(event))
        async with self._get_group_lock(scope_id):
            cached = self.recent_events[scope_id]
            deduped = [
                (cached_msg_id, urls)
                for cached_msg_id, urls in cached
                if cached_msg_id != msg_id
            ]
            cached.clear()
            cached.extend(deduped)
            cached.append((msg_id, image_ids))
        self.cache_manager.save_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )

    async def _touch_image_ids(self, image_urls: list[str]) -> list[str]:
        image_ids: list[str] = []
        for image_url in image_urls:
            image_id = await self.image_manager.ensure_image(
                image_url,
                self.image_url_lru,
                self.image_url_cache_size,
            )
            if image_id:
                image_ids.append(image_id)
        if image_ids:
            self.cache_manager.save_cache_state(
                self.group_histories,
                self.recent_events,
                self.image_url_lru,
            )
        return image_ids

    def _get_image_inject_tag(self, image_id: str) -> str:
        entry = self.image_manager.get_entry(image_id, self.image_url_lru)
        keyword = ""
        if entry:
            keyword = str(entry.get("keyword") or "").strip()

        meme_tags = self.meme_manager.get_tags_by_image_id(image_id)
        if meme_tags:
            tags_text = "|".join(meme_tags)
            if keyword:
                return self.image_manager.build_inject_tag(
                    image_id,
                    f"tag={tags_text}; keyword={keyword}",
                )
            return self.image_manager.build_inject_tag(image_id, f"tag={tags_text}")

        return self.image_manager.build_inject_tag(image_id, keyword)

    async def _inject_image_ids_to_text(self, event: AstrMessageEvent, text: str) -> str:
        normalized = str(text or "")
        if "[image]" not in normalized:
            return normalized

        image_ids = await self._touch_image_ids(extract_image_urls(event))
        if not image_ids:
            return normalized

        replaced = normalized
        for image_id in image_ids:
            if "[image]" not in replaced:
                break
            replaced = replaced.replace("[image]", self._get_image_inject_tag(image_id), 1)
        return replaced

    def _get_cached_image_payload(self, image_id: str) -> tuple[str, str] | None:
        entry = self.image_manager.get_entry(image_id, self.image_url_lru)
        if not entry:
            return None

        content = re.sub(r"\s+", " ", str(entry.get("content") or "").strip())
        if not content:
            return None

        keyword = re.sub(r"\s+", " ", str(entry.get("keyword") or "").strip())
        if not keyword:
            keyword = self.image_manager.build_keyword(content)
        return keyword, content

    def _set_cached_image_payload(self, image_id: str, keyword: str, content: str) -> None:
        entry = self.image_manager.set_description(
            image_id,
            self.image_url_lru,
            keyword,
            content,
            self.image_url_cache_size,
        )
        if not entry:
            return
        self.cache_manager.save_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )

    async def _get_cached_image_ids_by_msg_id(
        self,
        event: AstrMessageEvent,
        msg_id: str,
    ) -> list[str] | None:
        scope_id = self._get_event_scope_id(event)
        target_id = str(msg_id or "").strip()
        if not scope_id or not target_id:
            return None

        async with self._get_group_lock(scope_id):
            for cached_msg_id, cached_urls in reversed(self.recent_events.get(scope_id, [])):
                if cached_msg_id == target_id:
                    return list(cached_urls)
        return None

    async def _get_recent_history_lines(self, event: AstrMessageEvent) -> tuple[list[str], str]:
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return [], ""

        current_msg_id = str(getattr(event.message_obj, "message_id", "") or "").strip()
        current_event_ts = get_event_timestamp(event)
        
        history_lines = []
        search_lines = []
        
        async with self._get_group_lock(group_id):
            history = self.group_histories.get(group_id)
            if not history:
                return [], ""
            
            remaining_items = []
            for item_ts, item_msg_id, line in history:
                if item_ts <= current_event_ts:
                    if item_msg_id != current_msg_id:
                        # 提取实际文本部分用于检索关键词
                        parts = line.split("\n", 1)
                        text = parts[1] if len(parts) > 1 else line
                        search_lines.append(text)
                    history_lines.append(line)
                else:
                    remaining_items.append((item_ts, item_msg_id, line))
            history.clear()
            history.extend(remaining_items)
        self.cache_manager.save_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )
        inject_lines = history_lines[-self.history_inject_count :] if self.history_inject_count > 0 else []
        search_query = "\n".join(search_lines)
        return inject_lines, search_query

    def _load_summary_prompt_template(self) -> str:
        return self._load_text_asset("prompt.txt", "")

    async def _format_member_message_async(self, event: AstrMessageEvent) -> str:
        poke_text = self._format_poke_message(event)
        if poke_text:
            return poke_text

        nickname = event.get_sender_name() or "unknown"
        sender_id = event.get_sender_id() or "unknown"
        role = "admin" if event.is_admin() else "member"
        timestamp = getattr(event.message_obj, "timestamp", None)
        msg_id = getattr(event.message_obj, "message_id", None) or "unknown"
        text = normalize_message_text(event, self.face_desc_map)
        text = await self._inject_image_ids_to_text(event, text)
        return f"[{nickname}/{sender_id}/{format_time(timestamp)}] ({role})#msg{msg_id}\n{text}"

    def _load_jargon_prompt_rules(self) -> str:
        fallback = "请根据已注入黑话谨慎调用工具并回复消息。"
        return self._load_text_asset("jargon_prompt_rules.txt", fallback)

    def _load_text_asset(self, file_name: str, fallback: str = "") -> str:
        asset_path = Path(__file__).parent / "assets" / file_name
        if not asset_path.exists():
            return fallback
        try:
            content = asset_path.read_text(encoding="utf-8").strip()
            return content or fallback
        except Exception as exc:
            logger.warning("[myenhance] failed to load asset %s: %s", file_name, exc)
            return fallback

    def _load_managed_contexts(self) -> None:
        if not self.managed_contexts_file.exists():
            return
        try:
            data = json.loads(self.managed_contexts_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[myenhance] failed to load managed contexts: %s", exc)
            return

        scopes = data.get("scopes", {})
        if not isinstance(scopes, dict):
            return

        loaded: dict[str, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.context_chain_max_records)
        )
        for scope_id, items in scopes.items():
            if not isinstance(scope_id, str) or not isinstance(items, list):
                continue
            chain = deque(maxlen=self.context_chain_max_records)
            for item in items:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip().lower()
                if not role:
                    continue
                restored = dict(item)
                restored["role"] = role
                chain.append(restored)
            if chain:
                loaded[scope_id] = chain
        self.managed_contexts_by_scope = loaded

    def _save_managed_contexts(self) -> None:
        try:
            payload = {
                "scopes": {
                    scope_id: list(chain)
                    for scope_id, chain in self.managed_contexts_by_scope.items()
                    if chain
                }
            }
            self.managed_contexts_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[myenhance] failed to save managed contexts: %s", exc)

    def _get_context_role(self, ctx: Any) -> str:
        if isinstance(ctx, dict):
            return (ctx.get("role") or "").strip().lower()
        return (getattr(ctx, "role", "") or "").strip().lower()

    def _extract_context_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            parts = []
            for key in ("content", "text"):
                fragment = content.get(key)
                fragment_text = self._extract_context_text(fragment)
                if fragment_text:
                    parts.append(fragment_text)
            return "\n".join(parts).strip()
        if isinstance(content, list):
            parts = [self._extract_context_text(item) for item in content]
            return "\n".join(part for part in parts if part).strip()
        if hasattr(content, "text"):
            return self._extract_context_text(getattr(content, "text"))
        if hasattr(content, "content") and not isinstance(content, str):
            return self._extract_context_text(getattr(content, "content"))
        return ""

    def _contexts_to_summary_text(self, contexts: list[Any]) -> str:
        parts = []
        allowed_roles = {"user", "assistant"}
        for ctx in contexts:
            role = self._get_context_role(ctx) or "user"
            if role not in allowed_roles:
                continue
            if isinstance(ctx, dict):
                content = ctx.get("content")
            else:
                content = getattr(ctx, "content", None)
            text = self._extract_context_text(content)
            if not text:
                continue
            parts.append(f"{role.capitalize()}: {text}")
        return "\n".join(parts)

    def _extract_summary_keyword_content(self, raw_text: str) -> tuple[str, str] | None:
        payload = str(raw_text or "").strip()
        if not payload:
            return None

        candidates: list[str] = []

        # 优先提取 markdown 代码块中的 JSON。
        fenced_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        for match in fenced_pattern.finditer(payload):
            block = (match.group(1) or "").strip()
            if block:
                candidates.append(block)

        # 兜底：尝试直接解析全文。
        candidates.append(payload)

        # 再兜底：提取文本里的 JSON 对象片段。
        json_object_pattern = re.compile(r"\{[\s\S]*?\}")
        for match in json_object_pattern.finditer(payload):
            frag = (match.group(0) or "").strip()
            if frag:
                candidates.append(frag)

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                data = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue
            keyword = re.sub(r"\s+", " ", str(data.get("keyword") or "").strip())
            content = re.sub(r"\s+", " ", str(data.get("content") or "").strip())
            if content:
                return keyword, content
        return None

    async def _summarize_context_blocks(
        self,
        scope_id: str,
        contexts: list[Any],
    ) -> tuple[str, str] | None:
        if not contexts or not self.summary_prompt_template:
            return None

        conversation = self._contexts_to_summary_text(contexts)
        if not conversation:
            return None
        logger.info(
            "[myenhance] start summarizing contexts: blocks=%d chars=%d",
            len(contexts),
            len(conversation),
        )

        provider_origin = self.scope_provider_origins.get(scope_id) or scope_id
        provider = self.context.get_using_provider(provider_origin)
        if not provider:
            return None

        prompt = f"{self.summary_prompt_template}\n\n{conversation}"
        try:
            self.temp_summary_prompt_file.write_text(prompt, encoding="utf-8")
        except Exception as exc:
            logger.warning("[myenhance] failed to write temp summary prompt file: %s", exc)
        try:
            response = await provider.text_chat(
                prompt=prompt,
                session_id=uuid.uuid4().hex,
                persist=False,
            )
        except Exception as exc:
            logger.warning("[myenhance] failed to summarize contexts: %s", exc)
            return None

        text = (getattr(response, "completion_text", "") or "").strip()
        parsed = self._extract_summary_keyword_content(text)
        if parsed is None:
            shortened = self._shorten_memory_summary(text)
            if not shortened:
                return None
            keyword = self._build_memory_keyword(shortened)
            logger.info(
                "[myenhance] summarize fallback parsed: keyword_len=%d content_len=%d",
                len(keyword),
                len(shortened),
            )
            return keyword, shortened

        keyword, content = parsed
        shortened = self._shorten_memory_summary(content)
        if not shortened:
            return None
        final_keyword = keyword or self._build_memory_keyword(shortened)
        logger.info(
            "[myenhance] summarize parsed json: keyword_len=%d content_len=%d",
            len(final_keyword),
            len(shortened),
        )
        return final_keyword, shortened

    def _shorten_memory_summary(self, summary_text: str, max_chars: int = 320) -> str:
        normalized = re.sub(r"\s+", " ", str(summary_text or "").strip())
        if not normalized:
            return ""
        if len(normalized) <= max_chars:
            return normalized

        for sep in ("。", "！", "？", ";", "；", "，", ","):
            idx = normalized.find(sep)
            if 0 < idx <= max_chars:
                return normalized[: idx + 1].strip()

        shortened = normalized[:max_chars].rstrip("，,;；:： ")
        return f"{shortened}。"

    def _build_memory_keyword(self, summary_text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(summary_text or "").strip())
        if not normalized:
            return "summary"
        return normalized[:80]

    def _format_memory_time(self, raw_time: str) -> str:
        text = str(raw_time or "").strip()
        if not text:
            return "unknown"
        try:
            return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return text

    async def _store_system_memory_entry(
        self,
        scope_id: str,
        summary_keyword: str,
        summary_text: str,
    ) -> None:
        if not scope_id or not summary_text:
            return
        embedding = None
        provider = self._get_embedding_provider()
        if provider:
            try:
                embedding = await provider.get_embedding(summary_text)
            except Exception as exc:
                logger.warning("[myenhance] failed to get embedding for memory summary: %s", exc)

        keyword = re.sub(r"\s+", " ", str(summary_keyword or "").strip())
        if not keyword:
            keyword = self._build_memory_keyword(summary_text)
        try:
            self.memory_store.add_memory(
                scope_id,
                summary_text.strip(),
                keyword=keyword,
                embedding=embedding,
            )
        except ValueError as exc:
            logger.warning("[myenhance] failed to persist memory summary: %s", exc)

    async def _apply_context_memory_management(
        self,
        scope_id: str,
        contexts: list[Any] | None,
    ) -> int:
        if not contexts or self.context_user_limit <= 0:
            return 0

        context_list = list(contexts)
        user_positions: list[int] = []
        for idx, ctx in enumerate(context_list):
            if self._get_context_role(ctx) == "user":
                user_positions.append(idx)
        logger.debug(f"[myenhance] context user positions: {user_positions}")
        if len(user_positions) < self.context_user_limit:
            return 0

        # 保留最近 N 个 user 块及其后的上下文，而不是从头数第 N 个。
        keep_user_count = min(len(user_positions), max(1, self.context_user_keep_after))
        keep_from_user_idx = len(user_positions) - keep_user_count
        keep_start_idx = user_positions[keep_from_user_idx]
        if keep_start_idx <= 0:
            return 0

        summary_contexts = context_list[:keep_start_idx]
        kept_contexts = context_list[keep_start_idx:]
        logger.info(
            "[myenhance] context trimming triggered: total_user=%d keep_user=%d summarize_blocks=%d keep_blocks=%d",
            len(user_positions),
            keep_user_count,
            len(summary_contexts),
            len(kept_contexts),
        )
        summary_result = await self._summarize_context_blocks(scope_id, summary_contexts)
        if summary_result and scope_id:
            summary_keyword, summary_text = summary_result
            await self._store_system_memory_entry(scope_id, summary_keyword, summary_text)
            logger.info(
                "[myenhance] summarized %d contexts into system memory for scope %s",
                len(summary_contexts),
                scope_id,
            )
        return keep_start_idx

    def _build_recent_memory_context_block(self, scope_id: str, limit: int = 5) -> str | None:
        records = self.memory_store.list_memories(scope_id)
        if not records:
            return None

        selected = records[-max(1, limit) :]
        lines = [
            f"- [{record.id}] ({self._format_memory_time(record.created_at)}) {record.content}"
            for record in selected
        ]
        return (
            f"{self.MEMORY_CONTEXT_MARKER}\n"
            "以下是最近记忆（按时间从旧到新，供上下文参考）：\n"
            f"{'\n'.join(lines)}"
        )

    def _inject_recent_memory_context_block(self, contexts: list[Any] | None, scope_id: str) -> list[Any]:
        context_list = copy.deepcopy(list(contexts or []))
        marker = self.MEMORY_CONTEXT_MARKER
        context_list = [
            ctx
            for ctx in context_list
            if not (
                self._get_context_role(ctx) == "user"
                and marker in self._extract_context_text(
                    ctx.get("content") if isinstance(ctx, dict) else getattr(ctx, "content", None)
                )
            )
        ]
        block = self._build_recent_memory_context_block(scope_id, limit=5)
        if not block:
            return context_list

        context_list.insert(0, {"role": "user", "content": block})
        return context_list

    def _get_embedding_provider(self):
        if not self.embedding_provider_id:
            return None

        provider = self.context.get_provider_by_id(self.embedding_provider_id)
        if not provider:
            logger.warning(
                "[myenhance] embedding provider not found: %s",
                self.embedding_provider_id,
            )
            return None
        if not callable(getattr(provider, "get_embedding", None)):
            logger.warning(
                "[myenhance] provider %s does not implement get_embedding",
                self.embedding_provider_id,
            )
            return None
        return provider

    async def _build_embedding_scores(self, query: str, records: list[Any]) -> list[float] | None:
        provider = self._get_embedding_provider()
        if not provider or not records:
            return None

        # 尝试使用已有的 embedding，如果没有则实时获取
        documents_to_embed = []
        doc_indices_to_embed = []
        
        final_embedding_scores = [0.0] * len(records)
        document_vectors = [None] * len(records)

        for i, record in enumerate(records):
            if record.embedding is not None:
                document_vectors[i] = record.embedding
            else:
                documents_to_embed.append(record.content)
                doc_indices_to_embed.append(i)

        try:
            query_vector = await provider.get_embedding(query)
            
            # 如果有需要获取的 embedding
            if documents_to_embed:
                logger.info("[myenhance] fetching embeddings for %d records", len(documents_to_embed))
                if callable(getattr(provider, "get_embeddings", None)):
                    new_vectors = await provider.get_embeddings(documents_to_embed)
                else:
                    new_vectors = [await provider.get_embedding(doc) for doc in documents_to_embed]
                
                # 回填并持久化
                for i, vector in zip(doc_indices_to_embed, new_vectors):
                    document_vectors[i] = vector
                    records[i].embedding = vector
                self.jargon_store.save()

            return [self._cosine_similarity(query_vector, vector) for vector in document_vectors]

        except Exception as exc:
            logger.warning("[myenhance] failed to get embeddings: %s", exc)
            return None

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
        return dot / (left_norm * right_norm)

    async def _get_related_jargon(
        self,
        event: AstrMessageEvent,
        query: str,
        limit: int | None = None,
    ) -> list[JargonRecord]:
        target_limit = self.jargon_total_recall_count if limit is None else max(0, int(limit))
        if target_limit <= 0:
            return []

        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return []

        records = self.jargon_store.list_jargons(scope_id)
        if not records:
            return []

        embedding_scores = await self._build_embedding_scores(
            query,
            records,
        )
        return [
            item.record
            for item in hybrid_search(
                query,
                records,
                target_limit,
                bm25_weight=self.bm25_weight,
                embedding_weight=self.embedding_weight,
                embedding_scores=embedding_scores,
                rrf_k=self.rrf_k,
            )
        ]

    async def _get_related_memories(self, event: AstrMessageEvent, query: str) -> list[MemoryRecord]:
        if self.memory_recall_count <= 0:
            return []

        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return []

        records = self.memory_store.list_memories(scope_id)
        if not records:
            return []

        embedding_scores = await self._build_embedding_scores(
            query,
            records,
        )
        return [
            item.record
            for item in hybrid_search(
                query,
                records,
                self.memory_recall_count,
                bm25_weight=self.bm25_weight,
                embedding_weight=self.embedding_weight,
                embedding_scores=embedding_scores,
                rrf_k=self.rrf_k,
            )
        ]

    def _format_member_message(self, event: AstrMessageEvent) -> str:
        poke_text = self._format_poke_message(event)
        if poke_text:
            return poke_text

        nickname = event.get_sender_name() or "unknown"
        sender_id = event.get_sender_id() or "unknown"
        role = "admin" if event.is_admin() else "member"
        timestamp = getattr(event.message_obj, "timestamp", None)
        msg_id = getattr(event.message_obj, "message_id", None) or "unknown"
        text = normalize_message_text(event, self.face_desc_map)
        return f"[{nickname}/{sender_id}/{format_time(timestamp)}] ({role})#msg{msg_id}\n{text}"

    def _format_poke_message(self, event: AstrMessageEvent) -> str | None:
        raw_message = getattr(event.message_obj, "raw_message", None)
        if not isinstance(raw_message, dict):
            return None
        if str(raw_message.get("notice_type") or "") != "notify":
            return None
        if str(raw_message.get("sub_type") or "") != "poke":
            return None

        user_id = str(raw_message.get("user_id") or event.get_sender_id() or "unknown")
        target_id = str(raw_message.get("target_id") or "unknown")
        timestamp = raw_message.get("time") or getattr(event.message_obj, "timestamp", None)
        msg_id = getattr(event.message_obj, "message_id", None) or "unknown"
        return f"[system/system/{format_time(timestamp)}] #msg{msg_id}\n{user_id} 戳了戳 {target_id}"

    def _get_bot_id(self, event: AstrMessageEvent) -> str:
        return str(event.get_self_id() or "unknown").strip() or "unknown"

    def _parse_control_tags_to_chain(self, text: str) -> MessageChain | None:
        if not text:
            return None

        match = self.QUOTE_HEAD_RE.search(text)
        quote_id = match.group(1).strip() if match else ""
        body = text[:match.start()] + text[match.end() :] if match else text
        touched = bool(match)
        chain: list = []

        if quote_id:
            chain.append(Reply(id=quote_id))

        tag_re = re.compile(
            r'<mention\s+id="([^"]+)"\s*/?>|<image\s+id="([^"]+)"\s*/?>|<meme\s+tag="([^"]+)"\s*/?>',
            re.IGNORECASE,
        )
        cursor = 0
        for tag_match in tag_re.finditer(body):
            touched = True
            if tag_match.start() > cursor:
                plain = body[cursor : tag_match.start()]
                if plain:
                    chain.append(Plain(plain))
            mention_id = str(tag_match.group(1) or "").strip()
            image_id = str(tag_match.group(2) or "").strip()
            meme_tag = str(tag_match.group(3) or "").strip()
            if mention_id:
                chain.append(At(qq=mention_id, name=""))
            elif image_id:
                image_comp = self._build_image_component_by_id(image_id)
                if image_comp is not None:
                    chain.append(image_comp)
            elif meme_tag:
                meme_id = self.meme_manager.get_random_meme_id(meme_tag)
                if meme_id:
                    image_comp = self._build_image_component_by_id(meme_id)
                    if image_comp is not None:
                        chain.append(image_comp)
            cursor = tag_match.end()

        if cursor < len(body):
            tail = body[cursor:]
            if tail:
                chain.append(Plain(tail))

        if not touched:
            return None
        return MessageChain(chain=chain)

    def _build_image_component_by_id(self, image_id: str) -> Image | None:
        key = str(image_id or "").strip()
        if not key:
            return None
        entry = self.image_manager.get_entry(key, self.image_url_lru)
        if entry is None:
            entry = self.meme_manager.get_image_entry(key)
        if not entry:
            return None

        local_path = str(entry.get("local_path") or "").strip()
        url = str(entry.get("url") or "").strip()

        image_source = local_path if local_path else url
        if not image_source:
            return None

        # 兼容不同 astrbot 版本的 Image 构造参数。
        for kwargs in ({"file": image_source}, {"url": image_source}):
            try:
                return Image(**kwargs)
            except Exception:
                continue
        return None

    def _sample_animated_frames(self, image_id: str, image_entry: dict[str, str], max_frames: int = 4) -> list[str]:
        if PILImage is None:
            return []

        local_path = Path(str(image_entry.get("local_path") or "").strip())
        if not local_path.exists() or not local_path.is_file():
            return []

        # 仅对常见动图格式进行抽帧。
        suffix = local_path.suffix.lower()
        if suffix not in {".gif", ".webp"}:
            return []

        try:
            with PILImage.open(local_path) as img:
                frame_count = int(getattr(img, "n_frames", 1) or 1)
                if frame_count <= 1:
                    return []

                sample_dir = local_path.parent / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)

                target_count = max(2, min(int(max_frames), frame_count))
                if target_count == frame_count:
                    frame_indexes = list(range(frame_count))
                else:
                    frame_indexes = sorted({
                        int(round(i * (frame_count - 1) / (target_count - 1)))
                        for i in range(target_count)
                    })

                sampled_files: list[str] = []
                for seq, frame_index in enumerate(frame_indexes):
                    img.seek(frame_index)
                    frame = img.convert("RGB")
                    sample_path = sample_dir / f"{image_id}_f{seq}_{frame_index}.jpg"
                    frame.save(sample_path, format="JPEG", quality=85, optimize=True)
                    sampled_files.append(str(sample_path))
                return sampled_files
        except Exception as exc:
            logger.warning("[myenhance] failed to sample animated image frames: %s", exc)
            return []

    def _get_meme_tags_prompt_block(self) -> str:
        tags = self.meme_manager.list_tags()
        if not tags:
            return "暂无可用 meme tag。"
        return "可用 meme tag：" + "、".join(tags)

    def _should_active_reply(self, event: AstrMessageEvent) -> bool:
        if not self.active_reply_enable:
            return False
        if self._format_poke_message(event):
            return False
        if event.get_sender_id() == event.get_self_id():
            return False
        if event.is_at_or_wake_command:
            return False

        group_id = event.get_group_id()
        if not group_id:
            return False
        if self.active_reply_whitelist and group_id not in self.active_reply_whitelist:
            return False

        scope_id = self._get_event_scope_id(event)
        if scope_id and self._is_active_reply_blocked(scope_id):
            logger.debug("[myenhance] skip active_reply: scope %s active-reply block is held", scope_id)
            return False
        if scope_id and self._is_scope_request_locked(scope_id):
            logger.debug("[myenhance] skip active_reply: scope %s request lock is held", scope_id)
            return False

        text = normalize_message_text(event, self.face_desc_map)
        if not text:
            return False
        return random.random() < self.active_reply_probability

    @filter.llm_tool(name="describe_image")
    async def describe_image_with_llm(
        self,
        event: AstrMessageEvent,
        image_id: str = "",
    ) -> str:
        """调用当前聊天模型描述图片内容。

        Args:
            image_id(string): 图片 ID（消息中的 [image:id]）。优先使用该参数。

        Returns:
            string: 图片内容描述文本（仅 content，不返回 keyword）。
        """
        target_image_id = str(image_id or "").strip()
        if not target_image_id:
            return "Error: image_id is required for describe_image."

        entry = self.image_manager.get_entry(target_image_id, self.image_url_lru)
        if not entry:
            return f"Error: image_id not found in cache: {target_image_id}"

        target_image = str(entry.get("url") or "").strip()
        if not target_image:
            return f"Error: image source missing for image_id: {target_image_id}"

        sample_frames = self._sample_animated_frames(target_image_id, entry)
        image_inputs = sample_frames if sample_frames else [target_image]

        cached_payload = self._get_cached_image_payload(target_image_id)
        if cached_payload:
            logger.debug("[myenhance] describe_image hit id cache: %s", target_image_id)
            _, content = cached_payload
            return content

        provider = None
        if self.describe_image_provider_id:
            provider = self.context.get_provider_by_id(self.describe_image_provider_id)
            if not provider:
                logger.warning(
                    "[myenhance] describe_image configured provider not found: %s",
                    self.describe_image_provider_id,
                )

        if not provider:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            return "Error: no provider found for current session."

        describe_prompt = (
            f"{self.describe_image_ask}\n\n"
            "请严格返回 JSON："
            '{"keyword":"关键词","content":"图片描述"}。'
            "其中 keyword 需为简短关键词，但是需要包含关键表达内容，content 为简洁客观描述。"
            "若输入为动图抽帧，请综合所有帧描述整体内容。"
        )

        try:
            resp = await provider.text_chat(
                prompt=describe_prompt,
                session_id=uuid.uuid4().hex,
                image_urls=image_inputs,
                persist=False,
            )
        except Exception as exc:
            logger.exception("[myenhance] describe_image failed")
            return f"Error: failed to describe image: {exc}"

        text = (getattr(resp, "completion_text", "") or "").strip()
        if not text:
            return "Error: image description result is empty."

        parsed = self._extract_summary_keyword_content(text)
        if parsed is None:
            content = re.sub(r"\s+", " ", text).strip()
            if not content:
                return "Error: image description result is invalid."
            keyword = self.image_manager.build_keyword(content)
        else:
            keyword, content = parsed
            keyword = re.sub(r"\s+", " ", str(keyword or "").strip())
            content = re.sub(r"\s+", " ", str(content or "").strip())
            if not content:
                return "Error: image description result is invalid."
            if not keyword:
                keyword = self.image_manager.build_keyword(content)

        self._set_cached_image_payload(target_image_id, keyword, content)
        return content

    @filter.llm_tool(name="add_meme")
    async def add_meme(self, event: AstrMessageEvent, id: str = "", tag: str = "") -> str:
        """把图片加入 meme 库，供 <meme tag="..."/> 随机发送。

        Args:
            id(string): 图片 ID（即 [image:id] 里的 id）。
            tag(string): meme 标签。
        """
        image_id = str(id or "").strip()
        meme_tag = str(tag or "").strip()
        if not image_id:
            return "Error: id is empty."
        if not meme_tag:
            return "Error: tag is empty."

        image_entry = self.image_manager.get_entry(image_id, self.image_url_lru)
        if image_entry is None:
            return f"Error: image_id not found in cache: {image_id}"

        ok, msg = self.meme_manager.add_meme(image_id, meme_tag, image_entry=image_entry)
        if not ok:
            return f"Error: {msg}"
        return f"add_meme success: id={image_id} tag={meme_tag} status={msg}"

    @filter.llm_tool(name="add_jargon")
    async def add_jargon(self, event: AstrMessageEvent, content: str = "", keyword: str = "") -> str:
        """添加一条可长期复用的黑话。

        Args:
            content(string): 需要保存的一句话黑话内容。
            keyword(string): 关联的关键词，用于黑话检索，使用空格分隔，是这条黑话的主语或要解释的对象，所有关键词应当指代同一对象。
        """
        normalized_content = str(content or "").strip()
        normalized_keyword = str(keyword or "").strip()
        if not normalized_content:
            return "Error: content is empty."
        if not normalized_keyword:
            return "Error: keyword is empty."

        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return "Error: no valid scope for jargon."

        embedding = None
        provider = self._get_embedding_provider()
        if provider:
            try:
                embedding = await provider.get_embedding(normalized_content)
            except Exception as exc:
                logger.warning("[myenhance] failed to get embedding for new jargon: %s", exc)

        try:
            record = self.jargon_store.add_jargon(
                scope_id,
                normalized_content,
                keyword=normalized_keyword,
                embedding=embedding,
            )
        except ValueError as exc:
            return f"Error: {exc}"

        logger.info("[myenhance] added jargon %s in scope %s (with embedding: %s)",
                    record.id, scope_id, bool(embedding))
        return f"Added jargon: id={record.id} content={record.content}"

    @filter.llm_tool(name="update_jargon")
    async def update_jargon(
        self,
        event: AstrMessageEvent,
        jargon_id: str = "",
        content: str = "",
        keyword: str = "",
    ) -> str:
        """根据黑话 ID 修改已有黑话。

        Args:
            jargon_id(string): 需要修改的黑话 ID。
            content(string): 修改后的黑话内容，可留空表示不修改。
            keyword(string): 修改后的关键词，用于黑话检索，使用空格分隔，是这条黑话的主语或要解释的对象，所有关键词应当指代同一对象。
        """
        normalized_id = str(jargon_id or "").strip()
        normalized_content = str(content or "").strip()
        if normalized_content == "":
            normalized_content = None
        normalized_keyword = str(keyword or "").strip()
        if normalized_keyword == "":
            normalized_keyword = None
        if not normalized_id:
            return "Error: jargon_id is empty."
        if normalized_content is None and normalized_keyword is None:
            return "Error: nothing to update."

        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return "Error: no valid scope for jargon."

        embedding = None
        provider = self._get_embedding_provider() if normalized_content is not None else None
        if provider and normalized_content is not None:
            try:
                embedding = await provider.get_embedding(normalized_content)
            except Exception as exc:
                logger.warning("[myenhance] failed to get embedding for updated jargon: %s", exc)

        record = self.jargon_store.update_jargon(
            scope_id,
            normalized_id,
            normalized_content,
            keyword=normalized_keyword,
            embedding=embedding,
        )
        if not record:
            return f"Error: jargon not found: {normalized_id}"

        logger.info("[myenhance] updated jargon %s in scope %s (with embedding: %s)",
                    record.id, scope_id, bool(embedding))
        return f"Updated jargon: id={record.id} content={record.content}"

    @filter.llm_tool(name="delete_jargon")
    async def delete_jargon(
        self,
        event: AstrMessageEvent,
        jargon_ids: list[str] | None = None,
        keyword: str = "",
        content: str = "",
        jargon_id: str = "",
    ) -> str:
        """删除一批黑话并新增一条替换黑话。

        Args:
            jargon_ids(list): 需要删除的黑话 ID 列表。
            keyword(string): 新增黑话的关键词。
            content(string): 新增黑话内容。
            jargon_id(string): 兼容旧调用的单个黑话 ID（会并入 jargon_ids）。
        """
        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return "Error: no valid scope for jargon."

        normalized_ids: list[str] = []
        for item in jargon_ids or []:
            value = str(item or "").strip()
            if value and value not in normalized_ids:
                normalized_ids.append(value)
        legacy_id = str(jargon_id or "").strip()
        if legacy_id and legacy_id not in normalized_ids:
            normalized_ids.append(legacy_id)
        if not normalized_ids:
            return "Error: jargon_ids is empty."

        normalized_keyword = str(keyword or "").strip()
        normalized_content = str(content or "").strip()
        if not normalized_keyword:
            return "Error: keyword is empty."
        if not normalized_content:
            return "Error: content is empty."

        deleted_ids: list[str] = []
        missing_ids: list[str] = []
        for target_id in normalized_ids:
            if self.jargon_store.delete_jargon(scope_id, target_id):
                deleted_ids.append(target_id)
            else:
                missing_ids.append(target_id)

        if not deleted_ids:
            return f"Error: jargon not found: {', '.join(missing_ids)}"

        embedding = None
        provider = self._get_embedding_provider()
        if provider:
            try:
                embedding = await provider.get_embedding(normalized_content)
            except Exception as exc:
                logger.warning("[myenhance] failed to get embedding for replacement jargon: %s", exc)

        try:
            record = self.jargon_store.add_jargon(
                scope_id,
                normalized_content,
                keyword=normalized_keyword,
                embedding=embedding,
            )
        except ValueError as exc:
            return f"Error: {exc}"

        missing_suffix = f" Missing: {', '.join(missing_ids)}." if missing_ids else ""
        return (
            f"Deleted jargon ids: {', '.join(deleted_ids)}. "
            f"Added jargon: id={record.id} content={record.content}.{missing_suffix}"
        )

    @filter.llm_tool(name="mute_member")
    async def mute_member(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        duration: int = 0,
        reason: str = "",
    ) -> str:
        """使用配置好的管理端接口强制禁言群成员。
        
        Args:
            user_id(string): 需要禁言的用户 ID。
            duration(number): 禁言时长（秒）
            reason(string): 禁言理由。
        """
        normalized_user = str(user_id or "").strip()
        if not normalized_user:
            return "Error: user_id 为必填"
        scope_id = event.get_group_id()
        if not scope_id:
            return "Error: 无法确定会话 group_id"
        try:
            target_group_id = int(str(scope_id))
            target_user_id = int(normalized_user)
        except ValueError:
            return "Error: group_id 或 user_id 非数字"
        try:
            requested_duration = max(0, int(duration))
        except (TypeError, ValueError):
            requested_duration = 0
        if self.mod_max_mute_duration > 0 and requested_duration > self.mod_max_mute_duration:
            requested_duration = self.mod_max_mute_duration
        reason_text = str(reason or "").strip() or "由 MyEnhance 管理工具触发"
        payload = self._compose_mute_payload(
            target_group_id,
            target_user_id,
            requested_duration,
            reason_text,
        )
        success, message = await self._send_mute_request(payload)
        return message if success else f"Error: {message}"

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def record_group_message(self, event: AstrMessageEvent):
        group_id = event.get_group_id()
        if not group_id:
            return
        if event.get_sender_id() == event.get_self_id():
            return

        await self._cache_recent_event(event)

        line = await self._format_member_message_async(event)
        event_ts = get_event_timestamp(event)
        msg_id = str(getattr(event.message_obj, "message_id", "") or "unknown")
        await self._record_line(group_id, event_ts, msg_id, line)

        if not self._should_active_reply(event):
            return

        session_curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin,
        )
        if not session_curr_cid:
            return

        conv = await self.context.conversation_manager.get_conversation(
            event.unified_msg_origin,
            session_curr_cid,
        )
        if not conv:
            return

        logger.info("[myenhance] active_reply triggered for group %s", group_id)
        yield event.request_llm(
            prompt=normalize_message_text(event, self.face_desc_map),
            session_id=event.session_id,
            conversation=conv,
        )

    @filter.on_llm_request()
    async def inject_group_history_to_prompt(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        self._start_context_summary_task_if_possible()
        scope_id = self._get_event_scope_id(event)
        if scope_id:
            self.scope_provider_origins[scope_id] = event.unified_msg_origin
        await self._mark_active_reply_block_start(scope_id)
        original_prompt = await self._format_member_message_async(event)
        request_lock = self._get_scope_request_lock(scope_id) if scope_id else None
        if request_lock and request_lock.locked():
            logger.info("[myenhance] waiting previous llm_request to finish summarization for scope %s", scope_id)
        if request_lock:
            async with request_lock:
                # 将框架上下文中“最后一个 user 之后”的新增内容并入自维护上下文（可包含工具调用等信息）。
                incremental_contexts = self._extract_new_context_after_last_user(req.contexts)
                for item in incremental_contexts:
                    await self._append_managed_context(scope_id, item)

                managed_contexts = await self._get_managed_contexts(scope_id)

                req.system_prompt = (
                    f"{req.system_prompt}\n\n{self.reply_system_prompt_cn}"
                    if req.system_prompt
                    else self.reply_system_prompt_cn
                )

                # 获取用于注入的历史消息和用于检索的全部消息文本
                history_lines, all_history_text = await self._get_recent_history_lines(event)

                # 1. 优先检索当前消息相关的黑话
                current_jargon = await self._get_related_jargon(
                    event,
                    original_prompt,
                    limit=self.jargon_current_recall_count,
                )

                # 2. 检索历史背景相关的黑话
                context_jargon = []
                context_memories = []
                if all_history_text:
                    context_jargon = await self._get_related_jargon(
                        event,
                        all_history_text,
                        limit=self.jargon_total_recall_count,
                    )
                    context_memories = await self._get_related_memories(event, all_history_text)

                # 3. 合并黑话并去重，保持当前消息的相关黑话在前
                seen_ids = set()
                jargons = []
                for m in current_jargon + context_jargon:
                    if m.id not in seen_ids:
                        jargons.append(m)
                        seen_ids.add(m.id)

                # 限制最终注入的数量（取配置值）
                jargons = jargons[:self.jargon_total_recall_count]

                seen_memory_ids = set()
                memories: list[MemoryRecord] = []
                for item in context_memories:
                    if item.id in seen_memory_ids:
                        continue
                    memories.append(item)
                    seen_memory_ids.add(item.id)
                memories = memories[:self.memory_recall_count]

                histroy_prompt = " 最近历史消息：\n" + "\n\n".join(history_lines)
                jargon_prompt = "相关黑话：\n" + "\n".join(f"[{record.id}] (关键词：{record.keyword}) {record.content}" for record in jargons)
                memory_prompt = "相关记忆：\n" + "\n".join(
                    f"[{record.id}] ({self._format_memory_time(record.created_at)}) {record.content}"
                    for record in memories
                )
                meme_tags_prompt = self._get_meme_tags_prompt_block()
                bot_id = self._get_bot_id(event)
                current_msg_id = getattr(event.message_obj, "message_id", "unknown")
                role_label = " (admin)" if event.is_admin() else "(member)"
                history_current_block = (
                    f"{histroy_prompt}\n\n"
                    f"你的id是{bot_id}。\n"
                    f"请回复下面这条消息{role_label}（#msg{current_msg_id}）:\n"
                    f"{original_prompt}\n"
                )

                req.contexts = self._inject_recent_memory_context_block(managed_contexts, scope_id)
                await self._append_managed_context(
                    scope_id,
                    {"role": "user", "content": history_current_block},
                )
                
                req.prompt = (
                    f"{history_current_block}"
                    "\n\n<MEMORY>\n"
                    f"{memory_prompt if memories else '暂无相关记忆。'}\n"
                    "</MEMORY>\n\n"
                    "<JARGON>\n"
                    "以下是相关黑话："
                    f"{jargon_prompt}"
                    "\n=====\n"
                    f"{self.jargon_prompt_rules}\n"
                    "=====\n"
                    "</JARGON>"
                    "<MEME_TAGS>\n"
                    f"{meme_tags_prompt}\n"
                    "若需发送该类表情，可输出 <meme tag=\"标签\"/>。\n"
                    "=====\n"
                    "如果你发现新的图片可以作为表情包，你可以使用工具add_meme把它加入表情库，这可以帮助你更好地与群友沟通。\n"
                    "</MEME_TAGS>\n\n"
                )

                logger.info(
                    "[myenhance] injected %s related memories, %s related jargon entries and %s history messages",
                    len(memories),
                    len(jargons),
                    len(history_lines),
                )
            return

        req.system_prompt = (
            f"{req.system_prompt}\n\n{self.reply_system_prompt_cn}"
            if req.system_prompt
            else self.reply_system_prompt_cn
        )
        req.contexts = []

    @filter.on_decorating_result()
    async def parse_control_tags_in_decorating_result(self, event: AstrMessageEvent):
        scope_id = self._get_event_scope_id(event)
        try:
            result = event.get_result()
            if not result or not result.chain:
                return
            if any(not isinstance(comp, Plain) for comp in result.chain):
                return

            text = "".join(comp.text for comp in result.chain if isinstance(comp, Plain))
            text = re.sub(
                r"<\s*think\s*>.*?<\s*/\s*think\s*>",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            text = re.sub(
                r"</\s*(?:mention|quote|image|meme)\s*>",
                "",
                text,
                flags=re.IGNORECASE,
            )
            if self.REFUSE_ONLY_RE.match(text):
                result.chain = []
                event.stop_event()
                logger.info("[myenhance] got <refuse/>, suppress outgoing message")
                return

            parsed_chain = self._parse_control_tags_to_chain(text)
            if not parsed_chain:
                return

            result.chain = parsed_chain.chain
            logger.info("[myenhance] parsed control tags into chain in on_decorating_result")
        finally:
            if scope_id and self._is_active_reply_blocked(scope_id):
                await self._mark_active_reply_block_end(scope_id)

