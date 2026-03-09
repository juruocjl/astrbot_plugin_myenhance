from __future__ import annotations

import asyncio
from collections import OrderedDict, defaultdict, deque
from datetime import datetime
import json
from pathlib import Path
import random
import re
from typing import Deque
import uuid

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.quoted_message_parser import extract_quoted_message_images


@register("myenhance", "cjlqwq", "记录群消息并注入到 LLM 请求", "1.3")
class MyPlugin(Star):
    QUOTE_HEAD_RE = re.compile(r'^\s*<quote\s+id="([^"]+)"\s*/>')
    MENTION_RE = re.compile(r'<mention\s+id="([^"]+)"\s*/>')
    REFUSE_ONLY_RE = re.compile(r'^\s*<refuse\s*/>\s*$')

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self.config = config or {}

        raw_max_history = self.config.get("max_history", 300)
        try:
            parsed = int(raw_max_history)
        except (TypeError, ValueError):
            parsed = 300
        self.max_history = max(1, parsed)

        self.active_reply_enable = bool(self.config.get("active_reply_enable", False))
        raw_probability = self.config.get("active_reply_probability", 0.1)
        try:
            parsed_probability = float(raw_probability)
        except (TypeError, ValueError):
            parsed_probability = 0.1
        self.active_reply_probability = min(max(parsed_probability, 0.0), 1.0)
        raw_whitelist = self.config.get("active_reply_whitelist", [])
        if isinstance(raw_whitelist, list):
            self.active_reply_whitelist = {str(i) for i in raw_whitelist if str(i).strip()}
        else:
            self.active_reply_whitelist = set()

        raw_event_cache_size = self.config.get("cached_size", 120)
        try:
            parsed_event_cache_size = int(raw_event_cache_size)
        except (TypeError, ValueError):
            parsed_event_cache_size = 120
        self.event_cache_size = max(1, parsed_event_cache_size)

        raw_describe_provider_id = self.config.get("describe_image_provider_id", "")
        self.describe_image_provider_id = str(raw_describe_provider_id or "").strip()

        raw_describe_ask = self.config.get(
            "describe_image_ask",
            "请客观描述这张图片中的主要内容，简洁一些。",
        )
        self.describe_image_ask = str(raw_describe_ask or "").strip() or (
            "请客观描述这张图片中的主要内容，简洁一些。"
        )

        raw_image_url_cache_size = self.config.get("image_url_cache_size", 120)
        try:
            parsed_image_url_cache_size = int(raw_image_url_cache_size)
        except (TypeError, ValueError):
            parsed_image_url_cache_size = 120
        self.image_url_cache_size = max(1, parsed_image_url_cache_size)

        logger.debug(
            "[myenhance] active_reply config: enable=%s probability=%.6f whitelist_size=%s",
            self.active_reply_enable,
            self.active_reply_probability,
            len(self.active_reply_whitelist),
        )

        self.group_histories: dict[str, Deque[tuple[float, str]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self.recent_events: dict[str, Deque[tuple[str, list[str]]]] = defaultdict(
            lambda: deque(maxlen=self.event_cache_size)
        )
        self.image_url_lru: OrderedDict[str, str] = OrderedDict()
        self.group_history_locks: dict[str, asyncio.Lock] = {}
        self.cache_state_file = Path(__file__).with_name(".myenhance_cache_state.json")
        self._load_cache_state()

        self.reply_system_prompt_cn = (
            "你正在群聊中进行消息回复。你的整个输出必须是发给群聊的一条回复消息，不要输出额外说明。\n\n"
            "请优先引用要回复的消息。若可从上下文中确定目标消息 ID，使用 <quote id=\"msg_id\"/> 并且必须放在输出最开头。\n"
            "每次只能引用一条消息。\n\n"
            "当需要提及用户时，使用 <mention id=\"user_id\"/>，可提及多个用户。\n"
            "user_id 可从消息格式 [nickname/user_id/time] 中提取。\n"
            "mention 不是容器标签，绝对不要输出 </mention>。\n\n"
            "quote 不是容器标签，绝对不要输出 </quote>。\n"
            "若无法或不应回复，完整输出 <refuse/>，且前后不得有任何其他字符。\n\n"
            "当要回复的消息包含 [image] 或 [image,summary=...] 时，先调用工具 describe_image。\n"
            "工具参数规则：msgid 使用目标消息的编号（即 #msg 后面的 message_id），image_index 从 1 开始。\n"
            "如果有多张图片，按需要多次调用 describe_image 再组织最终回复。\n\n"
            "语言要求：始终使用聊天室当前主要语言回复。\n"
            "除 quote/mention/refuse 控制标签外，不要输出多余的格式控制信息。"
        )

    def _format_time(self, timestamp: int | float | None = None) -> str:
        if not timestamp:
            dt = datetime.now()
        else:
            dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _get_group_lock(self, group_id: str) -> asyncio.Lock:
        lock = self.group_history_locks.get(group_id)
        if lock is None:
            lock = asyncio.Lock()
            self.group_history_locks[group_id] = lock
        return lock

    def _get_event_timestamp(self, event: AstrMessageEvent) -> float:
        timestamp = getattr(event.message_obj, "timestamp", None)
        try:
            if timestamp is None:
                return datetime.now().timestamp()
            return float(timestamp)
        except (TypeError, ValueError):
            return datetime.now().timestamp()

    def _normalize_message_text(self, event: AstrMessageEvent) -> str:
        """Normalize incoming message content for prompt/history rendering.

        For image segments, render as "[image]" or "[image,summary=...]".
        """
        messages = event.get_messages() or []
        if not messages:
            return (event.message_str or "").strip()

        has_image = False
        rendered_parts: list[str] = []

        for comp in messages:
            if isinstance(comp, Plain):
                text = (comp.text or "").strip()
                if text:
                    rendered_parts.append(text)
                continue

            comp_type = str(getattr(comp, "type", "")).lower()
            if isinstance(comp, dict):
                comp_type = str(comp.get("type", "")).lower()

            if comp_type == "image" or comp_type.endswith(".image"):
                has_image = True
                summary = getattr(comp, "summary", None)
                if summary is None:
                    summary = getattr(comp, "alt", None)
                if summary is None and isinstance(comp, dict):
                    data = comp.get("data", {})
                    if isinstance(data, dict):
                        summary = data.get("summary") or data.get("alt")
                summary_text = "" if summary is None else str(summary).strip()
                if summary_text:
                    rendered_parts.append(f"[image,summary={summary_text}]")
                else:
                    rendered_parts.append("[image]")

        if has_image:
            return " ".join(rendered_parts).strip() or "[image]"

        return (event.message_str or "").strip()

    async def _record_line(self, group_id: str, event_ts: float, line: str) -> None:
        if not group_id:
            return
        lock = self._get_group_lock(group_id)
        async with lock:
            self.group_histories[group_id].append((event_ts, line))
        self._save_cache_state()

    def _get_event_scope_id(self, event: AstrMessageEvent) -> str:
        return event.get_group_id() or event.unified_msg_origin

    async def _cache_recent_event(self, event: AstrMessageEvent) -> None:
        scope_id = self._get_event_scope_id(event)
        msg_id = str(getattr(event.message_obj, "message_id", "") or "").strip()
        if not scope_id or not msg_id:
            return
        image_urls = self._extract_image_urls(event)

        lock = self._get_group_lock(scope_id)
        async with lock:
            cached = self.recent_events[scope_id]
            if cached:
                deduped = [(mid, urls) for mid, urls in cached if mid != msg_id]
                cached.clear()
                cached.extend(deduped)
            cached.append((msg_id, image_urls))
        self._save_cache_state()

    def _get_cached_image_desc(self, image_url: str) -> str | None:
        key = str(image_url or "").strip()
        if not key:
            return None
        value = self.image_url_lru.get(key)
        if value is None:
            return None
        # LRU touch
        self.image_url_lru.move_to_end(key)
        return value

    def _set_cached_image_desc(self, image_url: str, description: str) -> None:
        key = str(image_url or "").strip()
        if not key:
            return
        self.image_url_lru[key] = description
        self.image_url_lru.move_to_end(key)
        while len(self.image_url_lru) > self.image_url_cache_size:
            self.image_url_lru.popitem(last=False)
        self._save_cache_state()

    def _load_cache_state(self) -> None:
        if not self.cache_state_file.exists():
            return
        try:
            data = json.loads(self.cache_state_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[myenhance] failed to load cache state: %s", e)
            return

        try:
            raw_group_histories = data.get("group_histories", {})
            if isinstance(raw_group_histories, dict):
                for group_id, items in raw_group_histories.items():
                    if not isinstance(group_id, str) or not isinstance(items, list):
                        continue
                    dq: Deque[tuple[float, str]] = deque(maxlen=self.max_history)
                    for item in items:
                        if not (isinstance(item, list) and len(item) == 2):
                            continue
                        try:
                            ts = float(item[0])
                        except (TypeError, ValueError):
                            continue
                        line = str(item[1])
                        dq.append((ts, line))
                    if dq:
                        self.group_histories[group_id] = dq

            raw_recent_events = data.get("recent_events", {})
            if isinstance(raw_recent_events, dict):
                for scope_id, items in raw_recent_events.items():
                    if not isinstance(scope_id, str) or not isinstance(items, list):
                        continue
                    dq2: Deque[tuple[str, list[str]]] = deque(maxlen=self.event_cache_size)
                    for item in items:
                        if not (isinstance(item, list) and len(item) == 2):
                            continue
                        msg_id = str(item[0]).strip()
                        urls_raw = item[1]
                        if not msg_id or not isinstance(urls_raw, list):
                            continue
                        urls = [str(u).strip() for u in urls_raw if str(u).strip()]
                        dq2.append((msg_id, urls))
                    if dq2:
                        self.recent_events[scope_id] = dq2

            raw_image_url_lru = data.get("image_url_lru", [])
            if isinstance(raw_image_url_lru, list):
                self.image_url_lru.clear()
                for item in raw_image_url_lru:
                    if not (isinstance(item, list) and len(item) == 2):
                        continue
                    k = str(item[0]).strip()
                    v = str(item[1]).strip()
                    if k and v:
                        self.image_url_lru[k] = v
                while len(self.image_url_lru) > self.image_url_cache_size:
                    self.image_url_lru.popitem(last=False)
        except Exception as e:
            logger.warning("[myenhance] failed to parse cache state: %s", e)

    def _save_cache_state(self) -> None:
        try:
            payload = {
                "group_histories": {
                    group_id: [[ts, line] for ts, line in history]
                    for group_id, history in self.group_histories.items()
                },
                "recent_events": {
                    scope_id: [[msg_id, urls] for msg_id, urls in entries]
                    for scope_id, entries in self.recent_events.items()
                },
                "image_url_lru": [[k, v] for k, v in self.image_url_lru.items()],
            }
            self.cache_state_file.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("[myenhance] failed to save cache state: %s", e)

    async def _get_cached_image_urls_by_msg_id(
        self,
        event: AstrMessageEvent,
        msg_id: str,
    ) -> list[str] | None:
        scope_id = self._get_event_scope_id(event)
        target_id = str(msg_id or "").strip()
        if not scope_id or not target_id:
            return None

        lock = self._get_group_lock(scope_id)
        async with lock:
            for cached_msg_id, cached_urls in reversed(self.recent_events.get(scope_id, [])):
                if cached_msg_id == target_id:
                    return cached_urls
        return None

    def _extract_image_urls(self, event: AstrMessageEvent) -> list[str]:
        """Collect image URLs/paths from structured message segments with raw fallback."""
        urls: list[str] = []

        for comp in event.get_messages() or []:
            if isinstance(comp, Image):
                url = (getattr(comp, "url", None) or getattr(comp, "file", None) or "").strip()
                if url:
                    urls.append(url)
                continue

            if isinstance(comp, dict):
                comp_type = str(comp.get("type", "")).lower()
                if comp_type != "image":
                    continue
                data = comp.get("data", {})
                if isinstance(data, dict):
                    url = str(data.get("url") or data.get("file") or "").strip()
                    if url:
                        urls.append(url)

        if urls:
            return urls
        return urls

    @filter.llm_tool(name="describe_image")
    async def describe_image_with_llm(
        self,
        event: AstrMessageEvent,
        msgid: str = "",
        image_index: int = 1,
    ) -> str:
        """调用当前聊天模型描述图片内容。

        Args:
            msgid(string): 需要描述的消息编号（message_id）。会尝试从该消息中提取图片。
            image_index(number): 第几张图片，从 1 开始。

        """
        target_msg_id = (msgid or "").strip()

        try:
            idx = int(image_index)
        except (TypeError, ValueError):
            idx = 1
        idx = max(1, idx)
        selected_index = idx - 1

        candidate_images: list[str] = []

        if target_msg_id:
            cached_images = await self._get_cached_image_urls_by_msg_id(event, target_msg_id)
            if cached_images:
                candidate_images = cached_images

        if not candidate_images and target_msg_id:
            candidate_images = await extract_quoted_message_images(
                event,
                reply_component=Reply(id=target_msg_id),
            )

        if not candidate_images:
            candidate_images = self._extract_image_urls(event)

        if not candidate_images:
            return "Error: no image found for describe_image."

        if selected_index >= len(candidate_images):
            return (
                f"Error: image_index out of range. Found {len(candidate_images)} image(s), "
                f"but got {idx}."
            )

        target_image = candidate_images[selected_index]

        cached_desc = self._get_cached_image_desc(target_image)
        if cached_desc:
            logger.debug("[myenhance] describe_image hit url cache")
            return cached_desc

        provider = None
        if self.describe_image_provider_id:
            provider = self.context.get_provider_by_id(self.describe_image_provider_id)
            if not provider:
                logger.warning(
                    "[myenhance] describe_image configured provider not found: %s, fallback to using provider",
                    self.describe_image_provider_id,
                )

        if not provider:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            return "Error: no provider found for current session."

        ask = self.describe_image_ask
        try:
            resp = await provider.text_chat(
                prompt=ask,
                session_id=uuid.uuid4().hex,
                image_urls=[target_image],
                persist=False,
            )
        except Exception as e:
            logger.exception("[myenhance] describe_image failed")
            return f"Error: failed to describe image: {e}"

        text = (getattr(resp, "completion_text", "") or "").strip()
        if not text:
            return "Error: image description result is empty."
        self._set_cached_image_desc(target_image, text)
        logger.debug("[myenhance] describe_image got response: %s", text)
        return text

    def _format_member_message(self, event: AstrMessageEvent) -> str:
        nickname = event.get_sender_name() or "unknown"
        sender_id = event.get_sender_id() or "unknown"
        role = "admin" if event.is_admin() else "member"
        timestamp = getattr(event.message_obj, "timestamp", None)
        msg_id = getattr(event.message_obj, "message_id", None) or "unknown"
        text = self._normalize_message_text(event)
        return (
            f"[{nickname}/{sender_id}/{self._format_time(timestamp)}] ({role})#msg{msg_id}\n{text}"
        )

    def _parse_control_tags_to_chain(self, text: str) -> MessageChain | None:
        """将模型输出中的控制标签解析为消息链组件。"""
        if not text:
            return None

        match = self.QUOTE_HEAD_RE.match(text)
        quote_id = match.group(1).strip() if match else ""
        body = text[match.end() :] if match else text
        touched = bool(match)
        chain: list = []

        if quote_id:
            chain.append(Reply(id=quote_id))

        cursor = 0
        for m in self.MENTION_RE.finditer(body):
            touched = True
            if m.start() > cursor:
                plain = body[cursor : m.start()]
                if plain:
                    chain.append(Plain(plain))
            mention_id = m.group(1).strip()
            if mention_id:
                chain.append(At(qq=mention_id, name=""))
            cursor = m.end()

        if cursor < len(body):
            tail = body[cursor:]
            if tail:
                chain.append(Plain(tail))

        if not touched:
            return None

        return MessageChain(chain=chain)

    def _should_active_reply(self, event: AstrMessageEvent) -> bool:
        if not self.active_reply_enable:
            logger.debug("[myenhance] active_reply skipped: feature disabled")
            return False
        if event.get_sender_id() == event.get_self_id():
            logger.debug("[myenhance] active_reply skipped: self message")
            return False
        if event.is_at_or_wake_command:
            logger.debug("[myenhance] active_reply skipped: wake/at command message")
            return False

        group_id = event.get_group_id()
        if not group_id:
            logger.debug("[myenhance] active_reply skipped: non-group message")
            return False
        if self.active_reply_whitelist and group_id not in self.active_reply_whitelist:
            logger.debug(
                "[myenhance] active_reply skipped: group %s not in whitelist",
                group_id,
            )
            return False

        text = self._normalize_message_text(event)
        if not text:
            logger.debug("[myenhance] active_reply skipped: empty text")
            return False

        roll = random.random()
        logger.debug(
            "[myenhance] active_reply roll=%.6f threshold=%.6f group=%s",
            roll,
            self.active_reply_probability,
            group_id,
        )
        if roll >= self.active_reply_probability:
            logger.debug("[myenhance] active_reply skipped: roll not hit")
            return False
        return True

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def record_group_message(self, event: AstrMessageEvent):
        """记录群友消息，并按概率触发主动回复。"""
        group_id = event.get_group_id()
        if not group_id:
            return

        if event.get_sender_id() == event.get_self_id():
            return

        await self._cache_recent_event(event)

        line = self._format_member_message(event)
        event_ts = self._get_event_timestamp(event)

        await self._record_line(group_id, event_ts, line)

        if not self._should_active_reply(event):
            return

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            logger.error("[myenhance] active_reply: no provider found")
            return

        session_curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin,
        )
        if not session_curr_cid:
            logger.info("[myenhance] active_reply skipped: no active conversation")
            return

        conv = await self.context.conversation_manager.get_conversation(
            event.unified_msg_origin,
            session_curr_cid,
        )
        if not conv:
            logger.info("[myenhance] active_reply skipped: conversation not found")
            return

        logger.info(
            "[myenhance] active_reply triggered: group=%s probability=%.3f",
            group_id,
            self.active_reply_probability,
        )
        normalized_prompt = self._normalize_message_text(event)
        yield event.request_llm(
            prompt=normalized_prompt,
            session_id=event.session_id,
            conversation=conv,
        )

    @filter.on_llm_request()
    async def inject_group_history_to_prompt(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        """在 LLM 请求前，把群内已记录消息拼接到 req.prompt。"""
        if req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n\n{self.reply_system_prompt_cn}"
        else:
            req.system_prompt = self.reply_system_prompt_cn

        group_id = event.get_group_id()
        if not group_id:
            return

        lock = self._get_group_lock(group_id)
        async with lock:
            history = self.group_histories.get(group_id)
            if not history:
                return

            current_event_ts = self._get_event_timestamp(event)
            pop_lines: list[str] = []
            remaining: Deque[tuple[float, str]] = deque(maxlen=self.max_history)

            for item_ts, item_line in history:
                if item_ts <= current_event_ts:
                    pop_lines.append(item_line)
                    continue
                remaining.append((item_ts, item_line))

            if not pop_lines:
                return

            history_text = "\n\n".join(pop_lines)
            history_count = len(pop_lines)
            self.group_histories[group_id] = remaining
        self._save_cache_state()

        original_prompt = (req.prompt or "").strip()
        if not original_prompt:
            original_prompt = self._normalize_message_text(event)
        if original_prompt:
            req.prompt = (
                f"{history_text}\n\n"
                f"请回复下面这条消息（#msg{getattr(event.message_obj, 'message_id', 'unknown')}）:\n"
                f"{original_prompt}"
            )
        else:
            req.prompt = history_text

        logger.info(
            "[myenhance] injected %s history records into prompt for group %s and popped only records older than current event timestamp",
            history_count,
            group_id,
        )

    @filter.on_decorating_result()
    async def parse_control_tags_in_decorating_result(self, event: AstrMessageEvent):
        """发送前把输出文本中的 quote/mention 标签解析为真实消息链。"""
        result = event.get_result()
        if not result or not result.chain:
            return

        # 仅处理纯文本结果，避免覆盖插件/平台已构造好的非文本消息段。
        if any(not isinstance(comp, Plain) for comp in result.chain):
            return

        text = "".join(comp.text for comp in result.chain if isinstance(comp, Plain))

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
