from __future__ import annotations

import asyncio
from collections import OrderedDict, defaultdict, deque
import json
from pathlib import Path
import random
import re
import uuid

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.message_components import At, Face, Image, Plain, Reply, Json
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.utils.quoted_message_parser import extract_quoted_message_images

from .utils.face_map import load_face_desc_map
from .utils.message_utils import (
    format_time,
    get_event_timestamp,
    normalize_message_text,
    extract_image_urls,
)
from .utils.cache_manager import CacheManager


@register("myenhance", "cjlqwq", "记录群消息并注入到 LLM 请求", "1.4.0")
class MyPlugin(Star):
    QUOTE_HEAD_RE = re.compile(r'^\s*<quote\s+id="([^"]+)"\s*/>')
    MENTION_RE = re.compile(r'<mention\s+id="([^"]+)"\s*/>')
    REFUSE_ONLY_RE = re.compile(r'^\s*<refuse\s*/>\s*$')

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self.config = config or {}

        # 加载基础配置
        self._load_config()

        self.group_histories = defaultdict(lambda: deque(maxlen=self.max_history))
        self.recent_events = defaultdict(lambda: deque(maxlen=self.event_cache_size))
        self.image_url_lru: OrderedDict[str, str] = OrderedDict()
        self.group_history_locks: dict[str, asyncio.Lock] = {}
        
        # 加载表情映射
        self.face_desc_map = load_face_desc_map()
        
        # 初始化持久化存储
        plugin_data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        plugin_data_path.mkdir(parents=True, exist_ok=True)
        self.cache_state_file = plugin_data_path / ".myenhance_cache_state.json"
        self.cache_manager = CacheManager(
            self.cache_state_file,
            self.max_history,
            self.event_cache_size,
            self.image_url_cache_size
        )
        self.cache_manager.load_cache_state(self.group_histories, self.recent_events, self.image_url_lru)

        self.reply_system_prompt_cn = (
            "你正在群聊中进行消息回复。你的整个输出必须是发给群聊的一条回复消息，不要输出额外说明。\n\n"
            "请优先引用要回复的消息。若可从上下文中确定目标消息 ID，使用 <quote id=\"msg_id\"/> 并且必须放在输出最开头。\n"
            "每次只能引用一条消息。\n\n"
            "当需要提及用户时，使用 <mention id=\"user_id\"/>，可提及多个用户。\n"
            "user_id 可从消息格式 [nickname/user_id/time] 中提取。\n"
            "mention 不是容器标签，绝对不要输出 </mention>。\n\n"
            "quote 不是容器标签，绝对不要输出 </quote>。\n"
            "若无法或不应回复，完整输出 <refuse/>，且前后不得有任何其他字符。\n\n"
            "上下文中的每条消息记录格式为：\n"
            "[nickname/user_id/time] (role)#msg消息ID\\n消息内容\n"
            "其中 role 可能是 admin 或 member。\n\n"
            "消息内容中的消息段占位符说明：\n"
            "纯文本: 直接显示原文；\n"
            "图片: [image]；\n"
            "表情: [face:描述或id]；\n"
            "@用户: [at:昵称/qq]；\n"
            "@全体: [at:全体成员]；\n"
            "转发: [forward]；\n"
            "引用: [reply] 或 [reply:消息id,昵称/用户id]；\n"
            "其他类型: [type]。\n\n"
            "当要回复的消息包含 [image] 时，先调用工具 describe_image。\n"
            "工具参数规则：msgid 使用目标消息的编号（即 #msg 后面的 message_id），image_index 从 1 开始。\n"
            "如果有多张图片，按需要多次调用 describe_image 再组织最终回复。\n\n"
            "语言要求：始终使用聊天室当前主要语言回复。\n"
            "除 quote/mention/refuse 控制标签外，不要输出多余的格式控制信息。"
        )

    def _load_config(self):
        """解析插件配置。"""
        raw_max_history = self.config.get("max_history", 300)
        try: self.max_history = max(1, int(raw_max_history))
        except (TypeError, ValueError): self.max_history = 300

        self.active_reply_enable = bool(self.config.get("active_reply_enable", False))
        raw_probability = self.config.get("active_reply_probability", 0.1)
        try: self.active_reply_probability = min(max(float(raw_probability), 0.0), 1.0)
        except (TypeError, ValueError): self.active_reply_probability = 0.1
        
        raw_whitelist = self.config.get("active_reply_whitelist", [])
        self.active_reply_whitelist = {str(i) for i in raw_whitelist if str(i).strip()} if isinstance(raw_whitelist, list) else set()

        raw_event_cache_size = self.config.get("cached_size", 120)
        try: self.event_cache_size = max(1, int(raw_event_cache_size))
        except (TypeError, ValueError): self.event_cache_size = 120

        self.describe_image_provider_id = str(self.config.get("describe_image_provider_id", "") or "").strip()
        self.describe_image_ask = str(self.config.get("describe_image_ask", "请客观描述这张图片中的主要内容，简洁一些。")).strip()

        raw_image_url_cache_size = self.config.get("image_url_cache_size", 120)
        try: self.image_url_cache_size = max(1, int(raw_image_url_cache_size))
        except (TypeError, ValueError): self.image_url_cache_size = 120

        logger.debug("[myenhance] active_reply config: enable=%s probability=%.6f whitelist_size=%s",
                     self.active_reply_enable, self.active_reply_probability, len(self.active_reply_whitelist))

    def _get_group_lock(self, group_id: str) -> asyncio.Lock:
        lock = self.group_history_locks.get(group_id)
        if lock is None:
            lock = asyncio.Lock()
            self.group_history_locks[group_id] = lock
        return lock

    async def _record_line(self, group_id: str, event_ts: float, line: str) -> None:
        if not group_id: return
        async with self._get_group_lock(group_id):
            self.group_histories[group_id].append((event_ts, line))
        self.cache_manager.save_cache_state(self.group_histories, self.recent_events, self.image_url_lru)

    def _get_event_scope_id(self, event: AstrMessageEvent) -> str:
        return event.get_group_id() or event.unified_msg_origin

    async def _cache_recent_event(self, event: AstrMessageEvent) -> None:
        scope_id = self._get_event_scope_id(event)
        msg_id = str(getattr(event.message_obj, "message_id", "") or "").strip()
        if not scope_id or not msg_id: return
        image_urls = extract_image_urls(event)

        async with self._get_group_lock(scope_id):
            cached = self.recent_events[scope_id]
            deduped = [(mid, urls) for mid, urls in cached if mid != msg_id]
            cached.clear()
            cached.extend(deduped)
            cached.append((msg_id, image_urls))
        self.cache_manager.save_cache_state(self.group_histories, self.recent_events, self.image_url_lru)

    def _get_cached_image_desc(self, image_url: str) -> str | None:
        key = str(image_url or "").strip()
        if not key: return None
        value = self.image_url_lru.get(key)
        if value is not None:
            self.image_url_lru.move_to_end(key)
        return value

    def _set_cached_image_desc(self, image_url: str, description: str) -> None:
        key = str(image_url or "").strip()
        if not key: return
        self.image_url_lru[key] = description
        self.image_url_lru.move_to_end(key)
        while len(self.image_url_lru) > self.image_url_cache_size:
            self.image_url_lru.popitem(last=False)
        self.cache_manager.save_cache_state(self.group_histories, self.recent_events, self.image_url_lru)

    async def _get_cached_image_urls_by_msg_id(self, event: AstrMessageEvent, msg_id: str) -> list[str] | None:
        scope_id = self._get_event_scope_id(event)
        if not scope_id or not msg_id: return None
        async with self._get_group_lock(scope_id):
            for cached_msg_id, cached_urls in reversed(self.recent_events.get(scope_id, [])):
                if cached_msg_id == msg_id:
                    return cached_urls
        return None

    @filter.llm_tool(name="describe_image")
    async def describe_image_with_llm(self, event: AstrMessageEvent, msgid: str = "", image_index: int = 1) -> str:
        """调用当前聊天模型描述图片内容。
        Args:
            msgid(string): 需要描述的消息编号（message_id）。会尝试从该消息中提取图片。
            image_index(number): 第几张图片，从 1 开始。
        """
        target_msg_id = (msgid or "").strip()
        idx = max(1, int(image_index))
        selected_index = idx - 1

        candidate_images: list[str] = []
        if target_msg_id:
            cached_images = await self._get_cached_image_urls_by_msg_id(event, target_msg_id)
            if cached_images: candidate_images = cached_images

        if not candidate_images and target_msg_id:
            candidate_images = await extract_quoted_message_images(event, reply_component=Reply(id=target_msg_id))

        if not candidate_images: candidate_images = extract_image_urls(event)
        if not candidate_images: return "Error: no image found for describe_image."
        if selected_index >= len(candidate_images):
            return f"Error: image_index out of range. Found {len(candidate_images)} image(s), but got {idx}."

        target_image = candidate_images[selected_index]
        cached_desc = self._get_cached_image_desc(target_image)
        if cached_desc: return cached_desc

        provider = (self.context.get_provider_by_id(self.describe_image_provider_id) 
                   if self.describe_image_provider_id else None) or self.context.get_using_provider(event.unified_msg_origin)
        if not provider: return "Error: no provider found for current session."

        try:
            resp = await provider.text_chat(prompt=self.describe_image_ask, session_id=uuid.uuid4().hex,
                                          image_urls=[target_image], persist=False)
            text = (getattr(resp, "completion_text", "") or "").strip()
            if not text: return "Error: image description result is empty."
            self._set_cached_image_desc(target_image, text)
            return text
        except Exception as e:
            logger.exception("[myenhance] describe_image failed")
            return f"Error: failed to describe image: {e}"

    def _format_member_message(self, event: AstrMessageEvent) -> str:
        poke_text = self._format_poke_message(event)
        if poke_text: return poke_text

        nickname = event.get_sender_name() or "unknown"
        sender_id = event.get_sender_id() or "unknown"
        role = "admin" if event.is_admin() else "member"
        timestamp = getattr(event.message_obj, "timestamp", None)
        msg_id = getattr(event.message_obj, "message_id", None) or "unknown"
        text = normalize_message_text(event, self.face_desc_map)
        return f"[{nickname}/{sender_id}/{format_time(timestamp)}] ({role})#msg{msg_id}\n{text}"

    def _format_poke_message(self, event: AstrMessageEvent) -> str | None:
        raw_message = getattr(event.message_obj, "raw_message", None)
        if not isinstance(raw_message, dict) or str(raw_message.get("notice_type")) != "notify" or str(raw_message.get("sub_type")) != "poke":
            return None
        user_id = str(raw_message.get("user_id") or event.get_sender_id() or "unknown")
        target_id = str(raw_message.get("target_id") or "unknown")
        ts = raw_message.get("time") or getattr(event.message_obj, "timestamp", None)
        return f"[戳一戳/{format_time(ts)}]\n{user_id} 戳了戳 {target_id}"

    def _get_bot_id(self, event: AstrMessageEvent) -> str:
        return str(event.get_self_id() or "unknown").strip() or "unknown"

    def _parse_control_tags_to_chain(self, text: str) -> MessageChain | None:
        if not text: return None
        match = self.QUOTE_HEAD_RE.match(text)
        quote_id = match.group(1).strip() if match else ""
        body = text[match.end() :] if match else text
        touched, chain = bool(match), []

        if quote_id: chain.append(Reply(id=quote_id))
        cursor = 0
        for m in self.MENTION_RE.finditer(body):
            touched = True
            if m.start() > cursor:
                plain = body[cursor : m.start()]
                if plain: chain.append(Plain(plain))
            if m.group(1).strip(): chain.append(At(qq=m.group(1).strip(), name=""))
            cursor = m.end()

        if cursor < len(body):
            tail = body[cursor:]
            if tail: chain.append(Plain(tail))
        return MessageChain(chain=chain) if touched else None

    def _should_active_reply(self, event: AstrMessageEvent) -> bool:
        if not self.active_reply_enable or self._format_poke_message(event) or event.get_sender_id() == event.get_self_id() or event.is_at_or_wake_command:
            return False
        group_id = event.get_group_id()
        if not group_id or (self.active_reply_whitelist and group_id not in self.active_reply_whitelist):
            return False
        if not normalize_message_text(event, self.face_desc_map): return False
        roll = random.random()
        return roll < self.active_reply_probability

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def record_group_message(self, event: AstrMessageEvent):
        group_id = event.get_group_id()
        if not group_id or event.get_sender_id() == event.get_self_id(): return

        await self._cache_recent_event(event)
        line = self._format_member_message(event)
        event_ts = get_event_timestamp(event)
        await self._record_line(group_id, event_ts, line)

        if self._should_active_reply(event):
            session_curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not session_curr_cid: return
            conv = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, session_curr_cid)
            if not conv: return
            logger.info("[myenhance] active_reply triggered for group %s", group_id)
            yield event.request_llm(prompt=normalize_message_text(event, self.face_desc_map), session_id=event.session_id, conversation=conv)

    @filter.on_llm_request()
    async def inject_group_history_to_prompt(self, event: AstrMessageEvent, req: ProviderRequest):
        req.system_prompt = f"{req.system_prompt}\n\n{self.reply_system_prompt_cn}" if req.system_prompt else self.reply_system_prompt_cn
        group_id = event.get_group_id()
        if not group_id: return

        async with self._get_group_lock(group_id):
            history = self.group_histories.get(group_id)
            if not history: return
            current_event_ts = get_event_timestamp(event)
            pop_lines, remaining = [], deque(maxlen=self.max_history)
            for ts, ln in history:
                if ts <= current_event_ts: pop_lines.append(ln)
                else: remaining.append((ts, ln))
            if not pop_lines: return
            history_text = "\n\n".join(pop_lines)
            self.group_histories[group_id] = remaining
        self.cache_manager.save_cache_state(self.group_histories, self.recent_events, self.image_url_lru)

        original_prompt = (req.prompt or "").strip() or normalize_message_text(event, self.face_desc_map)
        if original_prompt:
            bot_id = self._get_bot_id(event)
            req.prompt = (f"{history_text}\n\n你的id是{bot_id}。\n请回复下面这条消息（#msg{getattr(event.message_obj, 'message_id', 'unknown')}）:\n{original_prompt}")
        else:
            req.prompt = history_text
        logger.info("[myenhance] injected %s history records into prompt", len(pop_lines))

    @filter.on_decorating_result()
    async def parse_control_tags_in_decorating_result(self, event: AstrMessageEvent):
        result = event.get_result()
        if not result or not result.chain or any(not isinstance(comp, Plain) for comp in result.chain):
            return
        text = "".join(comp.text for comp in result.chain)
        if self.REFUSE_ONLY_RE.match(text):
            result.chain = []
            event.stop_event()
            return
        parsed_chain = self._parse_control_tags_to_chain(text)
        if parsed_chain:
            result.chain = parsed_chain.chain

