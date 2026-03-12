from __future__ import annotations

import asyncio
from collections import OrderedDict, defaultdict, deque
import math
from pathlib import Path
import random
import re
from typing import Deque
import uuid

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.message_components import At, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.utils.quoted_message_parser import extract_quoted_message_images

from .utils.cache_manager import CacheManager
from .utils.face_map import load_face_desc_map
from .utils.hybrid_retrieval import hybrid_search
from .utils.memory_store import MemoryRecord, MemoryStore
from .utils.message_utils import extract_image_urls, format_time, get_event_timestamp, normalize_message_text
from .flask_ui import start_flask_app


@register("myenhance", "cjlqwq", "记录群消息并注入到 LLM 请求", "1.7.11")
class MyPlugin(Star):
    QUOTE_HEAD_RE = re.compile(r'<quote\s+id="([^"]+)"\s*/?>', re.IGNORECASE)
    MENTION_RE = re.compile(r'<mention\s+id="([^"]+)"\s*/?>', re.IGNORECASE)
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
        self.image_url_lru: OrderedDict[str, str] = OrderedDict()
        self.group_history_locks: dict[str, asyncio.Lock] = {}
        self.face_desc_map = load_face_desc_map()

        plugin_data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        plugin_data_path.mkdir(parents=True, exist_ok=True)
        self.cache_state_file = plugin_data_path / ".myenhance_cache_state.json"
        self.memory_store_file = plugin_data_path / ".myenhance_memories.json"
        self.cache_manager = CacheManager(
            self.cache_state_file,
            self.max_history,
            self.event_cache_size,
            self.image_url_cache_size,
        )
        self.cache_manager.load_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )
        self.memory_store = MemoryStore(self.memory_store_file, self.memory_max_records)
        self.reply_system_prompt_cn = self._build_reply_system_prompt()
        self._flask_server = None
        self._flask_thread = None
        self.stop_flask = None
        if self.web_port > 0:
            self.stop_flask = start_flask_app(self, self.web_port)

    async def terminate(self) -> None:
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
        return (
            "你正在群聊中进行消息回复。你的整个输出必须是发给群聊的一条回复消息，不要输出额外说明。\n\n"
            "第一步：请先检查当前对话内容和注入的上下文，判断是否有值得保存的稳定事实、约定，比如用户显式的告诉你缩写或黑称的含义。如果是，请立即调用 add_memory 或 update_memory。注意：记忆工具的调用不影响你对群聊回复的输出。\n\n"
            "第二步：根据上下文回复消息。请优先引用要回复的消息。若可从上下文中确定目标消息 ID，使用 <quote id=\"msg_id\"/> 并且必须放在输出最开头。\n"
            "每次只能引用一条消息。\n\n"
            "当需要提及用户时，使用 <mention id=\"user_id\"/>，可提及多个用户。\n"
            "注意你不应该直接输出用户ID，要提到用户时必须使用 mention 标签，且 mention 标签必须包含 id 属性，id 的值为用户ID。\n"
            "user_id 可从消息格式 [nickname/user_id/time] 中提取。\n"
            "mention 不是容器标签，绝对不要输出 </mention>。\n\n"
            "quote 不是容器标签，绝对不要输出 </quote>。\n"
            "若无法或不应回复，完整输出 <refuse/>，且前后不得有任何其他字符。\n\n"
            "系统会注入两类上下文：\n"
            "1. 与当前消息有关的记忆，格式为 <MEM>[mem-id] 内容；...</MEM>\n"
            "2. 最近历史消息，格式中包含 #msg消息ID 和消息内容。\n\n"
            "当你发现某个稳定事实、用户偏好、约定、长期任务背景值得保存时，并且不在<MEM>块内，调用 add_memory。\n"
            "add_memory 的参数 content 必须是一句可长期复用的记忆。\n"
            "当已有记忆不准确、过期或需要修正时，调用 update_memory。\n"
            "update_memory 需要传入 memory_id 和新的 content。只能修改已给出的记忆 ID。\n\n"
            "注意记忆中出现人物时，务必标注人物的ID，以便后续消息提及时能正确关联。\n\n"
            "注意：若存在不在<MEM>块内但值得记忆的稳定事实、约定，请**务必**调用 add_memory 添加记忆。\n"
            "你**不需要**记录群内发生了什么事，你只需要用户教你事实时调用工具。"
            "若用户的输入和你的记忆有偏差，请询问 admin 是否真实，你可以无条件相信 admin 给你提供的消息。\n"
            "当你确信信息有偏差后，请**务必**调用 update_memory 更新记忆。\n"
            "出现的人物请**务必**记录下对应的ID，以便后续消息提及时能正确关联。\n"
            "为了便于检索，你的 content 应当只包含需要的关键信息，不用包含更多的上下文信息，如发送人，时间，会话等不必要信息"
            "当你发现有重复的记忆时，或者有和你当前确认的消息不一致的记忆时，请**务必**调用 delete_memory 删除较简略的重复记忆，保留更详细的记忆。\n" 
            "你只会将完全确定的信息加入记忆管理库。"
            "当要回复的消息包含 [image] 时，先调用工具 describe_image。\n"
            "工具参数规则：msgid 使用目标消息的编号，即 #msg 后面的 message_id，image_index 从 1 开始。\n"
            "如果有多张图片，按需要多次调用 describe_image 再组织最终回复。\n\n"
            "语言要求：始终使用聊天室当前主要语言回复。\n"
            "除 quote/mention/refuse 控制标签外，不要输出多余的格式控制信息。"
        )

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

        raw_memory_recall_count = self.config.get("memory_recall_count", 5)
        try:
            self.memory_recall_count = max(0, int(raw_memory_recall_count))
        except (TypeError, ValueError):
            self.memory_recall_count = 5

        raw_history_inject_count = self.config.get("history_inject_count", 12)
        try:
            self.history_inject_count = max(0, int(raw_history_inject_count))
        except (TypeError, ValueError):
            self.history_inject_count = 12

        raw_memory_max_records = self.config.get("memory_max_records", 500)
        try:
            self.memory_max_records = max(1, int(raw_memory_max_records))
        except (TypeError, ValueError):
            self.memory_max_records = 500

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

        logger.debug(
            "[myenhance] config: active_reply=%s prob=%.4f history=%s memory_recall=%s memory_max=%s",
            self.active_reply_enable,
            self.active_reply_probability,
            self.history_inject_count,
            self.memory_recall_count,
            self.memory_max_records,
        )

    def _get_group_lock(self, group_id: str) -> asyncio.Lock:
        lock = self.group_history_locks.get(group_id)
        if lock is None:
            lock = asyncio.Lock()
            self.group_history_locks[group_id] = lock
        return lock

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

        image_urls = extract_image_urls(event)
        async with self._get_group_lock(scope_id):
            cached = self.recent_events[scope_id]
            deduped = [
                (cached_msg_id, urls)
                for cached_msg_id, urls in cached
                if cached_msg_id != msg_id
            ]
            cached.clear()
            cached.extend(deduped)
            cached.append((msg_id, image_urls))
        self.cache_manager.save_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )

    def _get_cached_image_desc(self, image_url: str) -> str | None:
        key = str(image_url or "").strip()
        if not key:
            return None
        value = self.image_url_lru.get(key)
        if value is None:
            return None
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
        self.cache_manager.save_cache_state(
            self.group_histories,
            self.recent_events,
            self.image_url_lru,
        )

    async def _get_cached_image_urls_by_msg_id(
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
                if item_ts > current_event_ts:
                    remaining_items.append((item_ts, item_msg_id, line))
            history.clear()
            history.extend(remaining_items)
        inject_lines = history_lines[-self.history_inject_count :] if self.history_inject_count > 0 else []
        search_query = "\n".join(search_lines)
        return inject_lines, search_query

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

    async def _build_embedding_scores(self, query: str, records: list[MemoryRecord]) -> list[float] | None:
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
                self.memory_store.save()

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

    async def _get_related_memories(self, event: AstrMessageEvent, query: str):
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

        cursor = 0
        for mention_match in self.MENTION_RE.finditer(body):
            touched = True
            if mention_match.start() > cursor:
                plain = body[cursor : mention_match.start()]
                if plain:
                    chain.append(Plain(plain))
            mention_id = mention_match.group(1).strip()
            if mention_id:
                chain.append(At(qq=mention_id, name=""))
            cursor = mention_match.end()

        if cursor < len(body):
            tail = body[cursor:]
            if tail:
                chain.append(Plain(tail))

        if not touched:
            return None
        return MessageChain(chain=chain)

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

        text = normalize_message_text(event, self.face_desc_map)
        if not text:
            return False
        return random.random() < self.active_reply_probability

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
            idx = max(1, int(image_index))
        except (TypeError, ValueError):
            idx = 1
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
            candidate_images = extract_image_urls(event)
        if not candidate_images:
            return "Error: no image found for describe_image."
        if selected_index >= len(candidate_images):
            return (
                f"Error: image_index out of range. Found {len(candidate_images)} image(s), but got {idx}."
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
                    "[myenhance] describe_image configured provider not found: %s",
                    self.describe_image_provider_id,
                )

        if not provider:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            return "Error: no provider found for current session."

        try:
            resp = await provider.text_chat(
                prompt=self.describe_image_ask,
                session_id=uuid.uuid4().hex,
                image_urls=[target_image],
                persist=False,
            )
        except Exception as exc:
            logger.exception("[myenhance] describe_image failed")
            return f"Error: failed to describe image: {exc}"

        text = (getattr(resp, "completion_text", "") or "").strip()
        if not text:
            return "Error: image description result is empty."

        self._set_cached_image_desc(target_image, text)
        return text

    @filter.llm_tool(name="add_memory")
    async def add_memory(self, event: AstrMessageEvent, content: str = "", keyword: str = "") -> str:
        """添加一条可长期复用的记忆。

        Args:
            content(string): 需要保存的一句话记忆内容。
            keyword(string): 关联的关键词，用于记忆检索，使用空格分格，是这条记忆的主语或要解释的对象。
        """
        normalized_content = str(content or "").strip()
        normalized_keyword = str(keyword or "").strip()
        if not normalized_content:
            return "Error: content is empty."
        if not normalized_keyword:
            return "Error: keyword is empty."

        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return "Error: no valid scope for memory."

        embedding = None
        provider = self._get_embedding_provider()
        if provider:
            try:
                embedding = await provider.get_embedding(normalized_content)
            except Exception as exc:
                logger.warning("[myenhance] failed to get embedding for new memory: %s", exc)

        try:
            record = self.memory_store.add_memory(
                scope_id,
                normalized_content,
                keyword=normalized_keyword,
                embedding=embedding,
            )
        except ValueError as exc:
            return f"Error: {exc}"

        logger.info("[myenhance] added memory %s in scope %s (with embedding: %s)", 
                    record.id, scope_id, bool(embedding))
        return f"Added memory: id={record.id} content={record.content}"

    @filter.llm_tool(name="update_memory")
    async def update_memory(
        self,
        event: AstrMessageEvent,
        memory_id: str = "",
        content: str = "",
        keyword: str = "",
    ) -> str:
        """根据记忆 ID 修改已有记忆。

        Args:
            memory_id(string): 需要修改的记忆 ID。
            content(string): 修改后的记忆内容，可留空表示不修改。
            keyword(string): 修改后的关键词，用于记忆检索，使用空格分格，是这条记忆的主语或要解释的对象。
        """
        normalized_id = str(memory_id or "").strip()
        normalized_content = str(content or "").strip()
        if normalized_content == "":
            normalized_content = None
        normalized_keyword = str(keyword or "").strip()
        if normalized_keyword == "":
            normalized_keyword = None
        if not normalized_id:
            return "Error: memory_id is empty."
        if normalized_content is None and normalized_keyword is None:
            return "Error: nothing to update."

        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return "Error: no valid scope for memory."

        embedding = None
        provider = self._get_embedding_provider() if normalized_content is not None else None
        if provider and normalized_content is not None:
            try:
                embedding = await provider.get_embedding(normalized_content)
            except Exception as exc:
                logger.warning("[myenhance] failed to get embedding for updated memory: %s", exc)

        record = self.memory_store.update_memory(
            scope_id,
            normalized_id,
            normalized_content,
            keyword=normalized_keyword,
            embedding=embedding,
        )
        if not record:
            return f"Error: memory not found: {normalized_id}"

        logger.info("[myenhance] updated memory %s in scope %s (with embedding: %s)", 
                    record.id, scope_id, bool(embedding))
        return f"Updated memory: id={record.id} content={record.content}"

    @filter.llm_tool(name="delete_memory")
    async def delete_memory(self, event: AstrMessageEvent, memory_id: str = "") -> str:
        """删除某条记忆，若检测到重复则优先清理较简略的内容。

        Args:
            memory_id(string): 直接删除指定 ID。
        """
        scope_id = self._get_event_scope_id(event)
        if not scope_id:
            return "Error: no valid scope for memory."

        if memory_id:
            records = self.memory_store.list_memories(scope_id)
            record = next((r for r in records if r.id == memory_id), None)
            if not record:
                return f"Error: memory not found: {memory_id}."
            normalized_content = (record.content or "").strip().lower()

            success = self.memory_store.delete_memory(scope_id, memory_id)
            if not success:
                return f"Error: memory not found: {memory_id}."

            message = [f"Deleted memory: id={memory_id}."]
            if normalized_content:
                records_after = self.memory_store.list_memories(scope_id)
                duplicates = [
                    r for r in records_after
                    if r.content.strip().lower() == normalized_content and r.id != memory_id
                ]
                if duplicates:
                    duplicates.sort(key=lambda r: len(r.content or ""))
                    shortest = duplicates[0]
                    longest = duplicates[-1]
                    if shortest.id != longest.id and self.memory_store.delete_memory(scope_id, shortest.id):
                        message.append(
                            f"发现重复记忆，删除内容较简略的 {shortest.id}，保留更详细的 {longest.id}."
                        )
            return " ".join(message)

        return "Error: memory_id is empty."

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def record_group_message(self, event: AstrMessageEvent):
        group_id = event.get_group_id()
        if not group_id:
            return
        if event.get_sender_id() == event.get_self_id():
            return

        await self._cache_recent_event(event)

        line = self._format_member_message(event)
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
        req.system_prompt = (
            f"{req.system_prompt}\n\n{self.reply_system_prompt_cn}"
            if req.system_prompt
            else self.reply_system_prompt_cn
        )

        original_prompt = self._format_member_message(event)
        
        # 获取用于注入的历史消息和用于检索的全部消息文本
        history_lines, all_history_text = await self._get_recent_history_lines(event)
        
        # 1. 优先检索当前消息相关的记忆
        logger.debug(f"[myenhance] retrieving related memories for current message with query: [{event.get_sender_id()}] {event.message_str}")
        current_memories = await self._get_related_memories(event, f"[{event.get_sender_id()}] {event.message_str}")
        
        # 2. 检索历史背景相关的记忆
        context_memories = []
        if all_history_text:
            context_memories = await self._get_related_memories(event, all_history_text)
            
        # 3. 合并记忆并去重，保持当前消息的相关记忆在前
        seen_ids = set()
        memories = []
        for m in current_memories + context_memories:
            if m.id not in seen_ids:
                memories.append(m)
                seen_ids.add(m.id)
        
        # 限制最终注入的数量（取配置值）
        memories = memories[:self.memory_recall_count]

        # 清理历史上下文中的 <MEM> 块，避免模型受到旧记忆干扰
        if req.contexts:
            import re
            mem_pattern = re.compile(r"<MEM>.*?</MEM>", re.DOTALL)
            for ctx in req.contexts:
                if isinstance(ctx, dict):
                    content = ctx.get("content")
                    if isinstance(content, str):
                        if "<MEM>" in content:
                            ctx["content"] = mem_pattern.sub("", content).strip()
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                if isinstance(text, str) and "<MEM>" in text:
                                    item["text"] = mem_pattern.sub("", text).strip()
                elif hasattr(ctx, "content"):
                    content = ctx.content
                    if isinstance(content, str):
                        if "<MEM>" in content:
                            ctx.content = mem_pattern.sub("", content).strip()
                    elif isinstance(content, list):
                        for item in content:
                            if (isinstance(item, dict) and item.get("type") == "text") or \
                               (hasattr(item, "type") and getattr(item, "type") == "text"):
                                
                                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
                                if isinstance(text, str) and "<MEM>" in text:
                                    if isinstance(item, dict):
                                        item["text"] = mem_pattern.sub("", text).strip()
                                    else:
                                        setattr(item, "text", mem_pattern.sub("", text).strip())

        sections: list[str] = []
        histroy_prompt = " 最近历史消息：\n" + "\n\n".join(history_lines)
        memories_prompt = "相关记忆：\n" + "\n".join(f"[{record.id}] (关键词：{record.keyword}) {record.content}" for record in memories)

        bot_id = self._get_bot_id(event)
        current_msg_id = getattr(event.message_obj, "message_id", "unknown")
        role_label = " (admin)" if event.is_admin() else "(member)"
        req.prompt = (
            f"{histroy_prompt}\n\n"
            f"你的id是{bot_id}。\n"
            f"请回复下面这条消息{role_label}（#msg{current_msg_id}）:\n"
            f"{original_prompt}\n\n"
            "<MEM>\n"
            "以下是相关记忆："
            f"{memories_prompt}"
            "\n=====\n"
            "请会议记忆管理完整流程：1. 检查是否删除重复记忆或错误记忆 2. 检查是否要更改记忆 3. 检查是否要新增记忆"
            "=====\n"
            "</MEM>"
        )

        logger.info(
            "[myenhance] injected %s related memories and %s history messages",
            len(memories),
            len(history_lines),
        )

    @filter.on_decorating_result()
    async def parse_control_tags_in_decorating_result(self, event: AstrMessageEvent):
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
            r"</\s*(?:mention|quote)\s*>",
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

