from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register


@register("myenhance", "cjlqwq", "记录群消息并注入到 LLM 请求", "1.1.1")
class MyPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self.config = config or {}

        raw_max_history = self.config.get("max_history", 300)
        try:
            parsed = int(raw_max_history)
        except (TypeError, ValueError):
            parsed = 300
        self.max_history = max(1, parsed)

        self.group_histories: dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    def _format_time(self, timestamp: int | float | None = None) -> str:
        if not timestamp:
            dt = datetime.now()
        else:
            dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _record_line(self, group_id: str, line: str) -> None:
        if not group_id:
            return
        self.group_histories[group_id].append(line)

    def _format_member_message(self, event: AstrMessageEvent) -> str:
        nickname = event.get_sender_name() or "unknown"
        sender_id = event.get_sender_id() or "unknown"
        role = "admin" if event.is_admin() else "member"
        timestamp = getattr(event.message_obj, "timestamp", None)
        msg_id = getattr(event.message_obj, "message_id", None) or "unknown"
        text = (event.message_str or "").strip()
        return (
            f"群友消息 [{nickname}/{sender_id}/{self._format_time(timestamp)}] "
            f"({role})#msg{msg_id}\n{text}"
        )

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def record_group_message(self, event: AstrMessageEvent):
        """记录群友消息。"""
        group_id = event.get_group_id()
        if not group_id:
            return

        if event.get_sender_id() == event.get_self_id():
            return

        line = self._format_member_message(event)

        self._record_line(group_id, line)

    @filter.on_llm_request()
    async def inject_group_history_to_prompt(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        """在 LLM 请求前，把群内已记录消息拼接到 req.prompt。"""
        group_id = event.get_group_id()
        if not group_id:
            return

        history = self.group_histories.get(group_id)
        if not history:
            return

        history_text = "\n\n".join(history)
        original_prompt = req.prompt or ""
        if original_prompt:
            req.prompt = (
                f"{history_text}\n\n"
                f"请回复下面这条消息:\n"
                f"{original_prompt}"
            )
        else:
            req.prompt = history_text
        logger.info(
            "[myenhance] injected %s history records into prompt for group %s",
            len(history),
            group_id,
        )
