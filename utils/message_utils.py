from __future__ import annotations
from datetime import datetime
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Plain, Image, Json, Face, At, AtAll, Forward, Reply

def format_time(timestamp: int | float | None = None) -> str:
    """格式化时间戳。"""
    if not timestamp:
        dt = datetime.now()
    else:
        dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_event_timestamp(event: AstrMessageEvent) -> float:
    """提取事件时间戳。"""
    timestamp = getattr(event.message_obj, "timestamp", None)
    try:
        if timestamp is None:
            return datetime.now().timestamp()
        return float(timestamp)
    except (TypeError, ValueError):
        return datetime.now().timestamp()

def normalize_message_text(event: AstrMessageEvent, face_desc_map: dict[str, str]) -> str:
    """归一化消息链为文本形式。"""
    messages = event.get_messages() or []
    if not messages:
        return (event.message_str or "").strip()

    rendered_parts: list[str] = []

    for comp in messages:
        if isinstance(comp, Plain):
            rendered_parts.append(comp.text)
        elif isinstance(comp, Image):
            rendered_parts.append("[image]")
        elif isinstance(comp, Json):
            payload = getattr(comp, "data", None)
            prompt = ""
            if isinstance(payload, dict):
                prompt = str(payload.get("prompt") or "").strip()
            if prompt:
                rendered_parts.append(f"[Json:{prompt}]")
            else:
                rendered_parts.append("[Json]")
        elif isinstance(comp, Face):
            face_id = str(getattr(comp, "id", "") or "").strip()
            face_desc = face_desc_map.get(face_id)
            if face_desc:
                rendered_parts.append(f"[face:{face_desc}]")
            else:
                rendered_parts.append(f"[face:{face_id}]")
        elif isinstance(comp, At):
            rendered_parts.append(f"[at:{comp.name}/{comp.qq}]")
        elif isinstance(comp, AtAll):
            rendered_parts.append("[at:全体成员]")
        elif isinstance(comp, Forward):
            rendered_parts.append("[forward]")
        elif isinstance(comp, Reply):
            if getattr(comp, "id", ""):
                rendered_parts.append(f"[reply:{comp.id},{comp.sender_nickname}/{comp.sender_id}]")
            else:
                rendered_parts.append("[reply]")
        else:
            rendered_parts.append(f"[{getattr(comp, 'type', '消息')}]")

        rendered_parts.append(" ")

    normalized = "".join(rendered_parts).strip()
    return normalized or (event.message_str or "").strip()

def extract_image_urls(event: AstrMessageEvent) -> list[str]:
    """提取图片 URL。"""
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
    return urls
