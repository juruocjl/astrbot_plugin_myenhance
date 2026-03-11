from __future__ import annotations
import json
from pathlib import Path
from astrbot.api import logger

def load_face_desc_map() -> dict[str, str]:
    """加载表情 ID 到描述的映射。"""
    # 假设 assets/data.json 是相对于本文件的父目录的 assets 文件夹
    data_file = Path(__file__).parent.parent / "assets" / "data.json"
    mapping: dict[str, str] = {}
    if not data_file.exists():
        return mapping

    try:
        raw = json.loads(data_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[myenhance] failed to load face data map: %s", e)
        return mapping

    if not isinstance(raw, list):
        return mapping

    for item in raw:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("describe") or item.get("QDes") or "").strip()
        if not desc:
            continue

        value = item.get("emojiId")
        if value is None:
            value = item.get("QSid")
        key = str(value).strip() if value is not None else ""
        if key and key not in mapping:
            mapping[key] = desc

    return mapping
