from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

from astrbot.api import logger


@dataclass(slots=True)
class MemoryRecord:
    id: str
    content: str
    created_at: str
    updated_at: str


class MemoryStore:
    """按会话范围管理持久化记忆。"""

    def __init__(self, store_file: Path, max_records_per_scope: int = 500):
        self.store_file = store_file
        self.max_records_per_scope = max(1, int(max_records_per_scope))
        self.memories_by_scope: dict[str, list[MemoryRecord]] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_file.exists():
            return

        try:
            data = json.loads(self.store_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[myenhance] failed to load memory store: %s", exc)
            return

        scopes = data.get("scopes", {})
        if not isinstance(scopes, dict):
            return

        loaded: dict[str, list[MemoryRecord]] = {}
        for scope_id, items in scopes.items():
            if not isinstance(scope_id, str) or not isinstance(items, list):
                continue

            records: list[MemoryRecord] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                memory_id = str(item.get("id") or "").strip()
                content = str(item.get("content") or "").strip()
                created_at = str(item.get("created_at") or "").strip()
                updated_at = str(item.get("updated_at") or created_at or "").strip()
                if not memory_id or not content:
                    continue
                records.append(
                    MemoryRecord(
                        id=memory_id,
                        content=content,
                        created_at=created_at or self._now_iso(),
                        updated_at=updated_at or self._now_iso(),
                    )
                )

            if records:
                loaded[scope_id] = records[-self.max_records_per_scope :]

        self.memories_by_scope = loaded

    def save(self) -> None:
        try:
            payload = {
                "scopes": {
                    scope_id: [asdict(record) for record in records]
                    for scope_id, records in self.memories_by_scope.items()
                }
            }
            self.store_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[myenhance] failed to save memory store: %s", exc)

    def list_memories(self, scope_id: str) -> list[MemoryRecord]:
        return list(self.memories_by_scope.get(scope_id, []))

    def add_memory(self, scope_id: str, content: str) -> MemoryRecord:
        normalized_scope = str(scope_id or "").strip()
        normalized_content = str(content or "").strip()
        if not normalized_scope:
            raise ValueError("scope_id is empty")
        if not normalized_content:
            raise ValueError("content is empty")

        records = self.memories_by_scope.setdefault(normalized_scope, [])
        now = self._now_iso()
        record = MemoryRecord(
            id=self._next_id(records),
            content=normalized_content,
            created_at=now,
            updated_at=now,
        )
        records.append(record)
        if len(records) > self.max_records_per_scope:
            del records[: len(records) - self.max_records_per_scope]
        self.save()
        return record

    def update_memory(self, scope_id: str, memory_id: str, content: str) -> MemoryRecord | None:
        normalized_scope = str(scope_id or "").strip()
        normalized_id = str(memory_id or "").strip()
        normalized_content = str(content or "").strip()
        if not normalized_scope or not normalized_id or not normalized_content:
            return None

        for record in self.memories_by_scope.get(normalized_scope, []):
            if record.id != normalized_id:
                continue
            record.content = normalized_content
            record.updated_at = self._now_iso()
            self.save()
            return record
        return None

    def _next_id(self, records: list[MemoryRecord]) -> str:
        max_index = 0
        for record in records:
            if not record.id.startswith("mem-"):
                continue
            suffix = record.id[4:]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return f"mem-{max_index + 1:04d}"

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")
