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
    keyword: str
    created_at: str
    updated_at: str
    embedding: list[float] | None = None


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
                embedding = item.get("embedding")
                if not isinstance(embedding, list):
                    embedding = None

                keyword = str(item.get("keyword") or "").strip() or content
                if not memory_id or not content:
                    continue
                records.append(
                    MemoryRecord(
                        id=memory_id,
                        content=content,
                        keyword=keyword,
                        created_at=created_at or self._now_iso(),
                        updated_at=updated_at or self._now_iso(),
                        embedding=embedding,
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

    def add_memory(
        self,
        scope_id: str,
        content: str,
        keyword: str,
        embedding: list[float] | None = None,
    ) -> MemoryRecord:
        normalized_scope = str(scope_id or "").strip()
        normalized_content = str(content or "").strip()
        normalized_keyword = str(keyword or "").strip() or normalized_content
        if not normalized_scope:
            raise ValueError("scope_id is empty")
        if not normalized_content:
            raise ValueError("content is empty")

        records = self.memories_by_scope.setdefault(normalized_scope, [])
        now = self._now_iso()
        record = MemoryRecord(
            id=self._next_memory_id(records),
            content=normalized_content,
            keyword=normalized_keyword,
            created_at=now,
            updated_at=now,
            embedding=embedding,
        )
        records.append(record)
        if len(records) > self.max_records_per_scope:
            del records[: len(records) - self.max_records_per_scope]
        self.save()
        return record

    def update_memory(
        self,
        scope_id: str,
        memory_id: str,
        content: str | None = None,
        keyword: str | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryRecord | None:
        normalized_scope = str(scope_id or "").strip()
        normalized_id = str(memory_id or "").strip()
        normalized_content = None
        if content is not None:
            normalized_content = str(content or "").strip() or None
        normalized_keyword = None
        if keyword is not None:
            normalized_keyword = str(keyword or "").strip() or None
        if not normalized_scope or not normalized_id:
            return None
        if normalized_content is None and normalized_keyword is None and embedding is None:
            return None

        records = self.memories_by_scope.get(normalized_scope, [])
        for record in records:
            if record.id != normalized_id:
                continue
            if normalized_content is not None:
                record.content = normalized_content
            if normalized_keyword is not None:
                record.keyword = normalized_keyword
            if embedding is not None:
                record.embedding = embedding
            record.updated_at = self._now_iso()
            self.save()
            return record
        return None

    def delete_memory(self, scope_id: str, memory_id: str) -> bool:
        normalized_scope = str(scope_id or "").strip()
        normalized_id = str(memory_id or "").strip()
        if not normalized_scope or not normalized_id:
            return False
        records = self.memories_by_scope.get(normalized_scope)
        if not records:
            return False

        original_len = len(records)
        filtered = [record for record in records if record.id != normalized_id]
        if len(filtered) == original_len:
            return False

        self.memories_by_scope[normalized_scope] = filtered
        self.save()
        return True

    def _next_memory_id(self, records: list[MemoryRecord]) -> str:
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
