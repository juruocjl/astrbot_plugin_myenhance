from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

from astrbot.api import logger


@dataclass(slots=True)
class JargonRecord:
    id: str
    content: str
    keyword: str
    created_at: str
    updated_at: str
    embedding: list[float] | None = None


class JargonStore:
    """按会话范围管理持久化黑话。"""

    def __init__(self, store_file: Path, max_records_per_scope: int = 500):
        self.store_file = store_file
        self.max_records_per_scope = max(1, int(max_records_per_scope))
        self.jargons_by_scope: dict[str, list[JargonRecord]] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_file.exists():
            return

        try:
            data = json.loads(self.store_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[myenhance] failed to load jargon store: %s", exc)
            return

        scopes = data.get("scopes", {})
        if not isinstance(scopes, dict):
            return

        loaded: dict[str, list[JargonRecord]] = {}
        for scope_id, items in scopes.items():
            if not isinstance(scope_id, str) or not isinstance(items, list):
                continue

            records: list[JargonRecord] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                jargon_id = str(item.get("id") or "").strip()
                content = str(item.get("content") or "").strip()
                created_at = str(item.get("created_at") or "").strip()
                updated_at = str(item.get("updated_at") or created_at or "").strip()
                embedding = item.get("embedding")
                if not isinstance(embedding, list):
                    embedding = None

                keyword = str(item.get("keyword") or "").strip()
                if not jargon_id or not content:
                    continue
                if not keyword:
                    keyword = content
                records.append(
                    JargonRecord(
                        id=jargon_id,
                        content=content,
                        keyword=keyword,
                        created_at=created_at or self._now_iso(),
                        updated_at=updated_at or self._now_iso(),
                        embedding=embedding,
                    )
                )

            if records:
                loaded[scope_id] = records[-self.max_records_per_scope :]

        self.jargons_by_scope = loaded

    def save(self) -> None:
        try:
            payload = {
                "scopes": {
                    scope_id: [asdict(record) for record in records]
                    for scope_id, records in self.jargons_by_scope.items()
                }
            }
            self.store_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[myenhance] failed to save jargon store: %s", exc)

    def list_jargons(self, scope_id: str) -> list[JargonRecord]:
        return list(self.jargons_by_scope.get(scope_id, []))

    def add_jargon(
        self,
        scope_id: str,
        content: str,
        keyword: str | None = None,
        embedding: list[float] | None = None,
    ) -> JargonRecord:
        normalized_scope = str(scope_id or "").strip()
        normalized_content = str(content or "").strip()
        normalized_keyword = str(keyword or "").strip()
        if not normalized_scope:
            raise ValueError("scope_id is empty")
        if not normalized_content:
            raise ValueError("content is empty")
        if not normalized_keyword:
            raise ValueError("keyword is empty")

        records = self.jargons_by_scope.setdefault(normalized_scope, [])
        now = self._now_iso()
        record = JargonRecord(
            id=self._next_jargon_id(records),
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

    def update_jargon(
        self,
        scope_id: str,
        jargon_id: str,
        content: str | None = None,
        keyword: str | None = None,
        embedding: list[float] | None = None,
    ) -> JargonRecord | None:
        normalized_scope = str(scope_id or "").strip()
        normalized_id = str(jargon_id or "").strip()
        normalized_content = None
        if content is not None:
            normalized_content = str(content or "").strip() or None
        normalized_keyword = None
        if keyword is not None:
            normalized_keyword = str(keyword or "").strip()
            if not normalized_keyword:
                return None
        if not normalized_scope or not normalized_id:
            return None
        if normalized_content is None and normalized_keyword is None and embedding is None:
            return None

        for record in self.jargons_by_scope.get(normalized_scope, []):
            if record.id != normalized_id:
                continue
            if normalized_content is not None:
                record.content = normalized_content
            record.updated_at = self._now_iso()
            if normalized_keyword is not None:
                record.keyword = normalized_keyword
            if embedding is not None:
                record.embedding = embedding
            self.save()
            return record
        return None

    def delete_jargon(self, scope_id: str, jargon_id: str) -> bool:
        normalized_scope = str(scope_id or "").strip()
        normalized_id = str(jargon_id or "").strip()
        if not normalized_scope or not normalized_id:
            return False

        if normalized_scope not in self.jargons_by_scope:
            return False

        records = self.jargons_by_scope[normalized_scope]
        original_len = len(records)
        new_records = [r for r in records if r.id != normalized_id]
        if len(new_records) < original_len:
            self.jargons_by_scope[normalized_scope] = new_records
            self.save()
            return True
        return False

    def _next_jargon_id(self, records: list[JargonRecord]) -> str:
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