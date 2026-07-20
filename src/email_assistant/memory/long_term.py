"""Long-term memory: a JSON document store with remember/recall/search.

Deliberately simple (the assessment allows "vector DB, JSON store, etc."):
records live in one JSON file, search is keyword-overlap scoring with a
recency tiebreak. The public API is backend-agnostic, so a vector store could
replace the internals without touching callers.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import MemoryKind, MemoryRecord


class MemoryStore:
    def __init__(self, path: Path | str, seed_path: Path | str | None = None) -> None:
        self._path = Path(path)
        self._records: list[MemoryRecord] = []

        if self._path.exists():
            self._load(self._path)
        elif seed_path and Path(seed_path).exists():
            self._load(Path(seed_path))
            self._save()

    # -- public API ------------------------------------------------------------

    def remember(
        self,
        kind: MemoryKind,
        key: str,
        value: str,
        source: str = "agent",
    ) -> MemoryRecord:
        """Insert or update the record identified by (kind, key)."""
        now = datetime.now(timezone.utc)
        for record in self._records:
            if record.kind == kind and record.key == key:
                record.value = value
                record.source = source
                record.updated_at = now
                self._save()
                return record
        record = MemoryRecord(kind=kind, key=key, value=value, source=source, updated_at=now)
        self._records.append(record)
        self._save()
        return record

    def recall(self, kind: MemoryKind | None = None, key: str | None = None) -> list[MemoryRecord]:
        records = self._records
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if key is not None:
            records = [r for r in records if r.key == key]
        return list(records)

    def search(
        self,
        query: str,
        kind: MemoryKind | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Keyword-overlap scoring over key+value, recency as tiebreak."""
        terms = {t for t in re.split(r"\W+", query.lower()) if len(t) > 2}
        if not terms:
            return []
        scored: list[tuple[int, MemoryRecord]] = []
        for record in self.recall(kind=kind):
            text = f"{record.key} {record.value}".lower()
            score = sum(1 for t in terms if t in text)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].updated_at.timestamp()))
        return [record for _, record in scored[:limit]]

    def __len__(self) -> int:
        return len(self._records)

    # -- persistence -------------------------------------------------------------

    def _load(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._records = [MemoryRecord.model_validate(r) for r in raw["records"]]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [r.model_dump(mode="json") for r in self._records]}
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
