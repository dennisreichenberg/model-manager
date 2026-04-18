"""Local tag storage for model grouping (~/.model-manager/tags.json)."""

from __future__ import annotations

import json
from pathlib import Path


_STORE_PATH = Path.home() / ".model-manager" / "tags.json"


def _load() -> dict[str, list[str]]:
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, list[str]]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_tags(model: str) -> list[str]:
    return _load().get(model, [])


def add_tag(model: str, tag: str) -> list[str]:
    data = _load()
    existing = data.setdefault(model, [])
    if tag not in existing:
        existing.append(tag)
    _save(data)
    return existing


def remove_tag(model: str, tag: str) -> list[str]:
    data = _load()
    existing = data.get(model, [])
    data[model] = [t for t in existing if t != tag]
    _save(data)
    return data[model]


def all_tags() -> dict[str, list[str]]:
    return _load()
