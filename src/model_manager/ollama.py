"""Ollama HTTP API client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import httpx


@dataclass
class ModelInfo:
    name: str
    size_bytes: int
    modified_at: datetime
    parameter_size: str
    quantization: str
    family: str
    digest: str

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    @property
    def vram_estimate_gb(self) -> float:
        """Estimate VRAM needed: model file size is a good proxy for GGUF models."""
        return self.size_bytes / (1024**3) * 1.05


def _parse_model(raw: dict) -> ModelInfo:
    details = raw.get("details", {})
    modified_raw = raw.get("modified_at", "")
    try:
        modified = datetime.fromisoformat(modified_raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        modified = datetime.utcnow()
    return ModelInfo(
        name=raw["name"],
        size_bytes=raw.get("size", 0),
        modified_at=modified,
        parameter_size=details.get("parameter_size", "?"),
        quantization=details.get("quantization_level", "?"),
        family=details.get("family", "?"),
        digest=raw.get("digest", ""),
    )


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base = base_url.rstrip("/")

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self._base, timeout=300)

    def list_models(self) -> list[ModelInfo]:
        with self._client() as c:
            resp = c.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [_parse_model(m) for m in data.get("models", [])]

    def pull_model(self, name: str) -> Iterator[str]:
        """Stream pull progress lines (status strings)."""
        with httpx.Client(base_url=self._base, timeout=3600) as c:
            with c.stream("POST", "/api/pull", json={"name": name}) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        status = payload.get("status", "")
                        completed = payload.get("completed")
                        total = payload.get("total")
                        if completed and total:
                            pct = completed / total * 100
                            yield f"{status} [{pct:.1f}%]"
                        elif status:
                            yield status
                    except json.JSONDecodeError:
                        yield line

    def delete_model(self, name: str) -> None:
        with self._client() as c:
            resp = c.request("DELETE", "/api/delete", json={"name": name})
            resp.raise_for_status()

    def show_model(self, name: str) -> dict:
        with self._client() as c:
            resp = c.post("/api/show", json={"name": name})
            resp.raise_for_status()
            return resp.json()
