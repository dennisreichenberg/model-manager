"""Tests for model-manager."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_manager.ollama import ModelInfo, OllamaClient, _parse_model
from model_manager.usecases import KNOWN_USE_CASES, USE_CASE_MODELS


# --- ollama module ---

def _raw_model(**kwargs) -> dict:
    base = {
        "name": "llama3.1:8b",
        "size": 4_500_000_000,
        "modified_at": "2024-06-01T12:00:00Z",
        "digest": "sha256:abc123",
        "details": {
            "family": "llama",
            "parameter_size": "8B",
            "quantization_level": "Q4_0",
        },
    }
    base.update(kwargs)
    return base


def test_parse_model_basic():
    info = _parse_model(_raw_model())
    assert info.name == "llama3.1:8b"
    assert info.parameter_size == "8B"
    assert info.quantization == "Q4_0"
    assert info.size_gb == pytest.approx(4_500_000_000 / (1024**3), rel=0.01)
    assert info.size_bytes == 4_500_000_000


def test_parse_model_size_gb():
    info = _parse_model(_raw_model(size=2 * 1024**3))
    assert info.size_gb == pytest.approx(2.0, rel=0.01)


def test_parse_model_vram_estimate_slightly_above_size():
    info = _parse_model(_raw_model(size=4 * 1024**3))
    assert info.vram_estimate_gb > info.size_gb


def test_parse_model_missing_details():
    raw = {"name": "mystery:latest", "size": 1_000_000, "modified_at": "2024-01-01T00:00:00Z", "digest": ""}
    info = _parse_model(raw)
    assert info.parameter_size == "?"
    assert info.quantization == "?"
    assert info.family == "?"


def test_parse_model_bad_date_fallback():
    info = _parse_model(_raw_model(modified_at="not-a-date"))
    assert isinstance(info.modified_at, datetime)


# --- tags module ---

def test_add_and_get_tag(tmp_path, monkeypatch):
    store = tmp_path / "tags.json"
    monkeypatch.setattr("model_manager.tags._STORE_PATH", store)
    from model_manager import tags
    tags._STORE_PATH = store

    result = tags.add_tag("llama3.1:8b", "coding")
    assert "coding" in result
    assert tags.get_tags("llama3.1:8b") == ["coding"]


def test_add_tag_idempotent(tmp_path, monkeypatch):
    store = tmp_path / "tags.json"
    import model_manager.tags as tag_mod
    monkeypatch.setattr(tag_mod, "_STORE_PATH", store)

    tag_mod.add_tag("model:latest", "fast")
    tag_mod.add_tag("model:latest", "fast")
    assert tag_mod.get_tags("model:latest").count("fast") == 1


def test_remove_tag(tmp_path, monkeypatch):
    store = tmp_path / "tags.json"
    import model_manager.tags as tag_mod
    monkeypatch.setattr(tag_mod, "_STORE_PATH", store)

    tag_mod.add_tag("llama3:latest", "work")
    tag_mod.add_tag("llama3:latest", "fast")
    remaining = tag_mod.remove_tag("llama3:latest", "work")
    assert "work" not in remaining
    assert "fast" in remaining


def test_all_tags(tmp_path, monkeypatch):
    store = tmp_path / "tags.json"
    import model_manager.tags as tag_mod
    monkeypatch.setattr(tag_mod, "_STORE_PATH", store)

    tag_mod.add_tag("a:latest", "x")
    tag_mod.add_tag("b:latest", "y")
    data = tag_mod.all_tags()
    assert "a:latest" in data
    assert "b:latest" in data


# --- usecases module ---

def test_known_usecases_not_empty():
    assert len(KNOWN_USE_CASES) > 0


def test_each_usecase_has_models():
    for uc in KNOWN_USE_CASES:
        assert len(USE_CASE_MODELS[uc]) > 0, f"No models for use-case '{uc}'"


def test_coding_usecase_present():
    assert "coding" in KNOWN_USE_CASES
    assert "chat" in KNOWN_USE_CASES
    assert "embedding" in KNOWN_USE_CASES


# --- OllamaClient ---

def test_list_models_parses_response():
    raw_resp = {
        "models": [
            _raw_model(name="llama3.1:8b"),
            _raw_model(name="mistral:7b"),
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = raw_resp
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp

        client = OllamaClient()
        models = client.list_models()

    assert len(models) == 2
    names = {m.name for m in models}
    assert "llama3.1:8b" in names
    assert "mistral:7b" in names
