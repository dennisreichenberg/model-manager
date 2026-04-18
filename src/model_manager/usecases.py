"""Use-case to recommended model mapping."""

from __future__ import annotations

USE_CASE_MODELS: dict[str, list[str]] = {
    "coding": [
        "qwen2.5-coder:7b",
        "deepseek-coder-v2:16b",
        "codellama:13b",
        "codellama:7b",
    ],
    "chat": [
        "llama3.2:3b",
        "llama3.1:8b",
        "mistral:7b",
        "gemma2:9b",
    ],
    "embedding": [
        "nomic-embed-text:latest",
        "mxbai-embed-large:latest",
        "all-minilm:latest",
    ],
    "vision": [
        "llava:13b",
        "llava:7b",
        "moondream:latest",
    ],
    "reasoning": [
        "deepseek-r1:14b",
        "deepseek-r1:7b",
        "qwq:32b",
    ],
}

KNOWN_USE_CASES = list(USE_CASE_MODELS.keys())
