# model-manager

**Unified Ollama model administration.** List installed models with VRAM estimates, pull models by use-case, remove models, and tag them for grouping — all from one command.

## Install

```bash
pip install -e .
```

## Commands

### `model-manager list`

Show all installed Ollama models with size, VRAM estimate, quantization, and tags.

```
model-manager list
model-manager list --tag coding        # filter by tag
```

| Column    | Description                                |
|-----------|--------------------------------------------|
| Model     | Ollama model name                          |
| Params    | Parameter count (e.g. 8B)                  |
| Quant     | Quantization level (e.g. Q4_0)            |
| Size      | Disk size on disk                          |
| VRAM est. | Estimated VRAM required (≈ disk size)      |
| Last used | Last modified date                         |
| Tags      | User-defined tags                          |

### `model-manager pull <use-case>`

Pull the recommended model for a given use-case.

```
model-manager pull coding
model-manager pull chat
model-manager pull embedding
model-manager pull --model qwen2.5-coder:14b coding
```

Available use-cases: `coding`, `chat`, `embedding`, `vision`, `reasoning`

### `model-manager rm <model>`

Remove an installed model (with confirmation prompt).

```
model-manager rm llama3.1:8b
model-manager rm llama3.1:8b --yes    # skip confirmation
```

### `model-manager tag <model> <tag>`

Assign a tag to a model for grouping. Tags are stored in `~/.model-manager/tags.json`.

```
model-manager tag llama3.1:8b coding
model-manager tag mistral:7b fast
model-manager tag llama3.1:8b coding --remove    # remove tag
```

### `model-manager tags`

Show all tagged models.

### `model-manager usecases`

List all use-cases and their recommended models.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host`, `-H` | `http://localhost:11434` | Ollama base URL |
| `--tag`, `-t` | — | Filter `list` by tag |
| `--yes`, `-y` | false | Skip confirmation for `rm` |

## Requirements

- Python >= 3.10
- Ollama running locally (`ollama serve`)
