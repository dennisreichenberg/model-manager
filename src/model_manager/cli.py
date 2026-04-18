"""CLI entry point for model-manager."""

from __future__ import annotations

from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .ollama import OllamaClient
from .tags import add_tag, all_tags, get_tags, remove_tag
from .usecases import KNOWN_USE_CASES, USE_CASE_MODELS

app = typer.Typer(
    name="model-manager",
    help="Unified Ollama model administration — list, pull, remove, and tag local models.",
    add_completion=False,
)
console = Console()
err = Console(stderr=True)


def _client(host: str) -> OllamaClient:
    return OllamaClient(base_url=host)


def _ollama_error(exc: Exception) -> None:
    err.print(f"\n[red]Cannot reach Ollama:[/red] {exc}")
    err.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
    raise typer.Exit(1)


@app.command("list")
def list_models(
    host: str = typer.Option("http://localhost:11434", "--host", "-H", help="Ollama base URL."),
    tag_filter: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag."),
) -> None:
    """List installed Ollama models with size, VRAM estimate, and tags."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]model-manager[/bold cyan] v{__version__} — Installed Models",
            border_style="cyan",
        )
    )
    console.print()

    try:
        models = _client(host).list_models()
    except Exception as exc:
        _ollama_error(exc)

    stored_tags = all_tags()

    if tag_filter:
        models = [m for m in models if tag_filter in stored_tags.get(m.name, [])]

    if not models:
        console.print("[dim]No models installed.[/dim]" if not tag_filter else f"[dim]No models tagged '{tag_filter}'.[/dim]")
        return

    table = Table(box=box.ROUNDED, border_style="dim", show_lines=False)
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Params", style="white", justify="right")
    table.add_column("Quant", style="yellow")
    table.add_column("Size", style="green", justify="right")
    table.add_column("VRAM est.", style="magenta", justify="right")
    table.add_column("Last used", style="dim")
    table.add_column("Tags", style="blue")

    for m in sorted(models, key=lambda x: x.name):
        tags = stored_tags.get(m.name, [])
        tag_str = ", ".join(tags) if tags else ""
        last_used = m.modified_at.strftime("%Y-%m-%d")
        table.add_row(
            m.name,
            m.parameter_size,
            m.quantization,
            f"{m.size_gb:.2f} GB",
            f"~{m.vram_estimate_gb:.2f} GB",
            last_used,
            tag_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(models)} model(s) installed.[/dim]")
    console.print()


@app.command("pull")
def pull_model(
    use_case: str = typer.Argument(
        ...,
        help=f"Use-case tag. Options: {', '.join(KNOWN_USE_CASES)}",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Pull a specific model name instead of use-case default.",
    ),
    host: str = typer.Option("http://localhost:11434", "--host", "-H", help="Ollama base URL."),
) -> None:
    """Pull a recommended model for a given use-case (coding, chat, embedding, ...)."""
    target: str
    if model:
        target = model
    elif use_case in USE_CASE_MODELS:
        candidates = USE_CASE_MODELS[use_case]
        target = candidates[0]
        console.print(f"\n[dim]Recommended models for [bold]{use_case}[/bold]: {', '.join(candidates)}[/dim]")
        console.print(f"[dim]Pulling top recommendation: [bold cyan]{target}[/bold cyan][/dim]\n")
    else:
        err.print(f"\n[red]Unknown use-case:[/red] '{use_case}'")
        err.print(f"[dim]Available: {', '.join(KNOWN_USE_CASES)}[/dim]")
        raise typer.Exit(1)

    console.print(f"[bold]Pulling [cyan]{target}[/cyan]...[/bold]\n")

    try:
        last_status = ""
        for status in _client(host).pull_model(target):
            if status != last_status:
                console.print(f"  [dim]{status}[/dim]")
                last_status = status
    except Exception as exc:
        _ollama_error(exc)

    console.print(f"\n[green]✓[/green] [bold]{target}[/bold] is ready.\n")


@app.command("rm")
def remove_model(
    model: str = typer.Argument(..., help="Model name to remove (e.g. llama3.1:8b)."),
    host: str = typer.Option("http://localhost:11434", "--host", "-H", help="Ollama base URL."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Remove an installed Ollama model."""
    if not yes:
        confirmed = typer.confirm(f"Delete model '{model}'?", default=False)
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    try:
        _client(host).delete_model(model)
    except Exception as exc:
        _ollama_error(exc)

    console.print(f"\n[green]✓[/green] Removed [bold cyan]{model}[/bold cyan].\n")


@app.command("tag")
def tag_model(
    model: str = typer.Argument(..., help="Model name to tag."),
    tag: str = typer.Argument(..., help="Tag to assign (e.g. 'work', 'fast', 'coding')."),
    remove: bool = typer.Option(False, "--remove", "-r", help="Remove this tag instead of adding it."),
) -> None:
    """Tag a model for grouping. Tags are stored locally in ~/.model-manager/tags.json."""
    if remove:
        remaining = remove_tag(model, tag)
        console.print(f"\n[yellow]−[/yellow] Removed tag [bold]{tag}[/bold] from [cyan]{model}[/cyan].")
        if remaining:
            console.print(f"  Remaining tags: {', '.join(remaining)}")
    else:
        tags = add_tag(model, tag)
        console.print(f"\n[green]+[/green] Tagged [cyan]{model}[/cyan] with [bold]{tag}[/bold].")
        console.print(f"  All tags: {', '.join(tags)}")
    console.print()


@app.command("tags")
def show_tags() -> None:
    """Show all model tags stored locally."""
    data = all_tags()
    if not data:
        console.print("\n[dim]No tags defined yet. Use: model-manager tag <model> <tag>[/dim]\n")
        return

    console.print()
    table = Table(title="Model Tags", box=box.ROUNDED, border_style="dim")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Tags", style="bold blue")

    for model_name, model_tags in sorted(data.items()):
        if model_tags:
            table.add_row(model_name, ", ".join(model_tags))

    console.print(table)
    console.print()


@app.command("usecases")
def list_usecases() -> None:
    """Show all available use-cases and their recommended models."""
    console.print()
    table = Table(title="Use-case Recommendations", box=box.ROUNDED, border_style="dim")
    table.add_column("Use-case", style="bold cyan")
    table.add_column("Recommended Models", style="white")

    for uc, models in USE_CASE_MODELS.items():
        table.add_row(uc, "\n".join(models))

    console.print(table)
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
