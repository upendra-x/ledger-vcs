"""``ledger ingest`` — write a directory into the object store.

The output is deliberately the created-versus-offered counts rather than a
progress bar, because those two numbers are the deduplication requirement made
visible: ingest the same tree twice and the second run creates nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.cli._shared import console
from src.clock import SystemClock
from src.instance import Ledger
from src.runtime.ingest import Ingester
from src.text import human_bytes

app = typer.Typer()


@app.command("ingest")
def ingest(
    source: Annotated[Path, typer.Argument(help="Directory to ingest.")],
    data_dir: Annotated[
        Path, typer.Option("--data-dir", "-d", help="Ledger data directory.")
    ] = Path("./data"),
    message: Annotated[str, typer.Option("--message", "-m")] = "ingest",
    author: Annotated[str, typer.Option("--author")] = "ledger-cli",
    commit: Annotated[
        bool, typer.Option("--commit/--tree-only", help="Write a commit naming the tree.")
    ] = True,
) -> None:
    """Chunk, store and name a directory tree.

    Run it twice on the same directory: the second run creates zero objects and
    stores zero bytes, because every name is a hash of its own content.
    """
    with Ledger(data_dir) as ledger:
        ingester = Ingester(ledger.store, clock=SystemClock())
        if commit:
            name, stats = ingester.ingest_commit(source, author=author, message=message)
            label = "commit"
        else:
            name, stats = ingester.ingest_directory(source)
            label = "tree"

    console.print(f"[green]{name}[/green]  [dim]({label})[/dim]\n")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right")
    table.add_column(style="dim")

    table.add_row("walked", f"{stats.files:,}", "files")
    table.add_row("", f"{stats.directories:,}", "directories")
    if stats.symlinks:
        table.add_row("", f"{stats.symlinks:,}", "symlinks")
    table.add_row("", f"{stats.chunks:,}", "chunks")
    table.add_row(
        "objects",
        f"{stats.objects_created:,}",
        f"created of {stats.objects_offered:,} offered",
    )
    table.add_row(
        "bytes",
        human_bytes(stats.bytes_stored),
        f"stored of {human_bytes(stats.bytes_offered)} offered",
    )
    if stats.bytes_offered:
        table.add_row("deduplicated", f"{stats.dedup_ratio:.1%}", "of offered bytes")

    console.print(table)
