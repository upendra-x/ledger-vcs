"""``ledger fmt-info`` — show the frozen object format.

This exists for a specific operational reason. The format fingerprint is the
identity of a corpus: two Ledger deployments agree on object names if and only
if they agree on this value. When a client cannot deduplicate against a server,
or an imported environment fails to share chunks with its parent, the first
question is whether the two sides are running the same format — and this is how
you answer it in one command rather than by reading source on both machines.
"""

from __future__ import annotations

import typer
from rich.table import Table

from src.cli._shared import console
from src.format import constants as C
from src.text import exact_bytes

app = typer.Typer()


@app.command("fmt-info")
def fmt_info() -> None:
    """Print the frozen object-format constants and their fingerprint."""
    table = Table(
        title="Ledger object format — FROZEN",
        caption="Changing any of these starts a new corpus",
        title_style="bold",
        caption_style="dim italic",
    )
    table.add_column("Constant", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_column("Why it is fixed", style="dim")

    rows: list[tuple[str, str, str]] = [
        ("hash", f"BLAKE3-{C.DIGEST_BYTES * 8}", "the object name is the content hash"),
        ("format_version", str(C.FORMAT_VERSION), "inside the framing, so inside the hash"),
        (
            "object kinds",
            ", ".join(f"{k.name}={k.value}" for k in C.ObjectKind),
            "tag is byte 0 of every preimage",
        ),
        (
            "entry kinds",
            ", ".join(f"{k.name}={k.value}" for k in C.EntryKind),
            "CONFLICT reserved, rejected in v1",
        ),
        ("min chunk", exact_bytes(C.MIN_CHUNK_BYTES), "below this, per-object costs dominate"),
        ("avg chunk", exact_bytes(C.AVG_CHUNK_BYTES), "dedup resolution vs read count"),
        ("max chunk", exact_bytes(C.MAX_CHUNK_BYTES), "bounds one fetch and one buffer"),
        ("cut mask (short)", f"{C.CUT_MASK_SHORT:#018x}", "22 bits — harder, before the average"),
        ("cut mask (long)", f"{C.CUT_MASK_LONG:#018x}", "18 bits — easier, after the average"),
        ("gear table", f"{C.GEAR_TABLE_DIGEST[:16]}…", "decides every chunk boundary"),
        ("split domain", C.SPLIT_DOMAIN.decode(), "level-salted, content-defined"),
        ("split period", str(C.SPLIT_PERIOD), "expected entries per node"),
        (
            "split clamps",
            f"{C.SPLIT_MIN_ENTRIES}–{C.SPLIT_MAX_ENTRIES} entries, "
            f"≤{exact_bytes(C.MAX_NODE_BYTES)}",
            "bounds depth and node size",
        ),
    ]
    for row in rows:
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[bold]format fingerprint[/bold]  [green]{C.FORMAT_FINGERPRINT}[/green]")
    console.print("[dim]Two deployments share a corpus iff they share this value.[/dim]")
