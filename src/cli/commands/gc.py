"""``ledger gc`` — plan and run collection.

Report-only is the default and ``--enforce`` is the flag you have to type.
Deletion is the only destructive operation in the system, so making the
destructive mode opt-in is worth the extra word — and a collector that can only
be run destructively is one nobody runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.cli._shared import DataDir, console
from src.errors import LedgerError
from src.ids import EnvName
from src.instance import Ledger
from src.maintenance.reclaim import run_maintenance
from src.service.refs import RefService
from src.text import human_bytes

app = typer.Typer(help="Garbage collection.")


@app.command("gc")
def gc(
    data_dir: DataDir = Path("./data"),
    enforce: Annotated[
        bool, typer.Option("--enforce", help="Actually delete. Off by default.")
    ] = False,
    rebuild: Annotated[
        str | None,
        typer.Option("--rebuild", help="Recompute this environment's keep-set first."),
    ] = None,
    skip_maintenance: Annotated[
        bool,
        typer.Option("--skip-maintenance", help="Do not expire refs, names and tombstones first."),
    ] = False,
) -> None:
    """Show what collection would reclaim, and optionally reclaim it.

    Expiring ephemeral refs, orphan name claims and stale tombstones runs first
    when enforcing, because all three decide what the diff is allowed to see: a
    branch whose TTL passed an hour ago is still a retention root until something
    removes it, and the sweep that follows would otherwise keep its content for
    another cycle.
    """
    maintenance = None
    try:
        with Ledger(data_dir) as ledger:
            if rebuild:
                recomputed = ledger.gc.rebuild_keep_set(rebuild)
                console.print(
                    f"[dim]rebuilt keep-set for {rebuild}: {recomputed:,} objects[/dim]\n"
                )
            if enforce and not skip_maintenance:
                refs = RefService(ledger)
                maintenance = run_maintenance(
                    ledger.repo,
                    ledger.store.tombstones,
                    clock=ledger.clock,
                    discard=lambda env, name: refs.delete(
                        env, name, principal="ledger-maintenance"
                    ),
                )
                # Keep-sets go stale on a timer, not only on deletion: the
                # operation-log entry holding a discarded branch alive ages out
                # at a moment no request coincides with.
                ledger.gc.refresh_keep_sets()
            report = ledger.gc.run(enforce=enforce)
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        raise typer.Exit(1) from exc

    plan = report.plan
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right")
    table.add_column(style="dim")

    table.add_row(
        "corpus", f"{plan.corpus_objects:,}", f"objects, {human_bytes(plan.corpus_bytes)}"
    )
    table.add_row("live", f"{plan.live_objects:,}", "objects reachable from roots")
    for guard, count in sorted(plan.protected.items()):
        table.add_row(f"protected by {guard}", f"{count:,}", "")
    table.add_row("candidates", f"{len(plan.candidates):,}", human_bytes(plan.bytes_reclaimable))
    if maintenance is not None and maintenance.total:
        table.add_row("expired refs", f"{maintenance.expired_refs:,}", "past their TTL")
        table.add_row("orphan names", f"{maintenance.orphan_names:,}", "claims with no environment")
        table.add_row(
            "tombstones purged", f"{maintenance.purged_tombstones:,}", "past their expiry"
        )
    console.print(table)

    if plan.aborted:
        console.print(f"\n[red]aborted by {plan.aborted_by}[/red]")
        console.print(f"[dim]{plan.abort_reason}[/dim]")
        raise typer.Exit(1)

    if report.enforced:
        console.print(
            f"\n[green]reclaimed[/green] {report.deleted:,} objects, "
            f"{human_bytes(report.bytes_freed)} — {report.tombstones_written:,} tombstones written"
        )
    else:
        console.print("\n[dim]report only. re-run with --enforce to reclaim.[/dim]")


@app.command("keepset")
def keepset(
    env: Annotated[str, typer.Argument(help="Environment, org/name.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Show how many objects an environment's keep-set holds."""
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            console.print(f"{ledger.keepsets.size(str(env_id)):,} objects")
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        raise typer.Exit(1) from exc
