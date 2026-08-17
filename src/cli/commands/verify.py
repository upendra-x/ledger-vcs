"""``ledger verify-requirements`` — the traceability table, checked.

A table in a document says what somebody believed when they wrote it. This one
resolves each requirement to the tests that demonstrate it and reports whether
those tests are still there and still running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.cli._shared import console
from src.verify import REQUIREMENTS, verify

app = typer.Typer()


@app.command("verify-requirements")
def verify_requirements(
    root: Annotated[Path | None, typer.Option("--root", help="Project root to check.")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Name the evidence for each requirement.")
    ] = False,
) -> None:
    """Report which requirements this build demonstrates.

    Exits non-zero if any requirement has lost its evidence, so it can gate a
    release rather than merely inform one.
    """
    statuses = verify(root)

    table = Table(box=None, pad_edge=False)
    table.add_column("requirement")
    table.add_column("")
    table.add_column("demonstrated by", style="dim")
    for status in statuses:
        if status.satisfied:
            mark, detail = "[green]✓[/green]", f"{len(status.present)} tests"
        elif status.present:
            mark, detail = "[yellow]partial[/yellow]", _problem(status)
        else:
            mark, detail = "[red]✗[/red]", _problem(status)
        table.add_row(status.requirement.name, mark, detail)
    console.print(table)

    if verbose:
        for status in statuses:
            console.print(f"\n[bold]{status.requirement.name}[/bold]")
            console.print(f"  [dim]{status.requirement.asks_for}[/dim]")
            for node in status.present:
                console.print(f"  [green]✓[/green] {node}")
            for node in status.skipped:
                console.print(f"  [yellow]skipped[/yellow] {node}")
            for node in status.missing:
                console.print(f"  [red]missing[/red] {node}")

    satisfied = sum(1 for s in statuses if s.satisfied)
    total = len(REQUIREMENTS)
    colour = "green" if satisfied == total else "red"
    console.print(f"\n[{colour}]{satisfied}/{total}[/{colour}] requirements demonstrated")

    if satisfied != total:
        raise typer.Exit(1)


def _problem(status: object) -> str:
    from src.verify import RequirementStatus

    assert isinstance(status, RequirementStatus)
    parts = []
    if status.missing:
        parts.append(f"{len(status.missing)} missing")
    if status.skipped:
        parts.append(f"{len(status.skipped)} skipped")
    return ", ".join(parts)
