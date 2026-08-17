"""``ledger diff``, ``ledger fork``.

The output is deliberately quantitative. A diff prints how many objects it
touched, and a fork prints the bytes it copied — which is always zero. Those
numbers are the cost claims, and printing them is what makes them
checkable rather than asserted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.cli._shared import DataDir, console, fail
from src.errors import LedgerError
from src.format.model import Commit
from src.fs.diff import ChangeKind, diff_trees
from src.ids import EnvName, ObjectName, RefName
from src.instance import Ledger
from src.service.environments import EnvironmentService

app = typer.Typer()


_MARK = {
    ChangeKind.ADDED: ("+", "green"),
    ChangeKind.REMOVED: ("-", "red"),
    ChangeKind.MODIFIED: ("~", "yellow"),
}


@app.command("diff")
def diff(
    before: Annotated[str, typer.Argument(help="Commit to compare from.")],
    after: Annotated[str, typer.Argument(help="Commit to compare to.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """Compare two versions.

    Cost is proportional to what changed, not to the size of the environment:
    an unchanged subtree has an unchanged hash, so an entire branch is dismissed
    by comparing two names.
    """
    try:
        with Ledger(data_dir) as ledger:
            left = ledger.store.get_as(ObjectName.parse(before), Commit).tree
            right = ledger.store.get_as(ObjectName.parse(after), Commit).tree
            changes = list(diff_trees(ledger.store, left, right))
    except LedgerError as exc:
        raise fail(exc) from exc

    for change in changes:
        mark, colour = _MARK[change.kind]
        delta = f"{change.size_delta:+,}" if change.size_delta else ""
        console.print(f"[{colour}]{mark}[/{colour}] {change.display_path}  [dim]{delta}[/dim]")
    console.print(f"\n[dim]{len(changes)} path(s) changed[/dim]")


@app.command("fork")
def fork(
    source: Annotated[str, typer.Argument(help="Environment to fork, org/name.")],
    name: Annotated[str, typer.Argument(help="Name for the new environment.")],
    data_dir: DataDir = Path("./data"),
    from_ref: Annotated[str | None, typer.Option("--from-ref")] = None,
    owner: Annotated[str, typer.Option("--owner")] = "",
) -> None:
    """Fork an environment. Copies no bytes.

    Objects carry no environment identity, so a fork is one environment record
    and one ref pointing at a commit that already exists — and from that moment
    the two are independent.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(source))
            before = ledger.store.catalog.total()
            result = EnvironmentService(ledger).fork(
                env_id,
                EnvName(name),
                from_ref=RefName(from_ref) if from_ref else None,
                owner=owner,
            )
            after = ledger.store.catalog.total()
    except LedgerError as exc:
        raise fail(exc) from exc

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("environment", f"{result.environment.env_id}  {result.environment.name}")
    table.add_row("forked from", f"{result.source_commit.hex[:12]}")
    table.add_row("closure", f"{result.closure_size:,} objects now reachable")
    table.add_row("bytes copied", f"[green]{after[1] - before[1]}[/green]")
    console.print(table)
