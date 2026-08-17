"""``ledger import`` — bring an existing git repository in.

The numbers it prints are the point. The first open question
is whether
2 GiB of unique content per environment is the right assumption, and an import
over real history is the only thing here that can answer it — so the report is
what the conversion *actually* cost, not a spinner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.cli._shared import DataDir, console
from src.errors import LedgerError
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.migrate.from_git import GitImporter, unsupported_summary
from src.text import human_bytes

app = typer.Typer()


@app.command("import")
def import_repository(
    env: Annotated[str, typer.Argument(help="org/environment to import into.")],
    from_git: Annotated[Path, typer.Option("--from-git", help="Path to a git repository.")],
    data_dir: DataDir = Path("./data"),
    ref: Annotated[str, typer.Option("--ref", "-r", help="Ledger ref to publish at.")] = (
        "refs/heads/main"
    ),
    source_ref: Annotated[
        str, typer.Option("--source-ref", help="Which git ref to import.")
    ] = "HEAD",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Import only the newest N commits.")
    ] = None,
    skip_unsupported: Annotated[
        bool,
        typer.Option(
            "--skip-unsupported",
            help="Import the rest and list what could not be converted faithfully.",
        ),
    ] = False,
    create: Annotated[
        bool, typer.Option("--create/--no-create", help="Create the environment if absent.")
    ] = True,
) -> None:
    """Convert a git repository's history into Ledger.

    Blobs are re-chunked rather than copied, so the imported history costs what
    its *content* costs rather than what git's representation costs. LFS
    pointers and submodules cannot be converted faithfully and stop the import
    unless you ask otherwise.
    """
    try:
        with Ledger(data_dir) as ledger:
            name = EnvName(env)
            try:
                env_id = ledger.repo.resolve_env_name(name)
            except LedgerError:
                if not create:
                    raise
                env_id = ledger.repo.create_env(name).env_id

            report = GitImporter(ledger, skip_unsupported=skip_unsupported).import_repository(
                from_git,
                env_id,
                RefName(ref),
                source_ref=source_ref,
                limit=limit,
            )
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        for key, value in exc.details.items():
            console.print(f"  [dim]{key}[/dim] {value}")
        raise typer.Exit(1) from exc

    console.print(f"[green]{report.head}[/green]  [dim]({ref})[/dim]\n")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("commits converted", str(report.commits))
    table.add_row("trees converted", str(report.trees))
    table.add_row("blobs converted", str(report.blobs))
    table.add_row("git objects already held", str(report.reused))
    table.add_row("content seen", human_bytes(report.git_bytes))
    table.add_row(
        "objects created", f"{report.stats.objects_created} of {report.stats.objects_offered}"
    )
    table.add_row("bytes stored", human_bytes(report.stats.bytes_stored))
    table.add_row("bytes on disk", human_bytes(report.stats.bytes_on_disk))
    table.add_row("deduplicated away", f"{report.dedup_ratio:.1%}")
    console.print(table)

    if report.unsupported:
        console.print(
            f"\n[yellow]{len(report.unsupported)} paths could not be converted "
            f"faithfully[/yellow] [dim]({unsupported_summary(report.unsupported)})[/dim]"
        )
        for entry in report.unsupported[:20]:
            console.print(f"  [dim]{entry.reason}[/dim] {entry.path}")
        if len(report.unsupported) > 20:
            console.print(f"  [dim]…and {len(report.unsupported) - 20} more[/dim]")
        console.print(
            "\n[yellow]this environment is incomplete[/yellow] "
            "[dim]— it is missing the content those paths stood for[/dim]"
        )
