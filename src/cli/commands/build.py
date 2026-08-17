"""``ledger build`` — drive the pipeline by hand.

In a running deployment the dispatcher and the workers are long-lived processes.
These verbs exist so the same pipeline can be stepped through one turn at a time,
which is how you debug a stuck environment and how the demo shows the whole loop
without a scheduler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.build.pipeline import BuildWorker, Dispatcher, trigger
from src.build.queue import BuildQueue
from src.build.results import BuildResults
from src.build.runner import RecordingRunner, SubprocessRunner
from src.build.sync import RecordingPlatform
from src.cli._shared import DataDir, console, fail
from src.errors import LedgerError
from src.ids import EnvName, ObjectName, RefName
from src.instance import Ledger

app = typer.Typer()

build_app = typer.Typer(help="The build and sync pipeline.")
app.add_typer(build_app, name="build")

DEFAULT_REF = "refs/heads/main"


@build_app.command("dispatch")
def build_dispatch(
    data_dir: DataDir = Path("./data"),
    limit: Annotated[int, typer.Option("--limit", help="Events to read this turn.")] = 200,
) -> None:
    """Turn ref-update events into queued builds.

    Reads the outbox each ref update wrote *inside its own transaction*, so an
    event can never describe a commit that was not published.
    """
    with Ledger(data_dir) as ledger:
        queue = BuildQueue(ledger.meta, clock=ledger.clock)
        report = Dispatcher(ledger, queue).poll(limit=limit)
        depth = queue.depth()

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("events read", str(report.events))
    table.add_row("builds queued", str(report.enqueued))
    table.add_row("ignored", str(report.ignored))
    table.add_row("queue depth", str(depth))
    console.print(table)


@build_app.command("run")
def build_run(
    data_dir: DataDir = Path("./data"),
    limit: Annotated[int, typer.Option("--limit", help="Builds to run this turn.")] = 20,
    worker: Annotated[str, typer.Option("--worker")] = "worker-1",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Record builds instead of running them.")
    ] = False,
) -> None:
    """Work until the queue is empty.

    ``--dry-run`` exercises the whole pipeline — lease, materialize, cache check,
    sync, note — without executing anybody's build command, which is how you
    check the wiring against a corpus you do not want to run code from.
    """
    runner = RecordingRunner() if dry_run else SubprocessRunner()
    platform = RecordingPlatform()

    with Ledger(data_dir) as ledger:
        outcomes = BuildWorker(ledger, runner=runner, platform=platform, worker_id=worker).drain(
            limit=limit
        )

    if not outcomes:
        console.print("[dim]nothing queued[/dim]")
        return

    table = Table(box=None, pad_edge=False)
    table.add_column("commit", style="dim")
    table.add_column("status")
    table.add_column("source")
    table.add_column("platform")
    for outcome in outcomes:
        table.add_row(
            outcome.commit[:19],
            str(outcome.result.status),
            "[green]cached[/green]" if outcome.cache_hit else "built",
            outcome.synced.platform_id if outcome.synced else "-",
        )
    console.print(table)
    cached = sum(1 for o in outcomes if o.cache_hit)
    console.print(f"\n[dim]{len(outcomes)} builds, {cached} served from cache[/dim]")


@build_app.command("trigger")
def build_trigger(
    env: Annotated[str, typer.Argument(help="org/environment")],
    ref: Annotated[str, typer.Option("--ref", "-r")] = DEFAULT_REF,
    data_dir: DataDir = Path("./data"),
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Forget the cached result first.")
    ] = False,
) -> None:
    """Ask for a build — reruns and backfills.

    A commit does not need this: publishing it queued its build already.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            queued = trigger(ledger, env_id, RefName(ref), rebuild=rebuild)
    except LedgerError as exc:
        raise fail(exc) from exc

    console.print("[green]queued[/green]" if queued else "[dim]already queued[/dim]")


@build_app.command("status")
def build_status(
    commit: Annotated[str, typer.Argument(help="Commit to look up.")],
    data_dir: DataDir = Path("./data"),
) -> None:
    """What a commit built to.

    Keyed by commit rather than by environment, which is why a fork can look up
    a build it never ran.
    """
    with Ledger(data_dir) as ledger:
        result = BuildResults(ledger.meta, clock=ledger.clock).get(ObjectName.parse(commit))

    if result is None:
        console.print("[dim]no build result for that commit[/dim]")
        raise typer.Exit(1)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("status", str(result.status))
    table.add_row("exit code", str(result.exit_code))
    table.add_row("duration", f"{result.duration_us / 1_000_000:.1f}s")
    table.add_row("attempts", str(result.attempts))
    table.add_row("worker", result.worker)
    table.add_row("images", "\n".join(result.images) or "-")
    console.print(table)


@build_app.command("failures")
def build_failures(
    data_dir: DataDir = Path("./data"),
    limit: Annotated[int, typer.Option("--limit")] = 50,
) -> None:
    """Every failed build in the corpus.

    One query, because results are keyed by commit rather than buried per
    environment — which is what makes a systemic build regression one signal
    instead of ten million silent ones.
    """
    with Ledger(data_dir) as ledger:
        failures = BuildResults(ledger.meta, clock=ledger.clock).failures(limit=limit)

    if not failures:
        console.print("[green]no failed builds[/green]")
        return

    table = Table(box=None, pad_edge=False)
    table.add_column("commit", style="dim")
    table.add_column("attempts", justify="right")
    table.add_column("exit")
    table.add_column("why")
    for result in failures:
        table.add_row(
            result.commit[:19],
            str(result.attempts),
            str(result.exit_code),
            result.log_excerpt.splitlines()[-1][:60] if result.log_excerpt else "-",
        )
    console.print(table)
