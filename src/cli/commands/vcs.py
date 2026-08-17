"""The version-control verbs, in jj's vocabulary.

The concepts are meant to transfer to agents that already
reason in `jj`: commits and *changes*, bookmarks, an operation log, and undo.
So the nouns here are jj's rather than git's — ``op log`` and ``undo`` exist,
``reflog`` and ``reset --hard`` do not — and a change id survives an amendment
the way an agent expects it to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import typer
from rich.table import Table

from src.cli._shared import DataDir, EnvOpt, RefOpt, console, fail
from src.errors import LedgerError
from src.ids import ChangeId, EnvName, ObjectName, RefName
from src.instance import Ledger
from src.service.commits import CommitService
from src.service.refs import RefService
from src.text import when

app = typer.Typer()


DEFAULT_REF = "refs/heads/main"


def _short(name: ObjectName) -> str:
    return name.hex[:12]


#: How much of a change id history shows. Named rather than inlined because the
#: value is a contract with ``ledger obslog``: this is the only form of a change
#: id anyone ever sees, so it is the form lookups have to accept. A test asserts
#: the two agree.
ABBREVIATED_CHANGE_ID: Final = 8


# ─────────────────────────────────────────────────────────────────────────────
# Environments
# ─────────────────────────────────────────────────────────────────────────────

env_app = typer.Typer(help="Create and inspect environments.")
app.add_typer(env_app, name="env")


@env_app.command("create")
def env_create(
    name: Annotated[str, typer.Argument(help="org/environment")],
    data_dir: DataDir = Path("./data"),
    owner: Annotated[str, typer.Option("--owner")] = "",
) -> None:
    """Create an environment."""
    try:
        with Ledger(data_dir) as ledger:
            created = ledger.repo.create_env(EnvName(name), owner=owner)
    except LedgerError as exc:
        raise fail(exc) from exc

    console.print(f"[green]{created.env_id}[/green]  {created.name}")


@env_app.command("list")
def env_list(
    data_dir: DataDir = Path("./data"),
    prefix: Annotated[str, typer.Option("--prefix", help="Name prefix filter.")] = "",
) -> None:
    """List environments."""
    with Ledger(data_dir) as ledger:
        names = ledger.repo.list_envs(prefix=prefix)
        rows = [(str(n), ledger.repo.resolve_env_name(n)) for n in names]

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan")
    table.add_column(style="dim")
    for name, env_id in rows:
        table.add_row(name, str(env_id))
    console.print(table)


@env_app.command("show")
def env_show(
    env: Annotated[str, typer.Argument(help="org/environment")], data_dir: DataDir = Path("./data")
) -> None:
    """Show an environment and its refs."""
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            record = ledger.repo.get_env(env_id)
            refs = ledger.repo.list_refs(env_id)
    except LedgerError as exc:
        raise fail(exc) from exc

    console.print(f"[cyan]{record.name}[/cyan]  [dim]{record.env_id}[/dim]")
    console.print(f"[dim]state[/dim]   {record.state}")
    if record.forked_from_env:
        console.print(f"[dim]forked[/dim]  from {record.forked_from_env}")
    console.print()

    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("ref", style="cyan")
    table.add_column("target", style="dim")
    table.add_column("gen", justify="right")
    table.add_column("lifecycle", style="dim")
    for ref in refs:
        table.add_row(str(ref.name), _short(ref.target), str(ref.generation), str(ref.lifecycle))
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Committing
# ─────────────────────────────────────────────────────────────────────────────


@app.command("commit")
def commit(
    source: Annotated[Path, typer.Argument(help="Directory to commit.")],
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
    ref: RefOpt = DEFAULT_REF,
    message: Annotated[str, typer.Option("--message", "-m")] = "",
    author: Annotated[str, typer.Option("--author")] = "ledger-cli",
    key: Annotated[
        str | None, typer.Option("--idempotency-key", help="Makes a retry safe to repeat.")
    ] = None,
) -> None:
    """Commit a directory as the next version of a ref."""
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            result = CommitService(ledger).commit(
                env_id,
                RefName(ref),
                source,
                author=author,
                message=message,
                idempotency_key=key,
            )
    except LedgerError as exc:
        raise fail(exc) from exc

    marker = " [dim](replayed)[/dim]" if result.replayed else ""
    console.print(f"[green]{result.commit}[/green]{marker}")
    console.print(
        f"[dim]{result.ref} generation {result.generation} · "
        f"{result.stats.objects_created} new objects, "
        f"{result.stats.bytes_stored:,} bytes stored of "
        f"{result.stats.bytes_offered:,} offered[/dim]"
    )


@app.command("resolve")
def resolve(env: EnvOpt, data_dir: DataDir = Path("./data"), ref: RefOpt = DEFAULT_REF) -> None:
    """Resolve a ref to a commit — the only read that touches mutable state."""
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            typer.echo(str(CommitService(ledger).resolve(env_id, RefName(ref))))
    except LedgerError as exc:
        raise fail(exc) from exc


@app.command("log")
def log(
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
    ref: RefOpt = DEFAULT_REF,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Show history. A plain read — snapshots, not a delta replay."""
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            entries = CommitService(ledger).log(env_id, RefName(ref), limit=limit)
    except LedgerError as exc:
        raise fail(exc) from exc

    for entry in entries:
        console.print(
            f"[yellow]{_short(entry.name)}[/yellow] "
            f"[magenta]{entry.commit.change_id.value[:ABBREVIATED_CHANGE_ID]}[/magenta] "
            f"[dim]{when(entry.commit.timestamp_us)}  {entry.commit.author}[/dim]"
        )
        console.print(f"  {entry.commit.message or '(no message)'}")


@app.command("revert")
def revert(
    to: Annotated[str, typer.Argument(help="Commit to go back to.")],
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
    ref: RefOpt = DEFAULT_REF,
    author: Annotated[str, typer.Option("--author")] = "ledger-cli",
) -> None:
    """Go back to an earlier version.

    One row update; no content moves. The old version's bytes never went
    anywhere, because every commit is a full snapshot over shared content.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            result = CommitService(ledger).revert(
                env_id, RefName(ref), ObjectName.parse(to), author=author
            )
    except LedgerError as exc:
        raise fail(exc) from exc

    console.print(f"[green]{result.commit}[/green]  [dim]generation {result.generation}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Bookmarks (refs)
# ─────────────────────────────────────────────────────────────────────────────

bookmark_app = typer.Typer(help="Create and delete refs. jj calls these bookmarks.")
app.add_typer(bookmark_app, name="bookmark")


@bookmark_app.command("set")
def bookmark_set(
    name: Annotated[str, typer.Argument(help="Ref name, e.g. refs/heads/exp/lr-3e4")],
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
    at: Annotated[
        str | None, typer.Option("--at", help="Commit; defaults to the default ref.")
    ] = None,
    ephemeral: Annotated[
        bool, typer.Option("--ephemeral", help="Discard automatically after --ttl-days.")
    ] = False,
    ttl_days: Annotated[int, typer.Option("--ttl-days")] = 14,
    author: Annotated[str, typer.Option("--author")] = "ledger-cli",
) -> None:
    """Create a branch.

    Costs no bytes: a branch is one metadata row over content that already
    exists. An ephemeral one carries a TTL so abandoned experiments cannot
    accumulate and pin their content forever.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            update = RefService(ledger).create(
                env_id,
                RefName(name),
                target=ObjectName.parse(at) if at else None,
                principal=author,
                ephemeral=ephemeral,
                ttl_days=ttl_days,
            )
    except LedgerError as exc:
        raise fail(exc) from exc

    console.print(f"[green]{update.ref.name}[/green] -> {_short(update.ref.target)}")


@bookmark_app.command("delete")
def bookmark_delete(
    name: Annotated[str, typer.Argument(help="Ref name.")],
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
    author: Annotated[str, typer.Option("--author")] = "ledger-cli",
) -> None:
    """Discard a branch.

    Instant and synchronous — no automation ever waits on collection. The
    storage comes back later and only once nothing else needs it, which includes
    the operation log: while the deletion is still within retention, ``undo`` can
    restore this branch, so its content is deliberately kept.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            outcome = RefService(ledger).delete(env_id, RefName(name), principal=author)
    except LedgerError as exc:
        raise fail(exc) from exc

    console.print(f"[green]deleted[/green] {name}")
    console.print(
        f"[dim]the environment's keep-set now holds {outcome.keep_set_size:,} objects. "
        f"Content this branch alone reached becomes collectable once the operation "
        f"log ages past its retention — until then `undo` can still restore it.[/dim]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The operation log
# ─────────────────────────────────────────────────────────────────────────────

op_app = typer.Typer(help="The operation log, and undo.")
app.add_typer(op_app, name="op")


@op_app.command("log")
def op_log(
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Every mutation, in order, with who did it.

    Appended in the same transaction as the mutation itself, so the log can
    never disagree with the refs it describes.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            ops = ledger.repo.list_ops(env_id, limit=limit)
    except LedgerError as exc:
        raise fail(exc) from exc

    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("op", style="cyan")
    table.add_column("ref")
    table.add_column("before", style="dim")
    table.add_column("after", style="dim")
    table.add_column("by", style="dim")
    for entry in ops:
        table.add_row(
            str(entry.sequence),
            str(entry.kind),
            str(entry.ref or ""),
            _short(entry.before) if entry.before else "",
            _short(entry.after) if entry.after else "",
            entry.principal,
        )
    console.print(table)


@app.command("undo")
def undo(
    sequence: Annotated[int, typer.Argument(help="Operation number to undo.")],
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
    author: Annotated[str, typer.Option("--author")] = "ledger-cli",
) -> None:
    """Undo an operation.

    An ordinary ref update, not a rewind: it expects the generation the ref has
    *now*, so a writer who moved it in the meantime produces the usual conflict
    and you decide whether their change survives.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            update = ledger.repo.undo(env_id, sequence, principal=author)
    except LedgerError as exc:
        raise fail(exc) from exc

    console.print(f"[green]{update.ref.name}[/green] -> {_short(update.ref.target)}")


@app.command("obslog")
def obslog(
    change: Annotated[str, typer.Argument(help="Change id, or any unambiguous prefix.")],
    env: EnvOpt,
    data_dir: DataDir = Path("./data"),
) -> None:
    """Every commit a change has had, newest first — jj's ``obslog``.

    A ``change_id`` is assigned once and survives rewrites, while the commit hash
    moves whenever anything changes. So this is
    how an automation names "the change that adds the verifier" across a rebase
    instead of chasing a hash — and neither resolution nor obsolescence is
    derivable from content, which is why the list is maintained explicitly under
    ``chg#`` rather than reconstructed.

    Takes the abbreviated id ``ledger log`` prints, because that is the only form
    anyone ever sees.
    """
    try:
        with Ledger(data_dir) as ledger:
            env_id = ledger.repo.resolve_env_name(EnvName(env))
            record = ledger.repo.resolve_change(env_id, ChangeId(change))
    except LedgerError as exc:
        raise fail(exc) from exc

    table = Table(box=None, pad_edge=False)
    table.add_column("commit", style="dim")
    table.add_column("")
    for index, commit in enumerate(record.commits):
        table.add_row(str(commit), "[green]current[/green]" if index == 0 else "superseded")
    console.print(table)
