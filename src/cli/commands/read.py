"""``ledger checkout``, ``ledger cat``, ``ledger ls`` — the read path.

The *Read File* requirement is "read a file easily and at scale
without needing to pull all of the repos". ``ledger cat --offset`` is that
requirement at the command line: it reads a byte range out of the middle of a
file without materializing the environment, and reports how many objects it
touched so the cost claim is visible rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from src.cli._shared import DataDir, console
from src.errors import LedgerError
from src.format.constants import EntryKind
from src.format.model import Commit
from src.fs.blob import BlobReader
from src.fs.tree import list_dir, resolve_path
from src.ids import ObjectName
from src.instance import Ledger
from src.runtime.materialize import Materializer
from src.store.cas import ObjectStore
from src.text import human_bytes

app = typer.Typer()


def _root_tree(store: ObjectStore, name: ObjectName) -> ObjectName:
    """Accept either a commit or a tree, so callers need not remember which."""
    obj = store.get_object(name)
    return obj.tree if isinstance(obj, Commit) else name


@app.command("checkout")
def checkout(
    name: Annotated[str, typer.Argument(help="Commit or tree name.")],
    destination: Annotated[Path, typer.Argument(help="Where to write it.")],
    data_dir: DataDir = Path("./data"),
    force: Annotated[bool, typer.Option("--force", "-f", help="Replace the destination.")] = False,
) -> None:
    """Materialize a commit onto disk, byte for byte.

    Modes and symlinks are restored, because both are inside the hash — an old
    commit that brought back a non-executable verifier would not be the same
    version.
    """
    try:
        with Ledger(data_dir) as ledger:
            stats = Materializer(ledger.store).materialize_tree(
                _root_tree(ledger.store, ObjectName.parse(name)), destination, overwrite=force
            )
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        raise typer.Exit(1) from exc

    console.print(f"[green]checked out[/green] {destination}")
    console.print(
        f"[dim]{stats.files} files, {stats.directories} directories, "
        f"{stats.symlinks} symlinks, {human_bytes(stats.bytes_written)}, "
        f"{stats.objects_fetched} objects fetched[/dim]"
    )


@app.command("cat")
def cat(
    name: Annotated[str, typer.Argument(help="Commit or tree name.")],
    path: Annotated[str, typer.Argument(help="Path within the tree.")],
    data_dir: DataDir = Path("./data"),
    offset: Annotated[int, typer.Option("--offset", help="Start byte.")] = 0,
    length: Annotated[int | None, typer.Option("--length", help="Bytes to read.")] = None,
) -> None:
    """Read one file, or a byte range of it, without materializing anything.

    The whole environment stays where it is. Reading a megabyte out of the
    middle of a forty-gigabyte dataset costs a handful of object fetches, set by
    the depth of the structure rather than the size of the file.
    """
    try:
        with Ledger(data_dir) as ledger:
            resolved = resolve_path(
                ledger.store, _root_tree(ledger.store, ObjectName.parse(name)), path
            )
            if resolved.kind is EntryKind.TREE:
                console.print("[red]not_a_file[/red] that path is a directory")
                raise typer.Exit(1)
            payload = BlobReader(ledger.store, resolved.target).read(offset, length)
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        raise typer.Exit(1) from exc

    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


@app.command("ls")
def ls(
    name: Annotated[str, typer.Argument(help="Commit or tree name.")],
    path: Annotated[str, typer.Argument(help="Directory within the tree.")] = "",
    data_dir: DataDir = Path("./data"),
    limit: Annotated[int, typer.Option("--limit", "-n", help="Entries per page.")] = 100,
    after: Annotated[
        str | None, typer.Option("--after", help="Resume after this entry name.")
    ] = None,
) -> None:
    """List a directory, in name order, with a resumable cursor.

    The cursor is simply the last name you saw, which is what makes paging a
    134-million-entry directory work exactly like paging a six-entry one.
    """
    try:
        with Ledger(data_dir) as ledger:
            tree = _root_tree(ledger.store, ObjectName.parse(name))
            if path:
                resolved = resolve_path(ledger.store, tree, path)
                if resolved.kind is not EntryKind.TREE:
                    console.print("[red]not_a_directory[/red] that path is a file")
                    raise typer.Exit(1)
                tree = resolved.target
            page = list_dir(
                ledger.store, tree, after=after.encode() if after else None, limit=limit
            )
    except LedgerError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.message}")
        raise typer.Exit(1) from exc

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(justify="right", style="dim")
    table.add_column()

    for entry in page.entries:
        kind = "dir" if entry.kind is EntryKind.TREE else entry.kind.name.lower()
        marker = "*" if entry.mode == 0o755 else ""
        size = "-" if entry.kind is EntryKind.TREE else human_bytes(entry.size)
        table.add_row(kind, size, entry.name.decode(errors="replace") + marker)

    console.print(table)
    if page.cursor is not None:
        console.print(
            f"[dim]… more. resume with --after {page.cursor.decode(errors='replace')!r}[/dim]"
        )
