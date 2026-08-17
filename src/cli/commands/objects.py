"""``ledger hash-object`` and ``ledger cat-object`` — inspect the format directly.

git's plumbing commands earned their keep by making the object model something
you can poke at rather than infer, and the same applies here: when a client and
a server disagree about a name, being able to hash the same bytes on both sides
turns a mystery into a one-line comparison.

These read and write files rather than addressing stored objects. The output
format is derived from the codec rather than from storage, so it does not
depend on where the bytes came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.cli._shared import console
from src.format.codec import decode, encode, name_of_encoded, peek_kind
from src.format.model import Blob, Chunk, Commit, Tree

app = typer.Typer()

#: Entries printed before truncating. A node can hold thousands, and a listing
#: that scrolls off the screen is worse than one that says how much it hid.
_MAX_LISTED = 40


def _note_truncation(total: int) -> None:
    if total > _MAX_LISTED:
        console.print(f"[dim]  … {total - _MAX_LISTED} more entries[/dim]")


@app.command("hash-object")
def hash_object(
    path: Annotated[Path, typer.Argument(help="File whose bytes become a chunk object.")],
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print only the name.")] = False,
) -> None:
    """Print the object name for a file's bytes, encoded as a single chunk.

    Note this is the name of a *chunk*, not of the file: a real file is split by
    content-defined chunking and named as a blob over those chunks. Use this to
    compare hashing between two implementations, not to predict a file's name.
    """
    data = path.read_bytes()
    framed = encode(Chunk(data))
    name = name_of_encoded(framed)

    if quiet:
        typer.echo(str(name))
        return
    console.print(f"[green]{name}[/green]")
    console.print("[dim]kind[/dim]    chunk")
    console.print(f"[dim]payload[/dim] {len(data)} bytes")
    console.print(f"[dim]stored[/dim]  {len(framed)} bytes  [dim](2-byte frame)[/dim]")


@app.command("cat-object")
def cat_object(
    path: Annotated[Path, typer.Argument(help="File containing one encoded object.")],
) -> None:
    """Decode an encoded object and describe it.

    Decoding is strict, so this doubles as a canonicality check: anything it
    prints is, by construction, the one valid encoding of what it means.
    """
    framed = path.read_bytes()
    kind = peek_kind(framed)
    obj = decode(framed)
    name = name_of_encoded(framed)

    console.print(f"[green]{name}[/green]")
    console.print(f"[dim]kind[/dim]  {kind.name.lower()}")

    match obj:
        case Chunk():
            console.print(f"[dim]size[/dim]  {obj.size} bytes")
        case Blob():
            role = "leaf" if obj.is_leaf else f"index node (level {obj.level})"
            console.print(f"[dim]shape[/dim] {role}, {len(obj.entries)} entries")
            console.print(f"[dim]covers[/dim] {obj.size} bytes")
            for blob_entry in obj.entries[:_MAX_LISTED]:
                console.print(f"  {blob_entry.target}  {blob_entry.size}")
            _note_truncation(len(obj.entries))
        case Tree():
            role = "leaf" if obj.is_leaf else f"interior (level {obj.level})"
            console.print(f"[dim]shape[/dim] {role}, {len(obj.entries)} entries")
            for tree_entry in obj.entries[:_MAX_LISTED]:
                mode = f"{tree_entry.mode:o}" if tree_entry.mode else "-"
                console.print(
                    f"  {tree_entry.kind.name.lower():8} {mode:>6} {tree_entry.size:>12}  "
                    f"{tree_entry.name.decode(errors='replace')}"
                )
            _note_truncation(len(obj.entries))
        case Commit():
            console.print(f"[dim]tree[/dim]   {obj.tree}")
            for parent in obj.parents:
                console.print(f"[dim]parent[/dim] {parent}")
            console.print(f"[dim]change[/dim] {obj.change_id}")
            console.print(f"[dim]author[/dim] {obj.author}")
            console.print(f"[dim]when[/dim]   {obj.timestamp_us}")
            for key, value in obj.metadata:
                console.print(f"[dim]meta[/dim]   {key} = {value}")
            console.print(f"\n{obj.message}")
