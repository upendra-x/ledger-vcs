"""Every object reachable from a tree or a commit.

This is the walk that decides what a write session protects and what a fork's
keep-set holds, so getting it wrong is not a performance problem — it is silent
data loss on the next sweep.

**Chunk digests are recorded without the chunks being fetched.** The rule
Garbage collection says the walk never touches chunks, and that means never
*read* them: the cost is proportional to the environment's *shape* rather than
its size. Reading it the other way — omitting chunk names from the result — would
build a metadata-only protection set, and the first sweep would take the content
the commit depends on while leaving the trees that point at it.

It lives here, below the services, because three callers need it — publishing a
commit, forking an environment, and importing a git history — and two of them
having their own copy is how the two copies drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.format.constants import EntryKind
from src.format.model import Blob, Commit, Tree

if TYPE_CHECKING:
    from src.ids import ObjectName
    from src.store.cas import ObjectStore

__all__ = ["commit_closure", "tree_closure"]

#: How a name reached us, which is the only thing that says how to read it: an
#: object's bytes do not announce their own type before they are fetched, and
#: fetching to find out is exactly what this walk avoids for chunks.
type _Kind = Literal["tree", "blob", "chunk"]


def tree_closure(store: ObjectStore, root: ObjectName) -> list[ObjectName]:
    """Every object the tree reaches, the root tree itself included."""
    return _walk(store, root, seed=())


def commit_closure(store: ObjectStore, commit: ObjectName) -> list[ObjectName]:
    """Every object the commit reaches, the commit object itself included.

    The commit is seeded rather than walked: it is not part of its own tree, but
    it is part of what a keep-set has to protect.
    """
    return _walk(store, store.get_as(commit, Commit).tree, seed=(commit,))


def _walk(
    store: ObjectStore,
    root: ObjectName,
    *,
    seed: tuple[ObjectName, ...],
) -> list[ObjectName]:
    """Depth-first from ``root``, with ``seed`` already counted as found.

    ``seed`` must never contain ``root`` — anything already in ``seen`` is
    skipped, so seeding the root would return an empty walk.
    """
    found: list[ObjectName] = list(seed)
    seen: set[ObjectName] = set(seed)
    stack: list[tuple[ObjectName, _Kind]] = [(root, "tree")]

    while stack:
        name, kind = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        found.append(name)

        if kind == "chunk":
            continue  # recorded, never fetched

        if kind == "tree":
            node = store.get_as(name, Tree)
            for entry in node.entries:
                # An interior tree node's children are more tree nodes whatever
                # their entry kind claims; only at level 0 does the kind decide.
                child: _Kind = (
                    "tree" if (node.level > 0 or entry.kind is EntryKind.TREE) else "blob"
                )
                stack.append((entry.target, child))
            continue

        blob = store.get_as(name, Blob)
        below: _Kind = "blob" if blob.level > 0 else "chunk"
        stack.extend((entry.target, below) for entry in blob.entries)

    return found
