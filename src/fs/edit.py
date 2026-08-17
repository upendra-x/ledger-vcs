"""Editing a tree: put one entry at one path, and get a new tree.

Every version is a full snapshot, so "changing a file" is really "building the
tree that differs from the old one at one path". The cost of that is the cost of
the *path*, not of the environment: nodes off the path keep their names, which is
the same property that makes diff and merge prune, stated from the write side.

Trees are immutable, so nothing here mutates anything — the old tree is still
there and still readable, which is why an edit can be published with a
compare-and-swap rather than a lock.

Deliberately narrow. This is not a general filesystem API: it is the one
operation a server-side writer needs when it has content that did not come from
a directory on disk — an imported image, a converted commit, a generated file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, final

from src.errors import InvalidRequest
from src.format.constants import EntryKind
from src.format.model import TreeEntry
from src.format.shape import PRODUCTION_SHAPE, build_tree
from src.fs.path import parse_path
from src.fs.tree import iter_entries

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.format.shape import Emit, ShapeParams
    from src.ids import ObjectName
    from src.store.cas import ObjectStore, PutOutcome

__all__ = ["EditResult", "empty_tree", "remove_path", "set_path"]


@final
@dataclass(frozen=True, slots=True)
class EditResult:
    tree: ObjectName
    #: Every node the edit wrote, with whether it was new. Outcomes rather than
    #: bare names so a caller can both record them against its write session and
    #: report what the edit actually cost — a node that already existed is the
    #: normal case and should not be counted as storage.
    written: tuple[PutOutcome, ...]

    @property
    def names(self) -> tuple[ObjectName, ...]:
        return tuple(outcome.name for outcome in self.written)


def empty_tree(store: ObjectStore, *, shape: ShapeParams = PRODUCTION_SHAPE) -> ObjectName:
    """The name of the empty directory — the root of an environment with nothing in it."""
    written: list[PutOutcome] = []
    return build_tree((), _emitter(store, written), shape)


def set_path(
    store: ObjectStore,
    root: ObjectName,
    path: str | bytes,
    entry: TreeEntry,
    *,
    shape: ShapeParams = PRODUCTION_SHAPE,
) -> EditResult:
    """Return the tree that is ``root`` with ``path`` set to ``entry``.

    Intermediate directories are created as needed. The entry's ``name`` is taken
    from the path rather than from the entry, so the two cannot disagree — a
    mismatch there would produce a tree whose lookup finds nothing at the path it
    was written to.
    """
    return _edit(store, root, parse_path(path), entry, shape)


def remove_path(
    store: ObjectStore,
    root: ObjectName,
    path: str | bytes,
    *,
    shape: ShapeParams = PRODUCTION_SHAPE,
) -> EditResult:
    """Return the tree that is ``root`` without ``path``. Absent is not an error."""
    return _edit(store, root, parse_path(path), None, shape)


def _edit(
    store: ObjectStore,
    root: ObjectName,
    components: Sequence[bytes],
    entry: TreeEntry | None,
    shape: ShapeParams,
) -> EditResult:
    if not components:
        raise InvalidRequest("the root of a tree is not an entry; it cannot be set or removed")
    written: list[PutOutcome] = []
    tree = _apply(store, root, components, entry, shape, written)
    return EditResult(tree=tree, written=tuple(written))


def _apply(
    store: ObjectStore,
    tree: ObjectName,
    components: Sequence[bytes],
    entry: TreeEntry | None,
    shape: ShapeParams,
    written: list[PutOutcome],
) -> ObjectName:
    name = components[0]
    # The whole directory is read and rebuilt, which is O(entries) in *this*
    # directory and nothing else — sibling directories keep their names and are
    # never fetched. A million-entry directory would want an incremental
    # rebuild; nothing in the requirements produces one.
    existing = {child.name: child for child in iter_entries(store, tree)}

    if len(components) == 1:
        if entry is None:
            if existing.pop(name, None) is None:
                return tree  # nothing to remove, so nothing to rewrite
        else:
            replacement = replace(entry, name=name)
            if existing.get(name) == replacement:
                return tree  # already exactly this, so the tree is unchanged
            existing[name] = replacement
    else:
        child = existing.get(name)
        if child is not None and child.kind is not EntryKind.TREE:
            raise InvalidRequest("path traverses a file", path=name.decode(errors="replace"))
        if child is None:
            if entry is None:
                return tree  # removing from a directory that does not exist
            subtree = build_tree((), _emitter(store, written), shape)
        else:
            subtree = child.target

        rebuilt = _apply(store, subtree, components[1:], entry, shape, written)
        if child is not None and rebuilt == child.target:
            return tree
        existing[name] = TreeEntry(name=name, kind=EntryKind.TREE, target=rebuilt, mode=0, size=0)

    ordered = [existing[key] for key in sorted(existing)]
    return build_tree(ordered, _emitter(store, written), shape)


def _emitter(store: ObjectStore, written: list[PutOutcome]) -> Emit:
    def emit(name: ObjectName, framed: bytes) -> None:
        written.append(store.put_encoded(name, framed))

    return emit
