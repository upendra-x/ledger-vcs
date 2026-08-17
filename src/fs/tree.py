"""Reading directories: lookup, path resolution, and paginated listing.

Three things are promised, and each falls out of the shape built in
``format.shape``:

* **O(depth) lookup.** An interior entry's name is the *last* key in the subtree
  it points at, so finding the child that could contain a key is a search over
  the node's own entries — no child is fetched to discover what it holds.

* **No cap on how many files a directory can list.** A 134-million-entry
  directory pages exactly like a six-entry one, because paging is a descent to a
  position rather than an offset into a flat list.

* **A cursor that means something.** It is simply "the last name you saw".
  Ordering by name rather than by hash is what makes that possible — a
  hash-sharded trie has the same asymptotics but its cursor is arbitrary, and a
  caller cannot resume from a name it recognises.

Immutability adds a property mutable systems cannot offer: **the listing is
stable**. Entries cannot shift, appear twice or vanish between pages, because
the tree being listed cannot change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from src.errors import NotFound
from src.format.constants import EntryKind
from src.format.model import Tree
from src.fs.path import parse_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.format.model import TreeEntry
    from src.ids import ObjectName
    from src.store.cas import ObjectStore

__all__ = ["DirectoryPage", "ResolvedEntry", "iter_entries", "list_dir", "lookup", "resolve_path"]


@final
@dataclass(frozen=True, slots=True)
class ResolvedEntry:
    """A path, resolved to the entry that names its content."""

    path: tuple[bytes, ...]
    entry: TreeEntry

    @property
    def target(self) -> ObjectName:
        return self.entry.target

    @property
    def kind(self) -> EntryKind:
        return self.entry.kind


@final
@dataclass(frozen=True, slots=True)
class DirectoryPage:
    """One page of a listing, plus where to resume."""

    entries: tuple[TreeEntry, ...]
    #: The last name returned, or ``None`` when the listing is exhausted. Opaque
    #: to callers by contract, though it is deliberately human-readable so a
    #: paginated API is debuggable.
    cursor: bytes | None

    @property
    def exhausted(self) -> bool:
        return self.cursor is None


def lookup(store: ObjectStore, tree: ObjectName, name: bytes) -> TreeEntry | None:
    """Find one entry by name, descending only the nodes that could hold it.

    Costs one fetch per level: a flat directory is one fetch, a split one is
    two or three.
    """
    node = store.get_as(tree, Tree)
    while node.level > 0:
        child = _child_covering(node, name)
        if child is None:
            return None
        node = store.get_as(child.target, Tree)

    return _find_in_leaf(node, name)


def _child_covering(node: Tree, name: bytes) -> TreeEntry | None:
    """The interior entry whose subtree could contain ``name``.

    Interior entries carry the *last* key of their subtree, so the first entry
    whose key is >= the target is the only one that can hold it. Linear here
    because a node holds at most a couple of thousand entries and the fetch
    dominates; a bisect is a drop-in if that ever stops being true.
    """
    for entry in node.entries:
        if name <= entry.name:
            return entry
    return None


def _find_in_leaf(node: Tree, name: bytes) -> TreeEntry | None:
    for entry in node.entries:
        if entry.name == name:
            return entry
        if entry.name > name:
            break  # entries are sorted, so it cannot appear later
    return None


def resolve_path(store: ObjectStore, root_tree: ObjectName, path: str | bytes) -> ResolvedEntry:
    """Resolve a path within a tree, one fetch per component.

    Raises ``NotFound`` rather than returning ``None``: a caller asking for a
    specific path has already decided it should exist, and the alternative is an
    ``Optional`` threaded through every read endpoint.
    """
    components = parse_path(path)
    if not components:
        raise NotFound("the root of a tree is not an entry; list it instead")

    current = root_tree
    for depth, component in enumerate(components):
        entry = lookup(store, current, component)
        if entry is None:
            raise NotFound(
                "path does not exist",
                path=_render(components),
                missing_at=_render(components[: depth + 1]),
            )
        if depth == len(components) - 1:
            return ResolvedEntry(path=components, entry=entry)
        if entry.kind is not EntryKind.TREE:
            raise NotFound(
                "path traverses a non-directory",
                path=_render(components),
                not_a_directory=_render(components[: depth + 1]),
            )
        current = entry.target

    raise AssertionError("unreachable")  # pragma: no cover


def list_dir(
    store: ObjectStore,
    tree: ObjectName,
    *,
    after: bytes | None = None,
    limit: int = 1000,
) -> DirectoryPage:
    """One page of a directory, in name order, resuming after ``after``.

    The descent is by *name* rather than by index, which is what makes paging
    cost O(depth) regardless of how far into a huge directory the cursor sits.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    collected: list[TreeEntry] = []
    for entry in iter_entries(store, tree, after=after):
        collected.append(entry)
        if len(collected) == limit:
            # Only report a cursor if there is genuinely more. Probing one entry
            # ahead avoids handing back a cursor that yields an empty page,
            # which is the classic off-by-one in cursor pagination.
            more = next(iter_entries(store, tree, after=entry.name), None)
            return DirectoryPage(tuple(collected), entry.name if more else None)

    return DirectoryPage(tuple(collected), None)


def iter_entries(
    store: ObjectStore, tree: ObjectName, *, after: bytes | None = None
) -> Iterator[TreeEntry]:
    """Every entry in name order, streaming.

    Used by listing, diff and the closure walk. Generating rather than
    materialising matters: a two-million-entry directory must be traversable
    without holding two million entries in memory.
    """
    node = store.get_as(tree, Tree)
    if node.level == 0:
        for entry in node.entries:
            if after is None or entry.name > after:
                yield entry
        return

    for child in node.entries:
        # An interior entry's name is the last key in its subtree, so a subtree
        # entirely at or before the cursor can be skipped without fetching it.
        if after is not None and child.name <= after:
            continue
        yield from iter_entries(store, child.target, after=after)


def _render(components: Sequence[bytes]) -> str:
    return "/".join(c.decode(errors="replace") for c in components)
