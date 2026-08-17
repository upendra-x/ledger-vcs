"""Comparing two versions.

**Diff cost is proportional to what changed, not to the size of the
environment**, and it falls out of content addressing rather than
from any cleverness here: an unchanged subtree has an unchanged hash, so an
entire branch is dismissed by comparing two 32-byte names.

    three-way tree merge, decided by hash:
      task/     base ≠ a1,  base = b1     →  take a1
      data/     base = a1,  base ≠ b1     →  take b1
      images/   base = a1 = b1            →  identical hash, skip the subtree
                                              ↑ 3.2 GiB compared in one comparison

Two versions of a 43 GiB environment differing in one file examine one path.

Diffs are produced as an iterator rather than a list. A two-million-entry change
must not have to fit in one response, and the API pages it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, final

from src.format.constants import EntryKind
from src.fs.tree import iter_entries

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.format.model import TreeEntry
    from src.ids import ObjectName
    from src.store.cas import ObjectStore

__all__ = ["Change", "ChangeKind", "diff_trees", "format_path"]


class ChangeKind(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@final
@dataclass(frozen=True, slots=True)
class Change:
    """One differing path."""

    path: tuple[bytes, ...]
    kind: ChangeKind
    before: TreeEntry | None
    after: TreeEntry | None

    @property
    def display_path(self) -> str:
        return format_path(self.path)

    @property
    def size_delta(self) -> int:
        after = self.after.size if self.after else 0
        before = self.before.size if self.before else 0
        return after - before


def format_path(path: Sequence[bytes]) -> str:
    return "/".join(component.decode(errors="replace") for component in path)


def diff_trees(
    store: ObjectStore,
    before: ObjectName | None,
    after: ObjectName | None,
    *,
    prefix: tuple[bytes, ...] = (),
) -> Iterator[Change]:
    """Yield every path that differs between two trees.

    Either side may be ``None``, which is how an added or removed directory is
    expressed without a special case at each call site.
    """
    if before == after:
        # The whole point. Two identical subtrees have one name, so an entire
        # branch — however large — is dismissed here.
        return

    left = list(iter_entries(store, before)) if before is not None else []
    right = list(iter_entries(store, after)) if after is not None else []

    yield from _merge_join(store, left, right, prefix)


def _merge_join(
    store: ObjectStore,
    left: list[TreeEntry],
    right: list[TreeEntry],
    prefix: tuple[bytes, ...],
) -> Iterator[Change]:
    """Walk two name-ordered entry lists together.

    Both sides are sorted by name, so this is a single linear pass with no
    lookups — and each matching pair is decided by comparing two names.
    """
    index_left = index_right = 0

    while index_left < len(left) or index_right < len(right):
        entry_left = left[index_left] if index_left < len(left) else None
        entry_right = right[index_right] if index_right < len(right) else None

        if entry_right is None or (entry_left is not None and entry_left.name < entry_right.name):
            assert entry_left is not None
            yield from _removed(store, entry_left, prefix)
            index_left += 1
        elif entry_left is None or entry_right.name < entry_left.name:
            yield from _added(store, entry_right, prefix)
            index_right += 1
        else:
            yield from _compare(store, entry_left, entry_right, prefix)
            index_left += 1
            index_right += 1


def _compare(
    store: ObjectStore, before: TreeEntry, after: TreeEntry, prefix: tuple[bytes, ...]
) -> Iterator[Change]:
    if before.target == after.target and before.mode == after.mode and before.kind is after.kind:
        return  # identical, including its whole subtree if it is one

    path = (*prefix, before.name)
    if before.kind is EntryKind.TREE and after.kind is EntryKind.TREE:
        yield from diff_trees(store, before.target, after.target, prefix=path)
        return

    # A file replaced by a directory, or vice versa, is a removal and an
    # addition rather than a modification — nothing about the content relates.
    if before.kind is not after.kind:
        yield from _removed(store, before, prefix)
        yield from _added(store, after, prefix)
        return

    yield Change(path=path, kind=ChangeKind.MODIFIED, before=before, after=after)


def _added(store: ObjectStore, entry: TreeEntry, prefix: tuple[bytes, ...]) -> Iterator[Change]:
    path = (*prefix, entry.name)
    if entry.kind is EntryKind.TREE:
        yield from diff_trees(store, None, entry.target, prefix=path)
        return
    yield Change(path=path, kind=ChangeKind.ADDED, before=None, after=entry)


def _removed(store: ObjectStore, entry: TreeEntry, prefix: tuple[bytes, ...]) -> Iterator[Change]:
    path = (*prefix, entry.name)
    if entry.kind is EntryKind.TREE:
        yield from diff_trees(store, entry.target, None, prefix=path)
        return
    yield Change(path=path, kind=ChangeKind.REMOVED, before=entry, after=None)
