"""The four object types.

    commit  ──▶  tree  ──▶  tree  ──▶  blob  ──▶  chunk
    a version   a directory  a directory  a file   bytes

They stack in one direction only, and that is the whole shape of the model.

Interior nodes are not a fifth type. A blob index node is a ``Blob`` with
``level > 0`` whose entries point at other ``Blob`` nodes; an interior directory
node is a ``Tree`` with ``level > 0`` whose entries point at other ``Tree``
nodes. There are "exactly four types", and then index
and interior nodes; the ``level`` field is what reconciles them without adding a
tag, and it gives depth validation for free.

These are data, not behaviour. Building them is ``format.shape``; reading them
is ``fs.blob`` and ``fs.tree``; naming them is ``format.codec``. Keeping the
types inert is what lets every one of those be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final

from src.format.constants import EntryKind, ObjectKind

if TYPE_CHECKING:
    from src.ids import ChangeId, ObjectName

__all__ = [
    "Blob",
    "BlobEntry",
    "Chunk",
    "Commit",
    "LedgerObject",
    "Tree",
    "TreeEntry",
]


@final
@dataclass(frozen=True, slots=True)
class Chunk:
    """A slice of a file's bytes — the leaf of the whole model, and the unit of
    deduplication and transfer.

    Boundaries are content-defined, so an insertion shifts only the
    chunk it landed in rather than every chunk after it.
    """

    data: bytes

    KIND = ObjectKind.CHUNK

    @property
    def size(self) -> int:
        return len(self.data)


@final
@dataclass(frozen=True, slots=True)
class BlobEntry:
    """One child of a blob node, with the byte span it covers.

    Carrying ``size`` is what makes a ranged read cheap: descending to the child
    covering byte *X* is a search over cumulative sizes, so no child has to be
    fetched to discover how much of the file it holds.
    """

    target: ObjectName
    #: Bytes covered by this child. At level 0 that is a chunk's length; above,
    #: it is the total covered by that subtree.
    size: int


@final
@dataclass(frozen=True, slots=True)
class Blob:
    """One file: an ordered list of chunk references, or of index nodes.

    A 1 TiB file at 1 MiB chunks is a million references. Holding them in one
    object would make that object enormous, and "no object is ever large" is what
    keeps every other property true — so the list splits into index nodes and
    the structure grows in depth rather than in width.
    """

    level: int
    entries: tuple[BlobEntry, ...]

    KIND = ObjectKind.BLOB

    @property
    def size(self) -> int:
        """Total bytes this node covers.

        Derived rather than stored. A stored total would be a second way to
        express the same fact, and two encoders disagreeing about a redundant
        field is precisely how one logical object acquires two names.
        """
        return sum(e.size for e in self.entries)

    @property
    def is_leaf(self) -> bool:
        return self.level == 0


@final
@dataclass(frozen=True, slots=True)
class TreeEntry:
    """One entry in a directory — or, in an interior node, one routing key.

    In a leaf node (``Tree.level == 0``) this is a real directory entry and
    ``name`` is the path component. In an interior node it is a separator:
    ``name`` is the *last* entry name in the subtree ``target`` covers, which is
    what lets a lookup pick a child without fetching any of them.

    One struct serves both because the shape is identical and the node's
    ``level`` already says which reading applies. A second struct would be a
    second thing to keep canonical.
    """

    #: The path component, as raw bytes. Sorted and compared by unsigned byte
    #: order — UTF-8 is order-preserving, so an implementation that sorts
    #: strings and one that sorts bytes cannot disagree.
    name: bytes
    kind: EntryKind
    target: ObjectName
    #: ``0o644`` or ``0o755`` for a file; ``0`` for anything else. Git's
    #: precedent: beyond "is it executable", permissions are neither portable
    #: nor worth putting inside a hash.
    mode: int
    #: Content bytes for a file or symlink; ``0`` for a subtree.
    #:
    #: Deliberately *not* a recursive directory total. That would be a derived
    #: value which cannot be validated without fetching children, so a decoder
    #: could not tell a canonical encoding from a wrong one — and nothing in the
    #: requirements needs it.
    size: int


@final
@dataclass(frozen=True, slots=True)
class Tree:
    """One directory, as a name-ordered node that splits past a threshold.

    The split is content-defined on the entry name rather than positional
    (see ``format.shape``): filling nodes left to right would make
    inserting one entry near the start rewrite every later node, which would
    take diff pruning, merge pruning and cross-version deduplication with it.
    """

    level: int
    entries: tuple[TreeEntry, ...]

    KIND = ObjectKind.TREE

    @property
    def is_leaf(self) -> bool:
        return self.level == 0


@final
@dataclass(frozen=True, slots=True)
class Commit:
    """A complete, self-contained snapshot of one environment.

    It does not describe a change; it names the whole tree. Because the content
    underneath is deduplicated, a snapshot costs what a delta would while a
    delta chain would make reading an old version cost the length of its chain —
    which is why history, diff and revert are all ordinary reads.

    Note what is absent: **a commit does not name its environment.** That
    omission is what lets a fork share every ancestor commit object with its
    parent instead of copying them.
    """

    tree: ObjectName
    #: Zero for a root, one normally, two or more for a merge.
    parents: tuple[ObjectName, ...]
    #: Stable across amendment and rebase, unlike the commit's own name.
    change_id: ChangeId
    author: str
    committer: str
    timestamp_us: int
    message: str
    #: Opaque key/value pairs — rollout id, QA verdict, the original git SHA on
    #: import. Ledger never interprets these.
    metadata: tuple[tuple[str, str], ...] = field(default=())

    KIND = ObjectKind.COMMIT

    @property
    def is_merge(self) -> bool:
        return len(self.parents) >= 2


#: Anything the store can hold. Every one of these is immutable, and its name is
#: the hash of its own canonical encoding.
type LedgerObject = Chunk | Blob | Tree | Commit
