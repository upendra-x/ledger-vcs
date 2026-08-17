"""Canonical shape: where a tree or a blob splits into nodes.

A wide directory becomes a B-tree ordered by entry name that
splits past a fanout threshold, and a long chunk list becomes index nodes. It
does not say *where* the split points go, and that choice decides whether its own
cost table is true.

**Why not fixed fanout.** Filling nodes left to right at 512 entries each is
deterministic and trivial. It is also wrong here, because these structures are
immutable and content-addressed: inserting one entry near the start shifts every
later boundary, so every later node gets new contents and therefore a new name.
A one-file change in a wide directory would re-store the whole directory, and
``Diff``'s subtree pruning, ``merge``'s subtree pruning and cross-version
deduplication would all stop working at the same time.

**Content-defined splitting on the key.** A boundary falls after a key whose
hash matches, exactly as chunking cuts where the content hash matches — the same
trick, applied to keys instead of bytes. The result is:

    deterministic   the shape is a pure function of the sorted entry set, so two
                    writers building the same directory produce the same name

    stable          an insertion changes the node containing it and its ancestors,
                    and nothing else — boundaries elsewhere re-sync immediately

This structure is sometimes called a *prolly tree* (probabilistic B-tree); Dolt
uses it for the same reason.

**The level salt is mandatory, and its absence fails silently.** At level *L*>0
every key is, by construction, a key that was already a boundary at level *L*−1.
Without a level in the hash input, the predicate that selected it fires again —
on *every* key — so the tree degenerates into fixed positional grouping. It does
not look broken: ``min_entries`` clamps it back into plausible-looking nodes, and
only the incremental-update cost gives it away.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final

from blake3 import blake3

from src.format.codec import encode, name_of_encoded
from src.format.constants import (
    MAX_NODE_BYTES,
    SPLIT_DOMAIN,
    SPLIT_MAX_ENTRIES,
    SPLIT_MIN_ENTRIES,
    SPLIT_PERIOD,
    EntryKind,
)
from src.format.model import Blob, BlobEntry, Tree, TreeEntry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from src.ids import ObjectName

__all__ = [
    "PRODUCTION_SHAPE",
    "Emit",
    "ShapeParams",
    "build_blob",
    "build_tree",
    "is_boundary",
    "partition",
]

#: Called once for every interior node the builder creates, with the node's name
#: and its canonical encoding. Returning nothing keeps the split between
#: *deciding* names (here) and *storing* bytes (the caller) explicit — this
#: module never learns what a store is.
#:
#: The encoded bytes are handed over rather than the object because producing the
#: name already required them: every caller stores what it is given, and passing
#: the object instead made each of them encode it a second time.
type Emit = Callable[[ObjectName, bytes], None]

#: An interior node must have at least two children, or it is a redundant
#: spelling of its only child and the codec rejects it.
_MIN_INTERIOR_CHILDREN: Final = 2

#: The level is salted in as one byte, so depth is bounded at 256. At the
#: production period that would address more entries than there are atoms worth
#: addressing; the check exists so the failure is loud rather than a wrapped byte.
_MAX_LEVEL: Final = 255


@final
@dataclass(frozen=True, slots=True)
class ShapeParams:
    """Everything that decides node boundaries.

    Injected for the same reason as ``ChunkParams``: production splits at ~448
    entries, and a test that needed 448 entries to observe a second level would
    be slow enough that nobody runs it.
    """

    domain: bytes
    period: int
    min_entries: int
    max_entries: int
    max_node_bytes: int

    def __post_init__(self) -> None:
        if not 1 <= self.min_entries < self.max_entries:
            raise ValueError("need 1 <= min_entries < max_entries")
        if self.period < 2:
            raise ValueError("period must be at least 2")
        if self.max_node_bytes < 64:
            raise ValueError("max_node_bytes is implausibly small")


#: The frozen production shape. Any change renames every wide tree in the corpus.
PRODUCTION_SHAPE: Final = ShapeParams(
    domain=SPLIT_DOMAIN,
    period=SPLIT_PERIOD,
    min_entries=SPLIT_MIN_ENTRIES,
    max_entries=SPLIT_MAX_ENTRIES,
    max_node_bytes=MAX_NODE_BYTES,
)


def is_boundary(key: bytes, level: int, params: ShapeParams) -> bool:
    """Whether a node boundary falls immediately after ``key`` at ``level``."""
    if not 0 <= level <= _MAX_LEVEL:
        raise ValueError(f"level out of range: {level}")
    digest = blake3(params.domain + bytes([level]) + key).digest(length=8)
    return int.from_bytes(digest, "big") % params.period == 0


def partition(
    keys: Sequence[bytes],
    entry_bytes: Sequence[int],
    level: int,
    params: ShapeParams,
) -> list[int]:
    """Split a run of keys into nodes. Returns each node's exclusive end index.

    Three things can end a node, in this order of precedence:

    * the encoded node reaching ``max_node_bytes`` — a hard ceiling, because
      "no object is ever large" is what keeps every other property true;
    * reaching ``max_entries``;
    * the content-defined predicate firing, once ``min_entries`` is satisfied.

    The clamps are what stop the geometric distribution from producing
    single-entry nodes (which would blow up depth) or unbounded ones.
    """
    if len(keys) != len(entry_bytes):
        raise ValueError("keys and entry_bytes must be the same length")

    ends: list[int] = []
    start = 0
    node_bytes = 0

    for index, key in enumerate(keys):
        node_bytes += entry_bytes[index]
        count = index - start + 1

        over_byte_budget = node_bytes >= params.max_node_bytes
        if count < params.min_entries and not over_byte_budget:
            continue

        if over_byte_budget or count >= params.max_entries or is_boundary(key, level, params):
            ends.append(index + 1)
            start = index + 1
            node_bytes = 0

    if start < len(keys):
        ends.append(len(keys))

    if level > 0:
        _ensure_no_lonely_tail(ends)
    return ends


def _ensure_no_lonely_tail(ends: list[int]) -> None:
    """Make sure no interior node ends up with a single child.

    Every node except the last is at least ``min_entries`` long, but the tail
    gets whatever is left — which can be one entry. At level 0 that is a
    perfectly good leaf; above it, the codec rejects it as a redundant spelling
    of its only child, so this has to be resolved before the node is built.

    **Preferred: move the boundary back.** The previous node gives up its last
    child, which costs nothing — the node only shrinks, so no cap it was already
    checked against can be exceeded.

    **Fallback: merge the two runs.** Reached only when the previous node cannot
    spare a child, and that case is tightly bounded. The tail is exactly one
    (a zero-length run is never produced), so exactly one child is needed; the
    previous node can only refuse if it holds two or fewer. The merged node
    therefore holds **at most three entries**.

    That merged node can, in principle, sit above ``max_node_bytes`` — if the
    previous node already reached the byte budget with two very large entries.
    It is worth being exact about why that is accepted rather than fixed: the
    only alternative is emitting an interior node with one child, which the codec
    refuses outright, so the choice is between a node marginally over a *budget*
    and a tree that cannot be encoded at all. ``max_node_bytes`` bounds the cost
    of one fetch; it is not an invariant anything depends on for correctness.

    Both branches are pure functions of the input, so determinism — and therefore
    every object name — is untouched.
    """
    if len(ends) < 2:
        return

    tail = ends[-1] - ends[-2]
    if tail >= _MIN_INTERIOR_CHILDREN:
        return

    previous_start = ends[-3] if len(ends) >= 3 else 0
    borrowable = (ends[-2] - previous_start) - _MIN_INTERIOR_CHILDREN
    needed = _MIN_INTERIOR_CHILDREN - tail

    if borrowable >= needed:
        ends[-2] -= needed
    else:
        del ends[-2]


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────
#
# Both are bottom-up: partition the entries into leaf nodes, emit them, then
# treat those nodes as the entries of the level above and repeat until one node
# remains. Depth therefore grows logarithmically and is never chosen — it falls
# out of how many entries there are.


def build_blob(
    chunks: Sequence[BlobEntry],
    emit: Emit,
    params: ShapeParams = PRODUCTION_SHAPE,
) -> ObjectName:
    """Build the blob node tree for one file, returning its root name.

    ``chunks`` are the file's chunk references in order. The *chunks themselves*
    are the caller's to store — this only builds the manifest and index nodes
    above them, which for a 1 TiB file is a few thousand small objects rather
    than a million.

    Index-node boundaries are content-defined on the child's *name*, using the
    same partitioner as trees. Fixed fanout would mean a same-length overwrite
    that changes the chunk count rewrites every subsequent index node.
    """
    if not chunks:
        return _emit(Blob(level=0, entries=()), emit)

    entries = list(chunks)
    level = 0
    while True:
        ends = partition(
            [e.target.digest for e in entries],
            [_BLOB_ENTRY_BYTES] * len(entries),
            level,
            params,
        )
        nodes: list[BlobEntry] = []
        start = 0
        for end in ends:
            run = entries[start:end]
            name = _emit(Blob(level=level, entries=tuple(run)), emit)
            nodes.append(BlobEntry(target=name, size=sum(e.size for e in run)))
            start = end

        if len(nodes) == 1:
            return nodes[0].target
        entries = nodes
        level += 1


def build_tree(
    entries: Sequence[TreeEntry],
    emit: Emit,
    params: ShapeParams = PRODUCTION_SHAPE,
) -> ObjectName:
    """Build the node tree for one directory, returning its root name.

    ``entries`` must already be sorted by name and unique — the codec enforces
    both, and sorting here would hide a caller that built its listing wrong.

    Above level 0 an entry is a routing key: its ``name`` is the last key in the
    subtree it points at, which is what lets a lookup choose a child without
    fetching any of them.
    """
    if not entries:
        return _emit(Tree(level=0, entries=()), emit)

    _reject_unsorted(entries)

    current = list(entries)
    level = 0
    while True:
        ends = partition(
            [e.name for e in current],
            [_tree_entry_bytes(e) for e in current],
            level,
            params,
        )
        nodes: list[TreeEntry] = []
        start = 0
        for end in ends:
            run = current[start:end]
            name = _emit(Tree(level=level, entries=tuple(run)), emit)
            nodes.append(
                TreeEntry(
                    name=run[-1].name,  # the last key this subtree covers
                    kind=EntryKind.TREE,
                    target=name,
                    mode=0,
                    size=0,
                )
            )
            start = end

        if len(nodes) == 1:
            return nodes[0].target
        current = nodes
        level += 1


def _reject_unsorted(entries: Sequence[TreeEntry]) -> None:
    for previous, current in itertools.pairwise(entries):
        if current.name <= previous.name:
            raise ValueError(
                "tree entries must be sorted and unique before building; sorting "
                "here would mask a caller that assembled its listing wrong"
            )


def _emit(node: Blob | Tree, emit: Emit) -> ObjectName:
    framed = encode(node)
    name = name_of_encoded(framed)
    emit(name, framed)
    return name


#: 32-byte child name plus a u64 size. Fixed, so a blob level's byte budget is
#: exactly proportional to its entry count.
_BLOB_ENTRY_BYTES: Final = 40


def _tree_entry_bytes(entry: TreeEntry) -> int:
    """Encoded size of one tree entry: name length byte, name, kind, target,
    mode, size.

    Computed rather than measured, because measuring would mean encoding the
    node once per candidate boundary.
    """
    return 1 + len(entry.name) + 1 + 32 + 2 + 8
