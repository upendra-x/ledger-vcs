"""Canonical shape: determinism, incremental stability, and the level salt.

Two properties matter, and they pull in opposite directions:

* **Determinism** — the shape must be a pure function of the
  entry set, so two writers building the same directory produce the same name.
  Fixed fanout has this trivially.
* **Incremental stability** — inserting
one entry must
  change one node and its ancestors, not everything after it. Fixed fanout fails
  this completely.

Content-defined splitting is the only thing that gets both, and the tests below
are arranged to fail loudly if either is lost.
"""

from __future__ import annotations

import random

import pytest

from src.format import constants as C
from src.format.codec import decode, decode_as, encode
from src.format.model import Blob, BlobEntry, LedgerObject, Tree, TreeEntry
from src.format.shape import (
    PRODUCTION_SHAPE,
    ShapeParams,
    build_blob,
    build_tree,
    is_boundary,
    partition,
)
from src.ids import ObjectName

#: Splits every ~8 entries instead of ~448, so a three-level tree needs hundreds
#: of entries rather than tens of millions.
TEST_SHAPE = ShapeParams(
    domain=b"ledger.tree.split.v1",
    period=8,
    min_entries=2,
    max_entries=32,
    max_node_bytes=4096,
)


class Store:
    """Collects emitted nodes. Stands in for the object store, which this layer
    must never need anyway.
    """

    def __init__(self) -> None:
        self.objects: dict[ObjectName, LedgerObject] = {}

    def emit(self, name: ObjectName, framed: bytes) -> None:
        # The builder hands over the canonical encoding, so decoding here is
        # also a check that what it emitted round-trips to what it named.
        self.objects[name] = decode(framed)

    def __len__(self) -> int:
        return len(self.objects)


def a_name(seed: int) -> ObjectName:
    return ObjectName(random.Random(seed).randbytes(32))


def file_entries(count: int, *, salt: int = 0) -> list[TreeEntry]:
    return [
        TreeEntry(
            # The salt varies the *names*, not just the targets: shape is a
            # function of the keys, so two "trials" with identical names
            # would be one sample repeated.
            name=f"s{salt:03d}_file{i:06d}.json".encode(),
            kind=C.EntryKind.BLOB,
            target=a_name(i + salt * 1_000_000),
            mode=C.MODE_REGULAR,
            size=100 + i,
        )
        for i in range(count)
    ]


def chunk_entries(count: int, *, salt: int = 0) -> list[BlobEntry]:
    return [BlobEntry(target=a_name(i + salt * 1_000_000), size=1000 + i) for i in range(count)]


def walk(store: Store, root: ObjectName) -> dict[ObjectName, LedgerObject]:
    """Every node reachable from a root — the closure a keep-set would record."""
    seen: dict[ObjectName, LedgerObject] = {}
    stack = [root]
    while stack:
        name = stack.pop()
        node = store.objects.get(name)
        if node is None or name in seen:
            continue
        seen[name] = node
        # Only interior nodes point at other nodes; a leaf's targets are chunks
        # or blobs, which this walk deliberately does not follow.
        if isinstance(node, Tree | Blob) and node.level > 0:
            stack.extend(e.target for e in node.entries)
    return seen


def leaf_entries_in_order(store: Store, root: ObjectName) -> list[TreeEntry]:
    """Flatten a built tree back to its logical entry list, in name order."""
    node = decode_as(encode(store.objects[root]), Tree)
    if node.level == 0:
        return list(node.entries)
    out: list[TreeEntry] = []
    for entry in node.entries:
        out.extend(leaf_entries_in_order(store, entry.target))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The predicate
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundaryPredicate:
    def test_is_deterministic(self) -> None:
        assert is_boundary(b"key", 0, TEST_SHAPE) == is_boundary(b"key", 0, TEST_SHAPE)

    def test_the_level_salt_changes_the_answer(self) -> None:
        """The whole point of the salt: a key that is a boundary at one level
        must not automatically be one at the level above.
        """
        keys = [f"k{i}".encode() for i in range(2000)]
        at_level_0 = {k for k in keys if is_boundary(k, 0, TEST_SHAPE)}
        at_level_1 = {k for k in keys if is_boundary(k, 1, TEST_SHAPE)}
        assert at_level_0 != at_level_1
        overlap = len(at_level_0 & at_level_1) / max(len(at_level_0), 1)
        assert overlap < 0.4, f"levels agree on {overlap:.0%} of boundaries — salt is weak"

    def test_fires_at_roughly_the_configured_rate(self) -> None:
        keys = [f"key{i:08d}".encode() for i in range(20_000)]
        hits = sum(1 for k in keys if is_boundary(k, 0, PRODUCTION_SHAPE))
        expected = len(keys) / PRODUCTION_SHAPE.period
        assert expected * 0.7 < hits < expected * 1.3, f"{hits} vs expected ~{expected:.0f}"

    def test_rejects_an_out_of_range_level(self) -> None:
        with pytest.raises(ValueError, match="level out of range"):
            is_boundary(b"k", 256, TEST_SHAPE)


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning
# ─────────────────────────────────────────────────────────────────────────────


class TestPartition:
    def test_covers_every_key_exactly_once(self) -> None:
        keys = [f"k{i:05d}".encode() for i in range(500)]
        ends = partition(keys, [40] * len(keys), 0, TEST_SHAPE)
        assert ends[-1] == len(keys)
        assert ends == sorted(ends)
        assert len(set(ends)) == len(ends)

    def test_respects_the_minimum_except_for_the_tail(self) -> None:
        keys = [f"k{i:05d}".encode() for i in range(500)]
        ends = partition(keys, [40] * len(keys), 0, TEST_SHAPE)
        starts = [0, *ends[:-1]]
        sizes = [e - s for s, e in zip(starts, ends, strict=True)]
        assert all(s >= TEST_SHAPE.min_entries for s in sizes[:-1])

    def test_respects_the_maximum(self) -> None:
        """Uniform keys that never trigger the predicate must still be capped."""
        keys = [b"identical"] * 300
        ends = partition(keys, [40] * len(keys), 0, TEST_SHAPE)
        starts = [0, *ends[:-1]]
        sizes = [e - s for s, e in zip(starts, ends, strict=True)]
        assert all(s <= TEST_SHAPE.max_entries for s in sizes)

    def test_byte_budget_overrides_the_minimum(self) -> None:
        """A handful of very long names must not produce an oversized node, even
        though the entry count is below the minimum.
        """
        keys = [f"k{i}".encode() for i in range(20)]
        ends = partition(keys, [3000] * len(keys), 0, TEST_SHAPE)
        starts = [0, *ends[:-1]]
        sizes = [e - s for s, e in zip(starts, ends, strict=True)]
        assert max(sizes) * 3000 <= TEST_SHAPE.max_node_bytes + 3000

    def test_interior_levels_never_leave_a_lonely_tail(self) -> None:
        """An interior node with one child is rejected by the codec, so the
        partitioner must not produce one at any level above zero.
        """
        for count in range(2, 200):
            keys = [f"k{i:05d}".encode() for i in range(count)]
            ends = partition(keys, [40] * len(keys), 1, TEST_SHAPE)
            starts = [0, *ends[:-1]]
            sizes = [e - s for s, e in zip(starts, ends, strict=True)]
            assert all(s >= 2 for s in sizes), f"count={count} produced {sizes}"

    def test_resolving_a_lonely_tail_costs_at_most_one_extra_entry(self) -> None:
        """The bound the merge fallback rests on, checked rather than argued.

        When the previous node cannot lend a child, the two runs are merged. That
        is safe only because the case is tightly constrained: the tail is exactly
        one and the previous node holds at most two, so the merged node holds at
        most three. If that ever stopped being true the fallback would be
        producing arbitrarily oversized interior nodes, silently.
        """
        for count in range(2, 400):
            keys = [f"k{i:05d}".encode() for i in range(count)]
            ends = partition(keys, [40] * len(keys), 1, TEST_SHAPE)
            starts = [0, *ends[:-1]]
            sizes = [e - s for s, e in zip(starts, ends, strict=True)]
            assert max(sizes) <= max(TEST_SHAPE.max_entries, 3), (
                f"count={count} produced a node of {max(sizes)} entries"
            )

    def test_rejects_mismatched_inputs(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            partition([b"a"], [1, 2], 0, TEST_SHAPE)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_tree_shape_is_insertion_order_independent(self) -> None:
        """The headline naming test for trees.

        The same directory assembled in eleven different orders must produce one
        root name. If shape ever depended on how a caller happened to accumulate
        its entries, two agents committing identical content would write two
        different trees and deduplication would silently stop.
        """
        entries = file_entries(2000)
        roots = set()
        for seed in range(11):
            shuffled = entries[:]
            random.Random(seed).shuffle(shuffled)
            store = Store()
            roots.add(build_tree(sorted(shuffled, key=lambda e: e.name), store.emit, TEST_SHAPE))
        assert len(roots) == 1

    def test_rebuilding_emits_identical_nodes(self) -> None:
        entries = file_entries(1000)
        first, second = Store(), Store()
        root_a = build_tree(entries, first.emit, TEST_SHAPE)
        root_b = build_tree(entries, second.emit, TEST_SHAPE)
        assert root_a == root_b
        assert first.objects.keys() == second.objects.keys()

    def test_blob_shape_is_deterministic(self) -> None:
        chunks = chunk_entries(1000)
        first, second = Store(), Store()
        assert build_blob(chunks, first.emit, TEST_SHAPE) == build_blob(
            chunks, second.emit, TEST_SHAPE
        )

    def test_unsorted_entries_are_rejected_not_sorted(self) -> None:
        """Sorting here would hide a caller that built its listing wrong — and a
        wrong listing is a wrong directory, not a formatting detail.
        """
        entries = file_entries(10)
        with pytest.raises(ValueError, match="sorted and unique"):
            build_tree([entries[1], entries[0], *entries[2:]], Store().emit, TEST_SHAPE)


# ─────────────────────────────────────────────────────────────────────────────
# Incremental stability — the reason for content-defined splitting
# ─────────────────────────────────────────────────────────────────────────────


class TestIncrementalStability:
    def test_inserting_one_entry_rewrites_only_a_few_nodes(self) -> None:
        """The cost table, as a measurement.

        Fixed-fanout packing would rewrite roughly half the nodes here — every
        node after the insertion point. Content-defined splitting rewrites the
        containing node and its ancestors.
        """
        entries = file_entries(4000)
        before_store = Store()
        before_root = build_tree(entries, before_store.emit, TEST_SHAPE)
        before = walk(before_store, before_root)

        inserted = TreeEntry(
            name=b"s000_file000123a.json",  # sorts into the middle of the run
            kind=C.EntryKind.BLOB,
            target=a_name(999_999),
            mode=C.MODE_REGULAR,
            size=7,
        )
        after_store = Store()
        after_root = build_tree(
            sorted([*entries, inserted], key=lambda e: e.name), after_store.emit, TEST_SHAPE
        )
        after = walk(after_store, after_root)

        new_nodes = after.keys() - before.keys()
        assert len(new_nodes) <= 6, f"{len(new_nodes)} nodes rewritten for one insertion"

    def test_reused_node_fraction_is_high(self) -> None:
        entries = file_entries(4000)
        store_a, store_b = Store(), Store()
        root_a = build_tree(entries, store_a.emit, TEST_SHAPE)
        changed = [*entries[:2000], *entries[2001:]]  # delete one entry
        root_b = build_tree(changed, store_b.emit, TEST_SHAPE)

        before = walk(store_a, root_a)
        after = walk(store_b, root_b)
        reused = len(before.keys() & after.keys()) / len(before)
        assert reused > 0.9, f"only {reused:.0%} of nodes survived a single deletion"

    def test_appending_at_the_end_does_not_disturb_the_start(self) -> None:
        entries = file_entries(2000)
        store_a, store_b = Store(), Store()
        root_a = build_tree(entries, store_a.emit, TEST_SHAPE)
        appended = TreeEntry(
            name=b"zzzz_new.json",
            kind=C.EntryKind.BLOB,
            target=a_name(888_888),
            mode=C.MODE_REGULAR,
            size=1,
        )
        root_b = build_tree([*entries, appended], store_b.emit, TEST_SHAPE)

        shared = walk(store_a, root_a).keys() & walk(store_b, root_b).keys()
        assert len(shared) > len(walk(store_a, root_a)) * 0.9

    def test_blob_index_nodes_are_stable_under_an_overwrite(self) -> None:
        """Fixed fanout 1024 would rewrite every index node after a
        same-length overwrite that changed the chunk count.
        """
        chunks = chunk_entries(2000)
        store_a, store_b = Store(), Store()
        root_a = build_blob(chunks, store_a.emit, TEST_SHAPE)
        edited = [*chunks[:900], BlobEntry(a_name(777_777), 1234), *chunks[901:]]
        root_b = build_blob(edited, store_b.emit, TEST_SHAPE)

        before, after = walk(store_a, root_a), walk(store_b, root_b)
        assert len(after.keys() - before.keys()) <= 6


# ─────────────────────────────────────────────────────────────────────────────
# Structure
# ─────────────────────────────────────────────────────────────────────────────


class TestStructure:
    def test_empty_directory_is_a_single_empty_leaf(self) -> None:
        store = Store()
        root = build_tree([], store.emit, TEST_SHAPE)
        assert store.objects[root] == Tree(level=0, entries=())

    def test_empty_file_is_a_single_empty_blob(self) -> None:
        store = Store()
        root = build_blob([], store.emit, TEST_SHAPE)
        assert store.objects[root] == Blob(level=0, entries=())

    def test_a_directory_below_the_minimum_is_always_flat(self) -> None:
        """The only *guaranteed* flatness, and it is worth being precise about.

        A cut can never be taken before ``min_entries``, so a directory that
        small is exactly one leaf node — always, not usually.

        The table reads "≤512 entries → depth 1, 1 fetch". Under
        content-defined splitting that becomes the *expected* case rather than a
        guarantee: a directory between the minimum and the period may take a cut
        and gain a level. That is a cost claim degrading gracefully, not a
        correctness property being lost — every lookup still works, and the
        alternative (fixed fanout, which would make the table exact) breaks
        incremental stability, which is the property everything else depends on.
        """
        for count in range(TEST_SHAPE.min_entries + 1):
            store = Store()
            root = build_tree(file_entries(count), store.emit, TEST_SHAPE)
            assert decode_as(encode(store.objects[root]), Tree).level == 0
            assert len(store) == 1

    def test_a_typical_directory_is_usually_flat(self) -> None:
        """The expected case, measured rather than assumed.

        At the production shape a directory well under the split period should
        almost always be a single node — if this ever drops, the period or the
        minimum has been mis-tuned and every listing got more expensive.
        """
        flat = 0
        trials = 40
        for salt in range(trials):
            store = Store()
            root = build_tree(file_entries(200, salt=salt), store.emit)
            if decode_as(encode(store.objects[root]), Tree).level == 0:
                flat += 1
        assert flat >= trials * 0.6, f"only {flat}/{trials} 200-entry directories were flat"

    def test_wide_directory_grows_in_depth(self) -> None:
        store = Store()
        root = build_tree(file_entries(5000), store.emit, TEST_SHAPE)
        assert decode_as(encode(store.objects[root]), Tree).level >= 2

    def test_every_emitted_node_encodes(self) -> None:
        """The builder must never produce a node the codec rejects — in
        particular an interior node with one child, or one over the byte budget.
        """
        store = Store()
        build_tree(file_entries(5000), store.emit, TEST_SHAPE)
        for node in store.objects.values():
            encode(node)  # raises NotCanonical if the builder got it wrong

    def test_every_emitted_blob_node_encodes(self) -> None:
        store = Store()
        build_blob(chunk_entries(5000), store.emit, TEST_SHAPE)
        for node in store.objects.values():
            encode(node)

    def test_flattening_recovers_the_original_entries(self) -> None:
        """A split directory must still *be* that directory — same entries, same
        order, regardless of how many nodes it took.
        """
        entries = file_entries(3000)
        store = Store()
        root = build_tree(entries, store.emit, TEST_SHAPE)
        assert leaf_entries_in_order(store, root) == entries

    def test_routing_keys_are_the_last_key_of_each_subtree(self) -> None:
        """What makes a lookup able to choose a child without fetching any."""
        store = Store()
        root = build_tree(file_entries(3000), store.emit, TEST_SHAPE)
        node = decode_as(encode(store.objects[root]), Tree)
        assert node.level > 0
        for entry in node.entries:
            covered = leaf_entries_in_order(store, entry.target)
            assert covered[-1].name == entry.name

    @pytest.mark.parametrize("count", [0, 1, 2, 3, 31, 32, 33, 100, 1000])
    def test_builds_at_every_interesting_size(self, count: int) -> None:
        store = Store()
        root = build_tree(file_entries(count), store.emit, TEST_SHAPE)
        assert leaf_entries_in_order(store, root) == file_entries(count)


class TestProductionShape:
    def test_production_shape_matches_the_frozen_constants(self) -> None:
        assert PRODUCTION_SHAPE.domain == C.SPLIT_DOMAIN
        assert PRODUCTION_SHAPE.period == C.SPLIT_PERIOD
        assert PRODUCTION_SHAPE.min_entries == C.SPLIT_MIN_ENTRIES
        assert PRODUCTION_SHAPE.max_entries == C.SPLIT_MAX_ENTRIES
        assert PRODUCTION_SHAPE.max_node_bytes == C.MAX_NODE_BYTES

    def test_a_directory_of_ten_thousand_files_stays_shallow(self) -> None:
        """Depth 2 covers ~262k entries at the production fanout."""
        store = Store()
        root = build_tree(file_entries(10_000), store.emit)
        assert decode_as(encode(store.objects[root]), Tree).level <= 2

    def test_production_nodes_respect_the_byte_budget(self) -> None:
        store = Store()
        build_tree(file_entries(10_000), store.emit)
        assert all(len(encode(n)) <= C.MAX_NODE_BYTES for n in store.objects.values())
