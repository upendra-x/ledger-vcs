"""Editing a tree: the write-side statement of the pruning property.

Reading a version is cheap because an unchanged subtree has an unchanged name.
This is the same fact from the other direction: writing one path rewrites the
nodes *on* that path and nothing else, so committing a one-line change to a
40 GiB environment costs the depth of the tree.

The tests that matter here are the ones about what is **not** written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.errors import InvalidRequest, NotFound
from src.format.constants import MODE_REGULAR, EntryKind
from src.format.model import Chunk, TreeEntry
from src.format.shape import ShapeParams
from src.fs.edit import empty_tree, remove_path, set_path
from src.fs.tree import iter_entries, resolve_path
from src.store.backend import LocalFsBackend
from src.store.cas import ObjectStore
from src.store.catalog import InMemoryWriteCatalog
from src.store.tombstone import SqliteTombstoneStore

if TYPE_CHECKING:
    from pathlib import Path

    from src.ids import ObjectName

SHAPE = ShapeParams(
    domain=b"ledger.tree.split.v1",
    period=32,
    min_entries=4,
    max_entries=64,
    max_node_bytes=16 * 1024,
)


@pytest.fixture
def store(tmp_path: Path) -> ObjectStore:
    return ObjectStore(
        LocalFsBackend(tmp_path / "objects"),
        catalog=InMemoryWriteCatalog(),
        tombstones=SqliteTombstoneStore.open(":memory:"),
        clock=ManualClock(),
    )


def blob_entry(store: ObjectStore, name: bytes, content: bytes) -> TreeEntry:
    """A leaf entry pointing at a one-chunk blob."""
    from src.format.model import Blob, BlobEntry

    chunk = store.put_object(Chunk(content))
    blob = store.put_object(Blob(level=0, entries=(BlobEntry(chunk.name, len(content)),)))
    return TreeEntry(name, EntryKind.BLOB, blob.name, MODE_REGULAR, len(content))


def build(store: ObjectStore, paths: dict[str, bytes]) -> ObjectName:
    tree = empty_tree(store, shape=SHAPE)
    for path, content in paths.items():
        entry = blob_entry(store, path.rsplit("/", 1)[-1].encode(), content)
        tree = set_path(store, tree, path, entry, shape=SHAPE).tree
    return tree


class TestSetPath:
    def test_the_entry_is_findable_where_it_was_put(self, store: ObjectStore) -> None:
        tree = build(store, {"a/b/c.txt": b"hello"})
        resolved = resolve_path(store, tree, "a/b/c.txt")
        assert resolved.entry.size == 5

    def test_intermediate_directories_are_created(self, store: ObjectStore) -> None:
        tree = build(store, {"deep/er/still/file": b"x"})
        assert resolve_path(store, tree, "deep").kind is EntryKind.TREE
        assert resolve_path(store, tree, "deep/er/still").kind is EntryKind.TREE

    def test_the_same_content_produces_the_same_tree(self, store: ObjectStore) -> None:
        """One name per content, applied to an edit: identical content has one name,
        however it was assembled.
        """
        first = build(store, {"a.txt": b"one", "b.txt": b"two"})
        second = build(store, {"b.txt": b"two", "a.txt": b"one"})
        assert first == second

    def test_re_setting_an_identical_entry_writes_nothing(self, store: ObjectStore) -> None:
        """A no-op edit must be free.

        Not merely an optimisation: an importer that re-applies the same state
        would otherwise churn a new tree object per pass, and every one of them
        would be a GC root for as long as its session lived.
        """
        tree = build(store, {"a.txt": b"one"})
        entry = blob_entry(store, b"a.txt", b"one")
        result = set_path(store, tree, "a.txt", entry, shape=SHAPE)
        assert result.tree == tree
        assert result.written == ()

    def test_siblings_keep_their_names(self, store: ObjectStore) -> None:
        """The pruning property, from the write side: an untouched subtree is
        still the same object, which is what makes diff and merge cheap.
        """
        tree = build(store, {"left/a.txt": b"a", "right/b.txt": b"b"})
        before = resolve_path(store, tree, "right").target

        updated = set_path(
            store, tree, "left/a.txt", blob_entry(store, b"a.txt", b"changed"), shape=SHAPE
        ).tree
        assert resolve_path(store, updated, "right").target == before
        assert updated != tree

    def test_a_path_through_a_file_is_refused(self, store: ObjectStore) -> None:
        tree = build(store, {"a.txt": b"one"})
        with pytest.raises(InvalidRequest, match="traverses a file"):
            set_path(store, tree, "a.txt/b", blob_entry(store, b"b", b"x"), shape=SHAPE)

    def test_the_root_is_not_an_entry(self, store: ObjectStore) -> None:
        tree = empty_tree(store, shape=SHAPE)
        with pytest.raises(InvalidRequest, match="root of a tree"):
            set_path(store, tree, "", blob_entry(store, b"x", b"x"), shape=SHAPE)

    def test_the_entry_name_comes_from_the_path(self, store: ObjectStore) -> None:
        """If the two could disagree, the result would be a tree whose lookup
        finds nothing at the path it was written to.
        """
        tree = empty_tree(store, shape=SHAPE)
        mislabelled = blob_entry(store, b"wrong-name", b"x")
        tree = set_path(store, tree, "right-name", mislabelled, shape=SHAPE).tree
        assert [e.name for e in iter_entries(store, tree)] == [b"right-name"]


class TestRemovePath:
    def test_removes_the_entry(self, store: ObjectStore) -> None:
        tree = build(store, {"a.txt": b"one", "b.txt": b"two"})
        tree = remove_path(store, tree, "a.txt", shape=SHAPE).tree
        with pytest.raises(NotFound):
            resolve_path(store, tree, "a.txt")
        assert resolve_path(store, tree, "b.txt").entry.size == 3

    def test_removing_something_absent_writes_nothing(self, store: ObjectStore) -> None:
        tree = build(store, {"a.txt": b"one"})
        result = remove_path(store, tree, "not-here", shape=SHAPE)
        assert result.tree == tree
        assert result.written == ()

    def test_removing_from_an_absent_directory_writes_nothing(self, store: ObjectStore) -> None:
        tree = build(store, {"a.txt": b"one"})
        result = remove_path(store, tree, "nope/deeper", shape=SHAPE)
        assert result.tree == tree
        assert result.written == ()

    def test_removing_and_re_adding_returns_the_original_tree(self, store: ObjectStore) -> None:
        """Immutability made visible: the old tree was never modified, so coming
        back to the same content comes back to the same name.
        """
        tree = build(store, {"a.txt": b"one", "b.txt": b"two"})
        without = remove_path(store, tree, "a.txt", shape=SHAPE).tree
        restored = set_path(
            store, without, "a.txt", blob_entry(store, b"a.txt", b"one"), shape=SHAPE
        ).tree
        assert restored == tree


class TestWrittenOutcomes:
    def test_reports_what_was_new_rather_than_only_what_was_touched(
        self, store: ObjectStore
    ) -> None:
        """The counts a commit reports have to include tree nodes, or a commit
        that created several objects reports zero.
        """
        tree = build(store, {"a.txt": b"one"})
        result = set_path(store, tree, "b.txt", blob_entry(store, b"b.txt", b"two"), shape=SHAPE)
        assert result.written, "rebuilding the root directory wrote at least one node"
        assert all(outcome.size > 0 for outcome in result.written)
        assert result.names == tuple(outcome.name for outcome in result.written)
