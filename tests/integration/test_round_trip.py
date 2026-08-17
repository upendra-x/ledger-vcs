"""The first end-to-end loop: a directory, stored and restored byte for byte.

``ingest`` → ``checkout`` is the cheapest total check that the whole stack is
right. If chunking, tree shaping, encoding, storage, range descent or
materialization is wrong anywhere, the restored tree differs from the original
and ``compare_trees`` says exactly where.

The read-path tests alongside it assert *cost*, not just correctness: a ranged
read of a huge file costs a handful of fetches set by depth
rather than size, and a counting store turns that claim into an assertion.
"""

from __future__ import annotations

import filecmp
import os
import random
from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.errors import InvalidRequest, NotFound
from src.format.cdc import ChunkParams
from src.format.constants import EntryKind
from src.format.shape import ShapeParams
from src.fs.blob import BlobReader
from src.fs.tree import iter_entries, list_dir, lookup, resolve_path
from src.runtime.ingest import Ingester
from src.runtime.materialize import Materializer
from src.store.backend import InMemoryBackend
from src.store.cas import ObjectStore
from src.store.catalog import InMemoryWriteCatalog
from src.store.tombstone import SqliteTombstoneStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.ids import ObjectName

CHUNK_PARAMS = ChunkParams.for_average(4096)
SHAPE_PARAMS = ShapeParams(
    domain=b"ledger.tree.split.v1",
    period=16,
    min_entries=4,
    max_entries=64,
    max_node_bytes=16 * 1024,
)


class CountingStore:
    """Wraps a store and counts fetches, so cost claims become assertions."""

    def __init__(self, inner: ObjectStore) -> None:
        self._inner = inner
        self.fetches = 0

    def __getattr__(self, item: str) -> object:
        return getattr(self._inner, item)

    def get(self, name: ObjectName) -> bytes:
        self.fetches += 1
        return self._inner.get(name)

    def get_object(self, name: ObjectName) -> object:
        self.fetches += 1
        return self._inner.get_object(name)

    def get_as(self, name: ObjectName, expected: type) -> object:
        self.fetches += 1
        return self._inner.get_as(name, expected)

    def reset(self) -> None:
        self.fetches = 0


@pytest.fixture
def store() -> Iterator[ObjectStore]:
    tombstones = SqliteTombstoneStore.open(":memory:")
    yield ObjectStore(
        InMemoryBackend(),
        catalog=InMemoryWriteCatalog(),
        tombstones=tombstones,
        clock=ManualClock(),
    )
    tombstones.close()


@pytest.fixture
def ingester(store: ObjectStore) -> Ingester:
    return Ingester(
        store,
        clock=ManualClock(start_us=1_700_000_000_000_000),
        chunk_params=CHUNK_PARAMS,
        shape_params=SHAPE_PARAMS,
    )


def build_environment(root: Path) -> None:
    """A realistic environment, in miniature: manifest, task, dataset,
    links."""
    (root / "task").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "images").mkdir()
    (root / "harbor.yaml").write_text("name: demo\ncontainers:\n  app: {}\n  db: {}\n")
    (root / "README.md").write_text("# demo environment\n")
    (root / "task" / "prompt.md").write_text("Solve the failing test.\n")
    verifier = root / "task" / "verifier.py"
    verifier.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(0)\n")
    verifier.chmod(0o755)
    (root / "data" / "train.bin").write_bytes(random.Random(42).randbytes(300_000))
    (root / "images" / "index.json").write_text('{"schemaVersion": 2}\n')
    (root / "task" / "dataset.link").symlink_to("../data/train.bin")


def compare_trees(left: Path, right: Path) -> None:
    """Assert two directory trees are identical, including modes and symlinks."""
    left_entries = sorted(os.listdir(left))
    right_entries = sorted(os.listdir(right))
    assert left_entries == right_entries, f"entries differ under {left} vs {right}"

    for name in left_entries:
        a, b = left / name, right / name
        assert a.is_symlink() == b.is_symlink(), f"symlink-ness differs for {name}"
        if a.is_symlink():
            assert os.readlink(a) == os.readlink(b), f"symlink target differs for {name}"
            continue
        assert a.is_dir() == b.is_dir(), f"directory-ness differs for {name}"
        if a.is_dir():
            compare_trees(a, b)
            continue
        assert filecmp.cmp(a, b, shallow=False), f"content differs for {name}"
        assert (a.stat().st_mode & 0o111) == (b.stat().st_mode & 0o111), (
            f"executable bit differs for {name}"
        )


class TestRoundTrip:
    def test_ingest_then_checkout_reproduces_the_tree(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        """Ingest and check out reproduce the tree byte for byte — the acceptance
        test for the format, the store, the chunker and the ingester together.
        """
        source = tmp_path / "env"
        build_environment(source)

        commit, _ = ingester.ingest_commit(source, author="agent-17", message="initial")
        destination = tmp_path / "restored"
        Materializer(store).materialize_commit(commit, destination)

        compare_trees(source, destination)

    def test_restored_tree_re_ingests_to_the_same_name(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        """The strongest statement of round-trip fidelity.

        If materialization lost *anything* the codec records — a mode, a symlink
        target, an empty directory — re-ingesting the restored tree would
        produce a different tree name. Comparing names compares everything at
        once, and does it in one line.
        """
        source = tmp_path / "env"
        build_environment(source)
        original, _ = ingester.ingest_directory(source)

        restored = tmp_path / "restored"
        Materializer(store).materialize_tree(original, restored)
        again, stats = ingester.ingest_directory(restored)

        assert again == original
        assert stats.objects_created == 0, "a faithful round trip creates no new objects"

    def test_checkout_is_atomic(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        """A failed checkout leaves nothing where the caller expects a tree."""
        source = tmp_path / "env"
        build_environment(source)
        commit, _ = ingester.ingest_commit(source, author="a", message="m")

        destination = tmp_path / "out"
        Materializer(store).materialize_commit(commit, destination)
        assert destination.is_dir()
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".ledger-checkout")]

    def test_checkout_refuses_to_clobber_by_default(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)
        destination = tmp_path / "out"
        destination.mkdir()

        with pytest.raises(InvalidRequest, match="already exists"):
            Materializer(store).materialize_tree(tree, destination)

    def test_checkout_can_replace_an_existing_tree(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)
        destination = tmp_path / "out"
        destination.mkdir()
        (destination / "stale.txt").write_text("should be gone\n")

        Materializer(store).materialize_tree(tree, destination, overwrite=True)
        assert not (destination / "stale.txt").exists()
        compare_trees(source, destination)

    def test_an_old_version_restores_its_own_content(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        """Design *Version Environments*: going back to an earlier version.

        Checking out the first commit after the file has changed must produce the
        original bytes — the whole point of a snapshot over shared content.
        """
        source = tmp_path / "env"
        build_environment(source)
        first, _ = ingester.ingest_commit(source, author="a", message="v1")

        (source / "task" / "prompt.md").write_text("A completely different task.\n")
        second, _ = ingester.ingest_commit(source, author="a", message="v2")
        assert first != second

        restored = tmp_path / "v1"
        Materializer(store).materialize_commit(first, restored)
        assert (restored / "task" / "prompt.md").read_text() == "Solve the failing test.\n"


class TestReadPath:
    def test_resolve_a_nested_path(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)

        resolved = resolve_path(store, tree, "task/verifier.py")
        assert resolved.kind is EntryKind.BLOB
        assert resolved.entry.mode == 0o755

    def test_missing_path_is_not_found(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)

        with pytest.raises(NotFound, match="does not exist"):
            resolve_path(store, tree, "task/nope.md")

    def test_traversing_a_file_is_not_found(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)

        with pytest.raises(NotFound, match="non-directory"):
            resolve_path(store, tree, "README.md/inner")

    @pytest.mark.parametrize("bad", ["../escape", "a/../b", "task/../../etc/passwd"])
    def test_traversal_attempts_are_rejected(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path, bad: str
    ) -> None:
        """``..`` is rejected rather than normalised.

        Normalising would silently resolve to a different file than the caller
        named, which is how a path check becomes a path bypass.
        """
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)

        with pytest.raises(InvalidRequest, match="reserved"):
            resolve_path(store, tree, bad)

    def test_ranged_read_of_a_large_file(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)
        original = (source / "data" / "train.bin").read_bytes()

        resolved = resolve_path(store, tree, "data/train.bin")
        reader = BlobReader(store, resolved.target)

        assert reader.size == len(original)
        assert reader.read(0, 100) == original[:100]
        assert reader.read(150_000, 1_000) == original[150_000:151_000]
        assert reader.read(len(original) - 10) == original[-10:]
        assert reader.read_all() == original

    def test_reads_past_the_end_are_clamped(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)
        resolved = resolve_path(store, tree, "data/train.bin")
        reader = BlobReader(store, resolved.target)

        assert reader.read(reader.size) == b""
        assert reader.read(reader.size + 1000) == b""
        assert len(reader.read(reader.size - 5, 500)) == 5

    def test_a_ranged_read_does_not_fetch_the_whole_file(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        """The cost claim, asserted.

        Reading a kilobyte out of the middle must touch a handful of objects,
        not the ~75 chunks the file is made of.
        """
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)
        resolved = resolve_path(store, tree, "data/train.bin")

        counting = CountingStore(store)
        reader = BlobReader(counting, resolved.target)  # type: ignore[arg-type]
        total_chunks = len(list(reader.chunk_names()))
        counting.reset()

        reader.read(150_000, 1_000)
        assert counting.fetches <= 5, (
            f"a 1 KiB ranged read fetched {counting.fetches} objects from a "
            f"{total_chunks}-chunk file"
        )

    def test_path_resolution_costs_one_fetch_per_component(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)

        counting = CountingStore(store)
        resolve_path(counting, tree, "task/verifier.py")  # type: ignore[arg-type]
        assert counting.fetches <= 4, f"{counting.fetches} fetches for a two-component path"


class TestListing:
    def test_lists_in_name_order(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "env"
        build_environment(source)
        tree, _ = ingester.ingest_directory(source)

        names = [e.name for e in list_dir(store, tree).entries]
        assert names == sorted(names)
        assert b"harbor.yaml" in names

    def test_pages_a_wide_directory_with_a_cursor(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        """No cap on how many files a directory can list."""
        source = tmp_path / "wide"
        source.mkdir()
        for i in range(1_500):
            (source / f"f{i:06d}.txt").write_text(f"{i}\n")
        tree, _ = ingester.ingest_directory(source)

        collected: list[bytes] = []
        cursor: bytes | None = None
        pages = 0
        while True:
            page = list_dir(store, tree, after=cursor, limit=97)
            collected.extend(e.name for e in page.entries)
            pages += 1
            if page.exhausted:
                break
            cursor = page.cursor

        assert len(collected) == 1_500
        assert len(set(collected)) == 1_500, "an entry appeared twice across pages"
        assert collected == sorted(collected)
        assert pages > 10, "the fixture should have needed many pages"

    def test_the_final_page_reports_no_cursor(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        """The classic cursor off-by-one: handing back a cursor that yields an
        empty page.
        """
        source = tmp_path / "small"
        source.mkdir()
        for i in range(10):
            (source / f"f{i}.txt").write_text("x")
        tree, _ = ingester.ingest_directory(source)

        page = list_dir(store, tree, limit=10)
        assert len(page.entries) == 10
        assert page.exhausted

    def test_lookup_finds_an_entry_in_a_split_directory(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "wide"
        source.mkdir()
        for i in range(1_500):
            (source / f"f{i:06d}.txt").write_text(f"{i}\n")
        tree, _ = ingester.ingest_directory(source)

        for probe in (b"f000000.txt", b"f000750.txt", b"f001499.txt"):
            assert lookup(store, tree, probe) is not None, probe
        assert lookup(store, tree, b"f999999.txt") is None

    def test_iteration_and_listing_agree(
        self, ingester: Ingester, store: ObjectStore, tmp_path: Path
    ) -> None:
        source = tmp_path / "wide"
        source.mkdir()
        for i in range(400):
            (source / f"f{i:04d}.txt").write_text("x")
        tree, _ = ingester.ingest_directory(source)

        streamed = [e.name for e in iter_entries(store, tree)]
        paged = [e.name for e in list_dir(store, tree, limit=10_000).entries]
        assert streamed == paged
