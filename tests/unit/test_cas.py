"""The content-addressed store: content addressing and verification enforced,
and unbypassable.

Three properties are load-bearing here, and each has a test that fails loudly:

* bytes that do not hash to their claimed name are refused on ingest;
* bytes that hash correctly but are not *canonical* are also refused — the gap
  is left open;
* bytes that rot in storage are an error on delivery, never a value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.errors import CorruptObject, NotCanonical, ObjectNotFound
from src.format import constants as C
from src.format.codec import encode, name_of, name_of_encoded
from src.format.model import Blob, BlobEntry, Chunk, Commit, Tree, TreeEntry
from src.format.wire import Writer
from src.ids import ChangeId, ObjectName
from src.store.backend import InMemoryBackend, LocalFsBackend
from src.store.cas import ObjectStore
from src.store.catalog import InMemoryWriteCatalog
from src.store.sharding import shard_of
from src.store.tombstone import NullTombstoneStore, SqliteTombstoneStore

if TYPE_CHECKING:
    from pathlib import Path

NAME_A = ObjectName(b"\xaa" * 32)
NAME_B = ObjectName(b"\xbb" * 32)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(start_us=1_700_000_000_000_000)


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def tombstones() -> SqliteTombstoneStore:
    return SqliteTombstoneStore.open(":memory:")


@pytest.fixture
def store(
    backend: InMemoryBackend, tombstones: SqliteTombstoneStore, clock: ManualClock
) -> ObjectStore:
    return ObjectStore(backend, catalog=InMemoryWriteCatalog(), tombstones=tombstones, clock=clock)


def a_tree() -> Tree:
    return Tree(
        level=0,
        entries=(
            TreeEntry(b"README.md", C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 11),
            TreeEntry(b"run.sh", C.EntryKind.BLOB, NAME_B, C.MODE_EXEC, 42),
        ),
    )


class TestRoundTrip:
    def test_put_then_get(self, store: ObjectStore) -> None:
        chunk = Chunk(b"hello world")
        outcome = store.put_object(chunk)
        assert outcome.created
        assert store.get_object(outcome.name) == chunk

    def test_get_as_narrows_the_type(self, store: ObjectStore) -> None:
        name = store.put_object(a_tree()).name
        assert store.get_as(name, Tree).level == 0

    def test_get_as_rejects_the_wrong_type(self, store: ObjectStore) -> None:
        """A tree entry with the wrong kind points at an unexpected object type;
        that should fail at the fetch, not several frames later.
        """
        from src.errors import MalformedObject

        name = store.put_object(Chunk(b"x")).name
        with pytest.raises(MalformedObject, match="not of the expected kind"):
            store.get_as(name, Tree)

    def test_missing_object_raises_not_found(self, store: ObjectStore) -> None:
        with pytest.raises(ObjectNotFound):
            store.get(NAME_A)

    def test_get_range_reads_into_the_payload(self, store: ObjectStore) -> None:
        """Offsets are into the *content*, not the framed bytes — a caller
        reading byte 100 of a file must not have to know about the header.
        """
        payload = bytes(range(256))
        name = store.put_object(Chunk(payload)).name
        assert store.get_range(name, 10, 5) == payload[10:15]


class TestDeduplication:
    def test_storing_the_same_object_twice_creates_once(self, store: ObjectStore) -> None:
        first = store.put_object(Chunk(b"same bytes"))
        second = store.put_object(Chunk(b"same bytes"))
        assert first.created
        assert not second.created
        assert first.name == second.name

    def test_identical_content_from_unrelated_trees_is_stored_once(
        self, store: ObjectStore, backend: InMemoryBackend
    ) -> None:
        """Nothing in an object identifies its environment, which is
        exactly what makes identical content shared across *unrelated*
        environments — the Deduplication requirement.
        """
        shared = Chunk(b"a base image layer, shared by everything")
        store.put_object(shared)
        before = len(backend)
        store.put_object(shared)
        assert len(backend) == before

    def test_missing_reports_only_what_is_absent(self, store: ObjectStore) -> None:
        present = store.put_object(Chunk(b"present")).name
        absent = name_of(Chunk(b"absent"))
        assert store.missing([present, absent]) == frozenset({absent})

    def test_missing_deduplicates_its_input(self, store: ObjectStore) -> None:
        absent = name_of(Chunk(b"absent"))
        assert store.missing([absent, absent, absent]) == frozenset({absent})

    def test_missing_on_an_empty_batch(self, store: ObjectStore) -> None:
        assert store.missing([]) == frozenset()


class TestIngestVerification:
    """A hash must mean the same thing to everyone."""

    def test_rejects_bytes_that_do_not_hash_to_the_claimed_name(self, store: ObjectStore) -> None:
        with pytest.raises(CorruptObject, match="do not hash to the name"):
            store.put(NAME_A, encode(Chunk(b"not what NAME_A names")))

    def test_a_poisoned_name_cannot_be_planted(
        self, store: ObjectStore, backend: InMemoryBackend
    ) -> None:
        """The cross-tenant attack this prevents.

        If a writer could store arbitrary bytes under a chosen name, the next
        honest writer to offer that hash would be told 'present', upload
        nothing, and inherit content that fails verification on every read — an
        unrelated environment made unreadable by a name it never chose.
        """
        honest = Chunk(b"the real content")
        target = name_of(honest)

        with pytest.raises(CorruptObject):
            store.put(target, encode(Chunk(b"malicious substitute")))
        assert len(backend) == 0

        store.put(target, encode(honest))
        assert store.get_object(target) == honest

    def test_rejects_a_valid_hash_over_a_non_canonical_encoding(self, store: ObjectStore) -> None:
        """The gap is left open, closed here.

        These bytes hash honestly to their own name — the hash check alone is
        satisfied — but
        the entries are out of order, so accepting them would give this directory
        a second valid name and split deduplication.
        """
        out_of_order = (
            Writer()
            .u8(C.ObjectKind.TREE.value)
            .u8(C.FORMAT_VERSION)
            .u8(0)
            .u32(2)
            .bytes_u8(b"zebra")
            .u8(C.EntryKind.BLOB.value)
            .name(NAME_A)
            .u16(C.MODE_REGULAR)
            .u64(1)
            .bytes_u8(b"apple")
            .u8(C.EntryKind.BLOB.value)
            .name(NAME_B)
            .u16(C.MODE_REGULAR)
            .u64(1)
            .finish()
        )
        honest_name = name_of_encoded(out_of_order)

        with pytest.raises(NotCanonical):
            store.put(honest_name, out_of_order)

    def test_rejects_garbage_that_is_not_an_object_at_all(self, store: ObjectStore) -> None:
        garbage = b"\x99\x99 definitely not a ledger object"
        with pytest.raises((CorruptObject, NotCanonical, Exception)):
            store.put(name_of_encoded(garbage), garbage)


class TestDeliveryVerification:
    """Verification on delivery: damaged data is detected, never handed out."""

    def test_detects_corruption_on_delivery(
        self, store: ObjectStore, backend: InMemoryBackend
    ) -> None:
        chunk = Chunk(b"the quick brown fox jumps over the lazy dog")
        name = store.put_object(chunk).name

        rotted = bytearray(encode(chunk))
        rotted[-1] ^= 0x01
        backend.corrupt(name.hex, bytes(rotted))

        with pytest.raises(CorruptObject, match="do not match the name"):
            store.get(name)

    def test_corruption_is_not_downgraded_to_a_miss(
        self, store: ObjectStore, backend: InMemoryBackend
    ) -> None:
        """A miss invites a retry; this means a medium or a cache is wrong and
        someone has to look at it.
        """
        name = store.put_object(Chunk(b"content")).name
        backend.corrupt(name.hex, b"\x01\x01totally different")

        with pytest.raises(CorruptObject):
            store.get(name)

    def test_a_single_flipped_bit_anywhere_is_caught(
        self, store: ObjectStore, backend: InMemoryBackend
    ) -> None:
        original = encode(Chunk(bytes(range(200))))
        name = store.put(name_of_encoded(original), original).name

        for position in range(0, len(original), 17):
            mutated = bytearray(original)
            mutated[position] ^= 0x80
            backend.corrupt(name.hex, bytes(mutated))
            with pytest.raises(CorruptObject):
                store.get(name)


class TestTombstoneInteraction:
    """The existence predicate includes the tombstone term from day one."""

    def test_a_tombstoned_object_reports_missing_even_when_present(
        self, store: ObjectStore, tombstones: SqliteTombstoneStore, clock: ManualClock
    ) -> None:
        """The anti-resurrection guard, in miniature.

        The bytes are physically there, but the store must still say 'missing'
        so the client re-uploads and the object is rewritten with its children.
        """
        name = store.put_object(Chunk(b"swept but still present")).name
        assert store.missing([name]) == frozenset()

        tombstones.record([name], expires_at_us=clock.now_us() + 86_400_000_000)
        assert store.missing([name]) == frozenset({name})

    def test_re_uploading_clears_the_tombstone(
        self, store: ObjectStore, tombstones: SqliteTombstoneStore, clock: ManualClock
    ) -> None:
        """Without this the store deadlocks: the object stays permanently
        'missing', so every write uploads it again and never converges.
        """
        chunk = Chunk(b"resurrected")
        name = store.put_object(chunk).name
        tombstones.record([name], expires_at_us=clock.now_us() + 86_400_000_000)
        assert store.missing([name]) == frozenset({name})

        store.put_object(chunk)
        assert store.missing([name]) == frozenset()

    def test_tombstones_expire(self, tombstones: SqliteTombstoneStore, clock: ManualClock) -> None:
        tombstones.record([NAME_A], expires_at_us=clock.now_us() + 1000)
        assert tombstones.count() == 1

        clock.advance_us(999)
        assert tombstones.purge_expired(clock.now_us()) == 0

        clock.advance_us(2)
        assert tombstones.purge_expired(clock.now_us()) == 1

    def test_null_tombstone_store_disables_the_guard(
        self, backend: InMemoryBackend, clock: ManualClock
    ) -> None:
        """Proves the guard is load-bearing rather than decorative — this is what
        ``tests/integration/test_gc.py::TestResurrection`` builds on.
        """
        unguarded = ObjectStore(
            backend,
            catalog=InMemoryWriteCatalog(),
            tombstones=NullTombstoneStore(),
            clock=clock,
        )
        name = unguarded.put_object(Chunk(b"x")).name
        unguarded.tombstones.record([name], expires_at_us=clock.now_us() + 1)
        assert unguarded.missing([name]) == frozenset(), "NullTombstoneStore should record nothing"


class TestCatalog:
    def test_records_every_new_object(self, backend: InMemoryBackend, clock: ManualClock) -> None:
        catalog = InMemoryWriteCatalog()
        store = ObjectStore(backend, catalog=catalog, tombstones=NullTombstoneStore(), clock=clock)
        store.put_object(Chunk(b"one"))
        store.put_object(Chunk(b"two"))
        count, total_bytes = catalog.total()
        assert count == 2
        assert total_bytes > 0

    def test_does_not_double_count_a_deduplicated_write(
        self, backend: InMemoryBackend, clock: ManualClock
    ) -> None:
        catalog = InMemoryWriteCatalog()
        store = ObjectStore(backend, catalog=catalog, tombstones=NullTombstoneStore(), clock=clock)
        store.put_object(Chunk(b"same"))
        store.put_object(Chunk(b"same"))
        assert catalog.total()[0] == 1

    def test_delete_forgets_the_catalog_entry(
        self, backend: InMemoryBackend, clock: ManualClock
    ) -> None:
        catalog = InMemoryWriteCatalog()
        store = ObjectStore(backend, catalog=catalog, tombstones=NullTombstoneStore(), clock=clock)
        name = store.put_object(Chunk(b"transient")).name
        assert store.delete([name]) == 1
        assert catalog.total()[0] == 0

    def test_sharding_is_stable_and_covers_the_space(self) -> None:
        names = [ObjectName(bytes([i]) + b"\x00" * 31) for i in range(256)]
        shards = {shard_of(n, 4) for n in names}
        assert shards == set(range(16))
        assert all(shard_of(n, 4) == shard_of(n, 4) for n in names)


class TestLocalFsBackend:
    def test_round_trip_through_the_filesystem(self, tmp_path: Path, clock: ManualClock) -> None:
        store = ObjectStore(
            LocalFsBackend(tmp_path / "objects"),
            catalog=InMemoryWriteCatalog(),
            tombstones=NullTombstoneStore(),
            clock=clock,
        )
        commit = Commit(
            tree=NAME_A,
            parents=(),
            change_id=ChangeId("ab" * 16),
            author="agent-17",
            committer="agent-17",
            timestamp_us=clock.now_us(),
            message="first",
        )
        name = store.put_object(commit).name
        assert store.get_object(name) == commit

    def test_objects_are_sharded_two_levels_deep(self, tmp_path: Path, clock: ManualClock) -> None:
        """The name *is* the address, and the two-level
        prefix keeps
        any single directory small and spreads cold-read load across prefixes.
        """
        root = tmp_path / "objects"
        store = ObjectStore(
            LocalFsBackend(root),
            catalog=InMemoryWriteCatalog(),
            tombstones=NullTombstoneStore(),
            clock=clock,
        )
        name = store.put_object(Chunk(b"sharded")).name
        expected = root / name.hex[:2] / name.hex[2:4] / name.hex
        assert expected.is_file()

    def test_ranged_read_does_not_load_the_whole_object(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        backend = LocalFsBackend(tmp_path / "objects")
        store = ObjectStore(
            backend,
            catalog=InMemoryWriteCatalog(),
            tombstones=NullTombstoneStore(),
            clock=clock,
        )
        payload = bytes(range(256)) * 100
        name = store.put_object(Chunk(payload)).name
        assert backend.read_range(name.hex, 2 + 500, 10) == payload[500:510]

    def test_write_is_atomic_leaving_no_partial_files(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        backend = LocalFsBackend(tmp_path / "objects")
        store = ObjectStore(
            backend,
            catalog=InMemoryWriteCatalog(),
            tombstones=NullTombstoneStore(),
            clock=clock,
        )
        for i in range(20):
            store.put_object(Chunk(f"object {i}".encode()))
        assert not [k for k in backend.iter_keys() if k.startswith(".tmp-")]

    def test_rejects_an_unsafe_key(self, tmp_path: Path) -> None:
        backend = LocalFsBackend(tmp_path / "objects")
        for key in ("../escape", "a/b", "ab"):
            with pytest.raises(ValueError, match="unsafe backend key"):
                backend.write(key, b"x")


class TestBlobAndTreeStorage:
    def test_a_blob_and_its_chunks_round_trip(self, store: ObjectStore) -> None:
        chunks = [Chunk(f"chunk {i}".encode()) for i in range(4)]
        entries = [BlobEntry(store.put_object(c).name, c.size) for c in chunks]
        blob = Blob(level=0, entries=tuple(entries))
        name = store.put_object(blob).name

        loaded = store.get_as(name, Blob)
        assert loaded == blob
        assert b"".join(store.get_as(e.target, Chunk).data for e in loaded.entries) == b"".join(
            c.data for c in chunks
        )

    def test_a_tree_round_trips(self, store: ObjectStore) -> None:
        tree = a_tree()
        assert store.get_as(store.put_object(tree).name, Tree) == tree
