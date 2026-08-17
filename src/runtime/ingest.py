"""Turn a directory on disk into a commit.

This is the write half of the round trip, and the first place all of the format
machinery is used together: a directory walk produces chunks, chunks produce
blobs, blobs and subtrees produce trees, and a tree produces a commit.

Two properties are worth stating because everything downstream leans on them:

**Ingesting the same directory twice creates nothing the second time.** Not as an
optimisation — as a consequence of every name being a content hash. It is also
the cheapest possible check that the whole pipeline is deterministic, which is
why the CLI prints created-versus-offered counts rather than a spinner.

**Nothing here knows what it is ingesting.** Files are bytes, directories are
trees, and the environment manifest is a blob like any other. The
*Format Agnostic* is not implemented anywhere — it is the absence of code, and
keeping it absent is a rule this package holds to rather than one a test
enforces.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, final

from src.errors import InvalidRequest
from src.format.cdc import PRODUCTION_PARAMS, ChunkParams, chunk_stream
from src.format.constants import MAX_ENTRY_NAME_BYTES, MODE_EXEC, MODE_REGULAR, EntryKind
from src.format.model import Blob, BlobEntry, Chunk, Commit, TreeEntry
from src.format.shape import PRODUCTION_SHAPE, ShapeParams, build_blob, build_tree
from src.ids import ChangeId

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from src.clock import Clock
    from src.format.cdc import ByteReader
    from src.format.model import LedgerObject
    from src.ids import ObjectName
    from src.store.cas import ObjectStore, PutOutcome

__all__ = ["IngestStats", "Ingester", "default_exclude"]

#: Directories never worth ingesting. ``.git`` in particular: importing a
#: repository means converting its history, not storing its
#: object database as opaque files.
_DEFAULT_EXCLUDED = frozenset({".git", ".jj", "__pycache__", ".DS_Store"})


def default_exclude(name: str) -> bool:
    return name in _DEFAULT_EXCLUDED


@final
@dataclass(frozen=True, slots=True)
class IngestStats:
    """What an ingest actually moved.

    ``created`` against ``offered`` is the deduplication metric, and
    it is reported rather than estimated because the store returns it per write.
    """

    files: int = 0
    directories: int = 0
    symlinks: int = 0
    chunks: int = 0
    objects_offered: int = 0
    objects_created: int = 0
    bytes_offered: int = 0
    bytes_stored: int = 0
    #: What the newly-stored objects occupy on the medium. Lower than
    #: ``bytes_stored`` wherever compression at rest paid off — reported
    #: rather than assumed, because for container layers and model weights it
    #: does not.
    bytes_on_disk: int = 0

    @property
    def dedup_ratio(self) -> float:
        """Fraction of offered bytes that did not need storing."""
        if self.bytes_offered == 0:
            return 0.0
        return 1.0 - (self.bytes_stored / self.bytes_offered)

    @classmethod
    def counting(cls, outcomes: Iterable[PutOutcome]) -> IngestStats:
        """Tally writes made outside a directory walk.

        Anything that builds objects directly — a tree edit, an image layout, a
        merge — must fold its writes in here, or the created-versus-offered
        counts describe only part of what a commit cost. Those two numbers are
        how deduplication is *reported* rather than asserted, so a partial tally
        is worse than none.
        """
        total = cls()
        for outcome in outcomes:
            total = replace(
                total,
                objects_offered=total.objects_offered + 1,
                objects_created=total.objects_created + int(outcome.created),
                bytes_offered=total.bytes_offered + outcome.size,
                bytes_stored=total.bytes_stored + (outcome.size if outcome.created else 0),
                bytes_on_disk=(total.bytes_on_disk + outcome.stored_size),
            )
        return total

    def __add__(self, other: IngestStats) -> IngestStats:
        return IngestStats(
            files=self.files + other.files,
            directories=self.directories + other.directories,
            symlinks=self.symlinks + other.symlinks,
            chunks=self.chunks + other.chunks,
            objects_offered=self.objects_offered + other.objects_offered,
            objects_created=self.objects_created + other.objects_created,
            bytes_offered=self.bytes_offered + other.bytes_offered,
            bytes_stored=self.bytes_stored + other.bytes_stored,
            bytes_on_disk=self.bytes_on_disk + other.bytes_on_disk,
        )


@final
class Ingester:
    """Walks a directory and writes it as objects.

    Chunk and shape parameters are injected rather than imported, so tests run
    at a scale where multi-level structures appear in kilobytes — and so nothing
    can accidentally chunk with a parameter set that differs from the one the
    corpus was built with.
    """

    __slots__ = ("_chunk_params", "_clock", "_exclude", "_shape_params", "_store")

    def __init__(
        self,
        store: ObjectStore,
        *,
        clock: Clock,
        chunk_params: ChunkParams = PRODUCTION_PARAMS,
        shape_params: ShapeParams = PRODUCTION_SHAPE,
        exclude: Callable[[str], bool] = default_exclude,
    ) -> None:
        self._store = store
        self._clock = clock
        self._chunk_params = chunk_params
        self._shape_params = shape_params
        self._exclude = exclude

    # ── public API ───────────────────────────────────────────────────────────

    def ingest_directory(self, root: Path) -> tuple[ObjectName, IngestStats]:
        """Write a directory tree, returning its root tree name."""
        root = Path(root)
        if not root.is_dir():
            raise InvalidRequest(f"not a directory: {root}")
        stats = _Accumulator()
        name = self._walk_directory(root, stats)
        return name, stats.frozen

    def ingest_commit(
        self,
        root: Path,
        *,
        author: str,
        message: str,
        parents: Sequence[ObjectName] = (),
        change_id: ChangeId | None = None,
        metadata: Sequence[tuple[str, str]] = (),
    ) -> tuple[ObjectName, IngestStats]:
        """Write a directory tree and a commit naming it."""
        tree_name, stats = self.ingest_directory(root)
        commit = Commit(
            tree=tree_name,
            parents=tuple(parents),
            change_id=change_id or ChangeId.new(),
            author=author,
            committer=author,
            timestamp_us=self._clock.now_us(),
            message=message,
            metadata=tuple(sorted(metadata)),
        )
        accumulator = _Accumulator(stats)
        name = accumulator.put(self._store, commit)
        return name, accumulator.frozen

    def ingest_stream(self, source: ByteReader) -> tuple[ObjectName, IngestStats]:
        """Write a stream of unknown length as a blob.

        The path that has no file behind it: a decompressed container layer, a
        converted git object, an upload still arriving. Streaming rather than
        buffering is what lets a 6 GiB layer be stored on a host that could not
        hold it — and it is the same code the directory walk uses, so the two
        cannot chunk differently.
        """
        stats = _Accumulator()
        name = self._write_stream(source, stats)
        return name, stats.frozen

    def ingest_bytes(self, data: bytes) -> tuple[ObjectName, IngestStats]:
        """Write bytes already in memory as a blob. For content that is small by
        construction — a manifest, an image config, a symlink target.
        """
        stats = _Accumulator()
        name = self._write_bytes(data, stats)
        return name, stats.frozen

    # ── internals ────────────────────────────────────────────────────────────

    def _walk_directory(self, path: Path, stats: _Accumulator) -> ObjectName:
        entries: list[TreeEntry] = []

        # Sorted by *encoded name bytes*, which is the order the codec requires.
        # Sorting by str would agree today because UTF-8 is order-preserving,
        # but making the byte order explicit is what stops a future locale-aware
        # sort from silently renaming every wide directory.
        children = sorted(os.scandir(path), key=lambda e: e.name.encode())

        for child in children:
            if self._exclude(child.name):
                continue
            name_bytes = child.name.encode()
            if len(name_bytes) > MAX_ENTRY_NAME_BYTES:
                raise InvalidRequest(
                    f"path component too long: {child.name!r}", limit=MAX_ENTRY_NAME_BYTES
                )
            entries.append(self._entry_for(Path(child.path), name_bytes, child, stats))

        stats.directories += 1
        return build_tree(entries, stats.emitter(self._store), self._shape_params)

    def _entry_for(
        self, path: Path, name_bytes: bytes, child: os.DirEntry[str], stats: _Accumulator
    ) -> TreeEntry:
        if child.is_symlink():
            target = os.readlink(path).encode()
            stats.symlinks += 1
            # A symlink's target is content, stored as an ordinary blob. That
            # keeps the model at four object types and means a link to a huge
            # path costs nothing special.
            blob_name = self._write_bytes(target, stats)
            return TreeEntry(name_bytes, EntryKind.SYMLINK, blob_name, 0, len(target))

        if child.is_dir(follow_symlinks=False):
            return TreeEntry(name_bytes, EntryKind.TREE, self._walk_directory(path, stats), 0, 0)

        if not child.is_file(follow_symlinks=False):
            # Sockets, FIFOs and devices have no content to version, and
            # silently skipping them would make a checkout differ from its
            # source without saying so.
            raise InvalidRequest(f"unsupported file type: {path}")

        info = child.stat(follow_symlinks=False)
        mode = MODE_EXEC if info.st_mode & stat.S_IXUSR else MODE_REGULAR
        stats.files += 1
        return TreeEntry(
            name_bytes, EntryKind.BLOB, self._write_file(path, stats), mode, info.st_size
        )

    def _write_file(self, path: Path, stats: _Accumulator) -> ObjectName:
        """Chunk a file and build its blob, streaming rather than loading it.

        A 40 GiB dataset must not need 40 GiB of memory, and this is the only
        place that guarantee is made.
        """
        with path.open("rb") as handle:
            return self._write_stream(handle, stats)

    def _write_stream(self, source: ByteReader, stats: _Accumulator) -> ObjectName:
        entries: list[BlobEntry] = []
        for payload in chunk_stream(source, self._chunk_params):
            entries.append(BlobEntry(stats.put(self._store, Chunk(payload)), len(payload)))
            stats.chunks += 1

        if not entries:
            return stats.put(self._store, Blob(level=0, entries=()))
        return build_blob(entries, stats.emitter(self._store), self._shape_params)

    def _write_bytes(self, data: bytes, stats: _Accumulator) -> ObjectName:
        if not data:
            return stats.put(self._store, Blob(level=0, entries=()))
        entries = [
            BlobEntry(stats.put(self._store, Chunk(bytes(payload))), len(payload))
            for payload in _chunk_all(data, self._chunk_params)
        ]
        stats.chunks += len(entries)
        return build_blob(entries, stats.emitter(self._store), self._shape_params)


def _chunk_all(data: bytes, params: ChunkParams) -> list[memoryview]:
    from src.format.cdc import chunk_bytes

    return list(chunk_bytes(data, params))


class _Accumulator:
    """Mutable counters, kept out of the public dataclass.

    ``IngestStats`` is frozen because it is a result someone may hold on to; the
    tallying during a walk is a different job, and mixing the two produces a
    result object whose value depends on when you looked at it.
    """

    __slots__ = ("_base", "chunks", "directories", "files", "outcome", "symlinks")

    def __init__(self, base: IngestStats | None = None) -> None:
        self._base = base or IngestStats()
        self.files = 0
        self.directories = 0
        self.symlinks = 0
        self.chunks = 0
        self.outcome = IngestStats()

    def put(self, store: ObjectStore, obj: LedgerObject) -> ObjectName:
        return self.record(store.put_object(obj)).name

    def record(self, result: PutOutcome) -> PutOutcome:
        """Tally one write. The single place the walk's counters move, so a new
        way of storing an object cannot quietly stop being counted.
        """
        self.outcome = replace(
            self.outcome,
            objects_offered=self.outcome.objects_offered + 1,
            objects_created=self.outcome.objects_created + int(result.created),
            bytes_offered=self.outcome.bytes_offered + result.size,
            bytes_stored=self.outcome.bytes_stored + (result.size if result.created else 0),
            bytes_on_disk=(self.outcome.bytes_on_disk + result.stored_size),
        )
        return result

    def emitter(self, store: ObjectStore) -> Callable[[ObjectName, bytes], None]:
        """Adapt ``put_encoded`` to the shape builders' emit callback."""

        def emit(name: ObjectName, framed: bytes) -> None:
            self.record(store.put_encoded(name, framed))

        return emit

    @property
    def frozen(self) -> IngestStats:
        return self._base + replace(
            self.outcome,
            files=self.files,
            directories=self.directories,
            symlinks=self.symlinks,
            chunks=self.chunks,
        )
