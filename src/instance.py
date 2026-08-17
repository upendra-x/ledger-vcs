"""The composition root for a local Ledger.

One place assembles the whole system, so nothing else has to know how the pieces
fit — and, more importantly, so nothing else can assemble a *partial* one. The
object store takes no default for its tombstone store precisely so that this is
the only spot where that choice is made, and this is the spot that refuses the
null one.

Layout under a data directory::

    objects/ content, sharded two levels by hash prefix
    catalog.db      the write catalog — the "stored" side of the GC diff
    keepsets.db     per-environment keep-sets — the "live" side
    tombstones.db   swept hashes, so dedup cannot resurrect them
    digests.db      SHA-256 → object name, for content named by another hash
    meta/           the metadata shards and the global keyspaces

The two stores have **no dependency on each other**: the object
store does not know what a ref is, and the metadata store never holds content.
That is what makes degrading one leave the other intact — and it is why the
99% read-only corpus stays readable through a total outage of the mutable side.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, final

from src.clock import SystemClock
from src.format.cdc import PRODUCTION_PARAMS
from src.format.shape import PRODUCTION_SHAPE
from src.maintenance.gc import GarbageCollector, GcConfig
from src.meta.repository import MetadataRepository
from src.meta.store import ShardedSqliteMetadataStore
from src.runtime.ingest import Ingester
from src.runtime.materialize import Materializer
from src.store.backend import LocalFsBackend
from src.store.cas import ObjectStore
from src.store.catalog import SqliteWriteCatalog
from src.store.compress import CompressedBackend
from src.store.digests import SqliteDigestIndex
from src.store.keepsets import SqliteKeepSetStore
from src.store.tombstone import NullTombstoneStore, SqliteTombstoneStore

if TYPE_CHECKING:
    from src.clock import Clock
    from src.format.cdc import ChunkParams
    from src.format.shape import ShapeParams

__all__ = ["Ledger"]

#: Shards for the metadata store. Sixteen is plenty for one machine and is
#: enough to make per-environment isolation observable rather than theoretical.
DEFAULT_SHARD_COUNT = 16


@final
class Ledger:
    """An open Ledger instance: both stores, wired and ready."""

    __slots__ = (
        "_catalog",
        "_chunk_params",
        "_clock",
        "_digests",
        "_gc",
        "_ingester",
        "_keepsets",
        "_materializer",
        "_meta",
        "_repo",
        "_root",
        "_shape_params",
        "_store",
        "_tombstones",
    )

    def __init__(
        self,
        root: Path | str,
        *,
        clock: Clock | None = None,
        chunk_params: ChunkParams = PRODUCTION_PARAMS,
        shape_params: ShapeParams = PRODUCTION_SHAPE,
        shard_count: int = DEFAULT_SHARD_COUNT,
        gc_config: GcConfig | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or SystemClock()
        self._chunk_params = chunk_params
        self._shape_params = shape_params

        self._catalog = SqliteWriteCatalog.open(self._root / "catalog.db")
        self._tombstones = SqliteTombstoneStore.open(self._root / "tombstones.db")
        self._store = ObjectStore(
            # Objects are packed on the way to disk and unpacked on the way
            # back. Naming happens above this, over the *uncompressed* bytes,
            # so the codec can change without renaming anything.
            CompressedBackend(LocalFsBackend(self._root / "objects")),
            catalog=self._catalog,
            tombstones=self._tombstones,
            clock=self._clock,
        )

        # The guard that keeps the anti-resurrection guard. A no-op tombstone
        # store has no symptom until a rollout fails half an hour in, so refuse
        # it here rather than discovering it in production.
        if isinstance(self._store.tombstones, NullTombstoneStore):  # pragma: no cover
            raise RuntimeError("a null tombstone store must never be wired into a real Ledger")

        self._meta = ShardedSqliteMetadataStore(self._root / "meta", shard_count=shard_count)
        self._repo = MetadataRepository(self._meta, clock=self._clock)

        self._ingester = Ingester(
            self._store,
            clock=self._clock,
            chunk_params=chunk_params,
            shape_params=shape_params,
        )
        self._materializer = Materializer(self._store)

        self._keepsets = SqliteKeepSetStore.open(self._root / "keepsets.db")
        self._digests = SqliteDigestIndex.open(self._root / "digests.db")
        self._gc = GarbageCollector(
            self._store,
            self._keepsets,
            self._repo,
            clock=self._clock,
            digests=self._digests,
            config=gc_config,
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def store(self) -> ObjectStore:
        return self._store

    @property
    def meta(self) -> ShardedSqliteMetadataStore:
        return self._meta

    @property
    def repo(self) -> MetadataRepository:
        return self._repo

    @property
    def ingester(self) -> Ingester:
        return self._ingester

    @property
    def materializer(self) -> Materializer:
        return self._materializer

    @property
    def keepsets(self) -> SqliteKeepSetStore:
        return self._keepsets

    @property
    def digests(self) -> SqliteDigestIndex:
        """SHA-256 → object name, for content that arrived named by another hash.

        A *hint*, never an existence predicate: see ``store.digests``. Nothing
        reads it to decide whether an object is present — that is
        ``ObjectStore.missing`` and only ``ObjectStore.missing``.
        """
        return self._digests

    @property
    def gc(self) -> GarbageCollector:
        """The collector. Read-only by default — ``run(enforce=True)`` is the
        only thing here that deletes anything.
        """
        return self._gc

    @property
    def chunk_params(self) -> ChunkParams:
        return self._chunk_params

    @property
    def shape_params(self) -> ShapeParams:
        """Exposed because anything that *builds* a tree — a merge, an import —
        must use the same shape the corpus was written with, or it silently
        produces a differently-named tree for identical content.
        """
        return self._shape_params

    def close(self) -> None:
        self._keepsets.close()
        self._digests.close()
        self._meta.close()
        self._catalog.close()
        self._tombstones.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Ledger({self._root})"
