"""Turn a commit back into real files on a host.

This is the dominant read operation. Two things decide whether it
is safe and whether it is fast, and they are separable:

**Safety.** A tree is data supplied by whoever wrote it, and materialization is
the moment that data becomes filesystem operations. The classic exploit is a
path that escapes the destination, so this is defended in depth:

* entry names cannot contain a separator, a NUL, ``.`` or ``..`` — the codec
  rejects them, so a traversal cannot be *expressed* in a valid tree at all;
* everything is written into a fresh staging directory that this process
  creates, so no pre-existing symlink can be traversed on the way in;
* the staging directory is published by a single ``rename``, so an interrupted
  checkout leaves nothing half-written where the caller expects a tree.

**Laziness is not implemented, and that is a deliberate scope decision.**
The fault-in-on-access model is a *latency* optimisation — it makes a
rollout start in 0.4 s instead of 43 s — and it needs a FUSE mount, which needs
macFUSE, which is not present. Everything below is eager. The interface is the
seam: a lazy implementation replaces ``Materializer`` without touching a caller.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, final

from src.errors import InvalidRequest
from src.format.constants import MODE_EXEC, EntryKind
from src.format.model import Commit
from src.fs.blob import BlobReader
from src.fs.tree import iter_entries
from src.metrics import COUNTERS

if TYPE_CHECKING:
    from src.ids import ObjectName
    from src.store.cas import ObjectStore

__all__ = ["MaterializeStats", "Materializer"]


@final
@dataclass(frozen=True, slots=True)
class MaterializeStats:
    files: int = 0
    directories: int = 0
    symlinks: int = 0
    bytes_written: int = 0
    objects_fetched: int = 0


@final
class Materializer:
    """Writes a tree or a commit onto disk.

    Eager. The seam a lazy implementation would slot into is this class, not its
    callers: they ask for a commit at a path and get a usable directory.
    """

    __slots__ = ("_store",)

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def materialize_commit(
        self, commit: ObjectName, destination: Path, *, overwrite: bool = False
    ) -> MaterializeStats:
        """Check out the tree a commit names."""
        return self.materialize_tree(
            self._store.get_as(commit, Commit).tree, destination, overwrite=overwrite
        )

    def materialize_tree(
        self, tree: ObjectName, destination: Path, *, overwrite: bool = False
    ) -> MaterializeStats:
        """Write a tree to ``destination``, atomically.

        Content is assembled in a sibling staging directory and published by one
        rename, so the destination either does not exist or is the complete
        tree — never a partial one that a rollout might start against.
        """
        COUNTERS.increment("ledger_materializations_total")
        destination = Path(destination).absolute()
        if destination.exists() and not overwrite:
            raise InvalidRequest(f"destination already exists: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".ledger-checkout-{destination.name}-", dir=destination.parent)
        )
        try:
            stats = self._write_tree(tree, staging)
            if destination.exists():
                # `os.replace` refuses a non-empty directory, so the old tree is
                # moved aside and removed only after the new one is in place.
                displaced = staging.with_name(staging.name + ".old")
                os.replace(destination, displaced)
                os.replace(staging, destination)
                shutil.rmtree(displaced, ignore_errors=True)
            else:
                os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return stats

    # ── internals ────────────────────────────────────────────────────────────

    def _write_tree(self, tree: ObjectName, directory: Path) -> MaterializeStats:
        directory.mkdir(parents=True, exist_ok=True)
        stats = MaterializeStats(directories=1, objects_fetched=1)

        for entry in iter_entries(self._store, tree):
            # Defence in depth. The codec already rejects these, so reaching
            # here means a tree was constructed by something that bypassed it —
            # which is exactly when a check is worth having.
            name = entry.name.decode()
            if "/" in name or name in (".", "..") or "\x00" in name:
                raise InvalidRequest(f"refusing to materialize unsafe entry name: {name!r}")

            target = directory / name
            match entry.kind:
                case EntryKind.TREE:
                    stats = _merge(stats, self._write_tree(entry.target, target))
                case EntryKind.SYMLINK:
                    link_target = BlobReader(self._store, entry.target).read_all()
                    os.symlink(link_target.decode(), target)
                    stats = replace(
                        stats,
                        symlinks=stats.symlinks + 1,
                        objects_fetched=stats.objects_fetched + 1,
                    )
                case EntryKind.BLOB:
                    written, fetched = self._write_file(entry.target, target, entry.mode)
                    stats = replace(
                        stats,
                        files=stats.files + 1,
                        bytes_written=stats.bytes_written + written,
                        objects_fetched=stats.objects_fetched + fetched,
                    )
                case EntryKind.CONFLICT:  # pragma: no cover - rejected by the codec
                    raise InvalidRequest("a conflicted entry cannot be materialized")
        return stats

    def _write_file(self, blob: ObjectName, path: Path, mode: int) -> tuple[int, int]:
        """Stream a file to disk without holding it in memory.

        ``O_NOFOLLOW`` on the create: the staging directory is ours, so nothing
        should be there — and if something is, following it is the one thing we
        must not do.
        """
        reader = BlobReader(self._store, blob)
        written = 0
        fetched = 1

        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            for payload in reader.stream():
                handle.write(payload)
                written += len(payload)
                fetched += 1

        os.chmod(path, 0o755 if mode == MODE_EXEC else 0o644)
        return written, fetched


def _merge(left: MaterializeStats, right: MaterializeStats) -> MaterializeStats:
    return MaterializeStats(
        files=left.files + right.files,
        directories=left.directories + right.directories,
        symlinks=left.symlinks + right.symlinks,
        bytes_written=left.bytes_written + right.bytes_written,
        objects_fetched=left.objects_fetched + right.objects_fetched,
    )
