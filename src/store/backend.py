"""Raw byte storage, deliberately unaware of object names.

This is the anti-bypass mechanism, and the reason it is a separate layer at all.

Bytes are verified against their name on ingest and again on
delivery. A rule like that survives only if it cannot be routed around, and the
usual way it gets routed around is a well-meaning caller reaching past the
verifying layer for "just this one hot path". So the backend is given no way to
express *fetch the object called X*: it speaks opaque string keys, and the
translation from an ``ObjectName`` to a key happens in exactly one place
(``store.cas``), which always verifies.

Two implementations ship. ``LocalFsBackend`` is the real one; ``InMemoryBackend``
exists because a test that has to touch a filesystem to check a hash comparison
is a slow test, and slow tests get skipped.

The seam this preserves is S3: ``write``/``read``/``exists_many``/``delete`` map
onto PutObject/GetObject/HeadObject/DeleteObjects with no impedance, and
``read_range`` onto a ranged GET — which is what is needed to read a
megabyte out of the middle of a forty-gigabyte file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from src.errors import BackendUnavailable, ObjectNotFound

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = ["InMemoryBackend", "LocalFsBackend", "RawBlobBackend"]


@runtime_checkable
class RawBlobBackend(Protocol):
    """Opaque key/value bytes. Never content-aware — see the module docstring."""

    def write(self, key: str, data: bytes) -> int:
        """Store ``data`` under ``key``, returning the bytes it actually occupies.

        Idempotence matters more than it looks: two writers uploading the same
        object concurrently is the normal case under global deduplication, and
        neither must observe a torn read while the other is writing.

        The return value is not ``len(data)`` in general — a backend that
        compresses at rest (``store.compress``) stores fewer. Reporting it here
        is what lets the corpus be costed on bytes that exist rather than on
        bytes that were offered.
        """
        ...

    def read(self, key: str) -> bytes:
        """Raise ``ObjectNotFound`` if absent."""
        ...

    def read_range(self, key: str, offset: int, length: int) -> bytes: ...

    def exists_many(self, keys: Sequence[str]) -> set[str]:
        """Which of ``keys`` are present. Batched, because the write path asks
        about a thousand at a time.
        """
        ...

    def delete(self, keys: Sequence[str]) -> int:
        """Remove keys, returning how many existed. Absent keys are not an error —
        garbage collection is retried and must be idempotent.
        """
        ...

    def iter_keys(self) -> Iterator[str]:
        """Every key present. Used only for index rebuilds and audits, never on
        the request path.
        """
        ...


@final
class LocalFsBackend:
    """Files on disk, sharded two levels deep by key prefix.

    The layout mirrors A standalone object's key is
    derived from its
    own hash, so *locating it requires no lookup at all* — the filesystem's own
    path resolution is the index. Two levels of two hex characters give ~65,000
    directories, which is what keeps any single directory small and, in the
    object-store analogue, spreads load across prefixes so no one prefix is hot
    when a cache tier is lost.
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, key: str) -> Path:
        if len(key) < 4 or "/" in key or "\\" in key or ".." in key:
            raise ValueError(f"unsafe backend key: {key!r}")
        return self._root / key[:2] / key[2:4] / key

    def write(self, key: str, data: bytes) -> int:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write-then-rename, so a reader never sees a partial object and a crash
        # mid-write leaves a temp file rather than a truncated one. Immutability
        # makes the overwrite case harmless: the bytes are identical by
        # construction, because the key is derived from them.
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except OSError as exc:
            Path(temp_name).unlink(missing_ok=True)
            raise BackendUnavailable(f"could not write object: {exc}", key=key) from exc
        return len(data)

    def read(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise ObjectNotFound("object is not present in the store", key=key) from exc
        except OSError as exc:
            raise BackendUnavailable(f"could not read object: {exc}", key=key) from exc

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("range offset and length must be non-negative")
        try:
            with self._path(key).open("rb") as handle:
                handle.seek(offset)
                return handle.read(length)
        except FileNotFoundError as exc:
            raise ObjectNotFound("object is not present in the store", key=key) from exc
        except OSError as exc:
            raise BackendUnavailable(f"could not read object: {exc}", key=key) from exc

    def exists_many(self, keys: Sequence[str]) -> set[str]:
        return {key for key in keys if self._path(key).exists()}

    def delete(self, keys: Sequence[str]) -> int:
        removed = 0
        for key in keys:
            try:
                self._path(key).unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise BackendUnavailable(f"could not delete object: {exc}", key=key) from exc
            removed += 1
        return removed

    def iter_keys(self) -> Iterator[str]:
        for path in self._root.rglob("*"):
            if path.is_file() and not path.name.startswith(".tmp-"):
                yield path.name

    def __repr__(self) -> str:
        return f"LocalFsBackend({self._root})"


@final
class InMemoryBackend:
    """A dict. For tests, and for exercising the store without a filesystem."""

    __slots__ = ("_data",)

    def __init__(self, initial: Iterable[tuple[str, bytes]] = ()) -> None:
        self._data: dict[str, bytes] = dict(initial)

    def write(self, key: str, data: bytes) -> int:
        self._data[key] = data
        return len(data)

    def read(self, key: str) -> bytes:
        try:
            return self._data[key]
        except KeyError as exc:
            raise ObjectNotFound("object is not present in the store", key=key) from exc

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        return self.read(key)[offset : offset + length]

    def exists_many(self, keys: Sequence[str]) -> set[str]:
        return {key for key in keys if key in self._data}

    def delete(self, keys: Sequence[str]) -> int:
        return sum(self._data.pop(key, None) is not None for key in keys)

    def iter_keys(self) -> Iterator[str]:
        return iter(list(self._data))

    def corrupt(self, key: str, data: bytes) -> None:
        """Overwrite bytes without changing the key.

        Only a test would want this, and only to prove that delivery
        verification catches it — which is exactly why it exists here rather
        than being simulated with mocks.
        """
        self._data[key] = data

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"InMemoryBackend({len(self._data)} objects)"
