"""Compression at rest, as a decorator over any backend.

*"Chunks average ~1 MiB with a 256 KiB floor and are stored
zstd-compressed. **The name is the hash of the uncompressed bytes**, so the
compression codec can change without renaming anything."*

That last clause is the whole design, and wrapping a backend is what makes it
literally true rather than merely intended. Naming happens in ``store.cas``, over
the framed bytes; nothing below this layer has ever seen a name, so nothing below
it can make one depend on how the bytes were packed. Change the codec, or turn
compression off entirely, and every object keeps the name it had.

A frame header records which codec was used, per object. Without it, changing the
codec would require rewriting the corpus, which is the exact coupling the format
avoids — and a corpus is not a thing anyone rewrites.

**Incompressible content is stored raw.** Container layers, model weights and
already-compressed datasets are most of the bytes here, and zstd on them costs
CPU to produce something slightly larger. So the result is kept only if it is
actually smaller, which the header then records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

from src.errors import CorruptObject

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.store.backend import RawBlobBackend

__all__ = ["CODEC_NONE", "CODEC_ZSTD", "CompressedBackend", "compress", "decompress"]

#: One byte in front of every stored object, naming how the rest is packed. A
#: version rather than a build-time setting: two deployments with different
#: settings must still be able to read each other's objects.
CODEC_NONE: Final = 0
CODEC_ZSTD: Final = 1

HEADER_BYTES: Final = 1

#: zstd's default. Level 3 is the point on the curve where the next increment
#: costs materially more CPU for a percent or two of size, and this runs on the
#: write path of a system sized for 2 commits per second.
DEFAULT_LEVEL: Final = 3

#: Below this, the header plus the codec's own framing is a meaningful fraction
#: of the object. Tree nodes and commits live here.
MIN_WORTH_COMPRESSING: Final = 256


def compress(payload: bytes, *, level: int = DEFAULT_LEVEL) -> bytes:
    """Frame ``payload`` for storage, compressing only if that helps."""
    if len(payload) < MIN_WORTH_COMPRESSING:
        return bytes([CODEC_NONE]) + payload

    from compression.zstd import compress as zstd_compress

    packed = zstd_compress(payload, level)
    if len(packed) >= len(payload):
        # Already-compressed content — container layers, model weights, most of
        # the bytes in this corpus. Storing the larger form would be a loss on
        # both axes.
        return bytes([CODEC_NONE]) + payload
    return bytes([CODEC_ZSTD]) + packed


def decompress(stored: bytes) -> bytes:
    """Undo ``compress``. Raises ``CorruptObject`` on an unreadable frame."""
    if not stored:
        raise CorruptObject("stored object is empty", length=0)

    codec = stored[0]
    body = stored[HEADER_BYTES:]
    match codec:
        case 0:
            return body
        case 1:
            from compression.zstd import ZstdError
            from compression.zstd import decompress as zstd_decompress

            try:
                return zstd_decompress(body)
            except ZstdError as exc:
                # Never downgraded to a miss. A miss invites a retry; this means
                # the medium is actively wrong and someone has to look at it.
                raise CorruptObject(
                    "stored object could not be decompressed", error=str(exc)
                ) from exc
        case _:
            raise CorruptObject(
                "stored object uses a compression codec this build cannot read",
                codec=codec,
            )


@final
class CompressedBackend:
    """Any backend, with objects packed on the way in and out.

    Transparent by construction: ``store.cas`` verifies the bytes it gets back
    against the name it asked for, so a codec that returned anything other than
    exactly what was written fails the ordinary delivery check rather than
    quietly serving damaged content.

    ``read_range`` is the one operation that cannot pass through. A ranged read
    of a compressed object has no meaning without decompressing it first, so the
    object is read whole and then sliced. That is honest rather than free — and
    it costs nothing in practice, because ranged reads descend to *chunks*, which
    are capped at 4 MiB precisely so that no single object needs ranged access.
    """

    __slots__ = ("_inner", "_level")

    def __init__(self, inner: RawBlobBackend, *, level: int = DEFAULT_LEVEL) -> None:
        self._inner = inner
        self._level = level

    @property
    def inner(self) -> RawBlobBackend:
        return self._inner

    def write(self, key: str, data: bytes) -> int:
        return self._inner.write(key, compress(data, level=self._level))

    def read(self, key: str) -> bytes:
        return decompress(self._inner.read(key))

    def read_range(self, key: str, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("range offset and length must be non-negative")
        return self.read(key)[offset : offset + length]

    def exists_many(self, keys: Sequence[str]) -> set[str]:
        return self._inner.exists_many(keys)

    def delete(self, keys: Sequence[str]) -> int:
        return self._inner.delete(keys)

    def iter_keys(self) -> Iterator[str]:
        return self._inner.iter_keys()

    def __repr__(self) -> str:
        return f"CompressedBackend({self._inner!r})"
