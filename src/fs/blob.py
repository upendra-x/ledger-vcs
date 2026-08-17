"""Reading files: byte-range descent through blob index nodes.

Reading a megabyte out of the middle of a forty-gigabyte file costs six object
fetches, and the count is set by *depth* rather than by size — the same read
against a one-terabyte file costs seven.

What makes it work is that every blob entry carries the byte span its child
covers, so descending to the node containing offset *X* is a search over
cumulative sizes. No child is fetched to discover how much of the file it holds,
which is the difference between O(depth) and O(chunks).

    ReadFile(c9f2…, "data/train.bin", bytes 4 GiB … 4 GiB + 1 MiB)

      commit → tree → tree → blob manifest → index node → chunk
                                                          ─────
                                                          6 objects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from src.errors import InvalidRequest
from src.format.model import Blob, Chunk

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.ids import ObjectName
    from src.store.cas import ObjectStore

__all__ = ["BlobReader"]


@final
@dataclass(frozen=True, slots=True)
class _Located:
    """A chunk, and where in the file it sits."""

    name: ObjectName
    start: int
    size: int


@final
class BlobReader:
    """Random access over one file.

    Holds the root node only. Interior nodes are fetched on the way down and not
    cached here, because caching belongs to the store and the host — putting a
    cache in the reader would give each open file its own, which is exactly the
    duplication that keying everything on content avoids.
    """

    __slots__ = ("_root", "_root_name", "_store")

    def __init__(self, store: ObjectStore, name: ObjectName) -> None:
        self._store = store
        self._root_name = name
        self._root = store.get_as(name, Blob)

    @property
    def name(self) -> ObjectName:
        return self._root_name

    @property
    def size(self) -> int:
        """Total bytes. Derived by summing the root's entries — never stored,
        because a stored total would be a second way to say the same thing.
        """
        return self._root.size

    def read(self, offset: int = 0, length: int | None = None) -> bytes:
        """Read a byte range.

        A range beyond the end is clamped rather than an error, matching what a
        file read does — a caller asking for more than exists gets what exists.
        """
        return b"".join(self.stream(offset, length))

    def read_all(self) -> bytes:
        """The whole file. Only for content known to be small — an environment
        manifest, a symlink target, an OCI image index.
        """
        return self.read()

    def stream(self, offset: int = 0, length: int | None = None) -> Iterator[bytes]:
        """Yield a byte range in order, without holding it in memory.

        This is what materialization uses, and it is why a 40 GiB dataset can be
        checked out on a host with 8 GiB of RAM. It is also how the registry
        serves a layer and how a ranged HTTP read is answered: the descent
        prunes to the requested span, so resuming a download at 3 GiB does not
        read the first three gigabytes to get there.

        Arguments are validated here rather than inside the generator, so a bad
        offset is an error at the call rather than at the first ``next()`` —
        which in a streaming response is after the status line has been sent.
        """
        if offset < 0:
            raise InvalidRequest("read offset must be non-negative", offset=offset)
        if length is not None and length < 0:
            raise InvalidRequest("read length must be non-negative", length=length)

        total = self.size
        if offset >= total:
            return iter(())
        end = total if length is None else min(offset + length, total)
        return self._yield_range(offset, end)

    def _yield_range(self, offset: int, end: int) -> Iterator[bytes]:
        for located in self._chunks_covering(offset, end):
            data = self._store.get_as(located.name, Chunk).data
            begin = max(offset - located.start, 0)
            stop = min(end - located.start, located.size)
            # Whole chunks are yielded without copying; only the chunks at the
            # two ends of the range are ever sliced.
            yield data if (begin, stop) == (0, located.size) else data[begin:stop]

    def chunk_names(self) -> Iterator[ObjectName]:
        """Every chunk this file is made of, in order.

        Used by the garbage collector's closure walk, which must *record* chunk
        digests without fetching the chunks themselves.
        """
        for located in self._walk(self._root, 0):
            yield located.name

    # ── descent ──────────────────────────────────────────────────────────────

    def _chunks_covering(self, start: int, end: int) -> Iterator[_Located]:
        """Chunks overlapping [start, end), fetching only nodes on the path.

        The pruning is the whole point: a subtree whose span lies entirely
        outside the requested range is skipped without being fetched, so a
        one-megabyte read of a forty-gigabyte file touches one leaf path.
        """
        if start >= end:
            return
        yield from self._descend(self._root, 0, start, end)

    def _descend(self, node: Blob, base: int, start: int, end: int) -> Iterator[_Located]:
        offset = base
        for entry in node.entries:
            child_end = offset + entry.size
            if child_end > start and offset < end:
                if node.level == 0:
                    yield _Located(entry.target, offset, entry.size)
                else:
                    child = self._store.get_as(entry.target, Blob)
                    yield from self._descend(child, offset, start, end)
            offset = child_end
            if offset >= end:
                return

    def _walk(self, node: Blob, base: int) -> Iterator[_Located]:
        offset = base
        for entry in node.entries:
            if node.level == 0:
                yield _Located(entry.target, offset, entry.size)
            else:
                yield from self._walk(self._store.get_as(entry.target, Blob), offset)
            offset += entry.size
