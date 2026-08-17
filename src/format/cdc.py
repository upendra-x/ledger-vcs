"""FastCDC — content-defined chunking, and the reason a 1 MiB edit costs 1 MiB.

A 40 GiB dataset with 2 MiB of edits should cost 2 MiB. That requires splitting
files into pieces, and *where* the splits go decides whether it works:

    original           [---- A ----][---- B ----][---- C ----][---- D ----]

    insert 10 bytes at the start of B

    fixed blocks       [---- A ----][-- xB' ---][-- C' ---][-- D' ---]
                       every later boundary shifts  →  B, C, D all re-stored

    content-defined    [---- A ----][---- xB' ----][---- C ----][---- D ----]
                       boundary re-syncs after one chunk  →  only B' is new

A rolling gear hash over a sliding window declares a boundary wherever the hash
matches a mask, so boundaries follow the *content*: an insertion shifts only the
chunk it landed in.

**Normalized chunking, level 2.** Plain CDC produces a geometric size
distribution with a long tail — many tiny chunks and a hard ceiling of clipped
maximum-size ones, both of which hurt. Normalization uses a *harder* mask before
the target average (making an early cut unlikely) and an *easier* one after it
(making a late cut likely), which pulls the distribution towards the average.

**Parameters are injected, not read from module scope.** Production uses 256 KiB
/ 1 MiB / 4 MiB, but a test that had to write 40 MiB to observe three chunks
would be too slow to run often, and a slow test does not get run. ``ChunkParams``
scales the whole parameter set from one number, and the production instance is
derived by the same function as the test ones — so there is one derivation to be
right about rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, Self, final

from src.format.constants import (
    AVG_CHUNK_BYTES,
    CUT_MASK_LONG,
    CUT_MASK_SHORT,
    GEAR,
    MAX_CHUNK_BYTES,
    MIN_CHUNK_BYTES,
)

if TYPE_CHECKING:
    from collections.abc import Buffer, Iterator

__all__ = [
    "PRODUCTION_PARAMS",
    "ByteReader",
    "ChunkParams",
    "chunk_bytes",
    "chunk_stream",
    "cut_point",
]


class ByteReader(Protocol):
    """The only thing ``chunk_stream`` needs from its source.

    Narrower than ``BinaryIO`` on purpose: an upload arrives as a socket, a
    dataset as a file, and a migrated blob as a git object stream, and none of
    those should have to pretend to be seekable to be chunked.
    """

    def read(self, size: int = ..., /) -> bytes: ...


_MASK64: Final = 0xFFFF_FFFF_FFFF_FFFF

#: Distance from the average, in bits, that the two masks sit at. Level 2 is the
#: FastCDC paper's recommendation and is frozen alongside the masks it produces.
_NORMALIZATION_LEVEL: Final = 2


@final
@dataclass(frozen=True, slots=True)
class ChunkParams:
    """The complete set of values that decide where chunk boundaries fall.

    Two different parameter sets produce different boundaries and therefore
    different names for the same file. That is correct and expected — it is why
    the production set is frozen in ``format.constants`` — but it means a
    ``ChunkParams`` must be threaded through explicitly rather than defaulted at
    each call site, so nothing can silently chunk with the wrong one.
    """

    min_size: int
    avg_size: int
    max_size: int
    mask_short: int
    mask_long: int

    def __post_init__(self) -> None:
        if not 0 < self.min_size <= self.avg_size <= self.max_size:
            raise ValueError(
                f"chunk sizes must satisfy 0 < min <= avg <= max, got "
                f"{self.min_size}/{self.avg_size}/{self.max_size}"
            )
        if self.mask_short.bit_count() <= self.mask_long.bit_count():
            raise ValueError(
                "the pre-average mask must be strictly harder than the post-average "
                "one, or normalization widens the size distribution instead of "
                "tightening it"
            )

    @classmethod
    def for_average(cls, avg_size: int) -> Self:
        """Derive a full parameter set from the target average chunk size.

        One derivation, used for both production and test scales, so there is a
        single place where the relationship between average size, the min/max
        clamps and the two masks is expressed.
        """
        if avg_size < 8 or avg_size.bit_count() != 1:
            raise ValueError(f"average chunk size must be a power of two >= 8, got {avg_size}")
        avg_bits = avg_size.bit_length() - 1
        return cls(
            min_size=avg_size // 4,
            avg_size=avg_size,
            max_size=avg_size * 4,
            mask_short=_mask_with_bits(avg_bits + _NORMALIZATION_LEVEL),
            mask_long=_mask_with_bits(avg_bits - _NORMALIZATION_LEVEL),
        )


def _mask_with_bits(bits: int) -> int:
    """The top ``bits`` bits of a 64-bit word.

    High bits, not low. In the gear hash ``h_i = Σ_{j<64} G[b_{i-j}] << j``, bit
    *k* receives contributions only from bytes within distance *k* — so the high
    bits are the ones that depend on the whole window. A low-bit mask would make
    a boundary depend on a handful of recent bytes.
    """
    if not 1 <= bits <= 63:
        raise ValueError(f"mask width out of range: {bits}")
    return ((1 << bits) - 1) << (64 - bits)


#: The frozen production parameters. Asserted to match ``format.constants``, so
#: the derivation above and the pinned values can never drift apart.
PRODUCTION_PARAMS: Final = ChunkParams.for_average(AVG_CHUNK_BYTES)

assert PRODUCTION_PARAMS.min_size == MIN_CHUNK_BYTES, "derived min chunk size drifted"
assert PRODUCTION_PARAMS.max_size == MAX_CHUNK_BYTES, "derived max chunk size drifted"
assert PRODUCTION_PARAMS.mask_short == CUT_MASK_SHORT, "derived short mask drifted"
assert PRODUCTION_PARAMS.mask_long == CUT_MASK_LONG, "derived long mask drifted"


def cut_point(data: Buffer, params: ChunkParams) -> int:
    """Length of the first chunk of ``data``, following FastCDC.

    Returns the index at which the chunk ends, exclusive — so the byte whose
    hash matched the mask begins the *next* chunk. The first ``min_size`` bytes
    are never hashed, which is FastCDC's cut-point skipping: it removes a
    quarter of the hashing work and cannot cost a boundary, because a boundary
    there would be rejected by the minimum anyway.

    This is the reference implementation and the definition of correctness. A
    faster one may replace it only if a differential test proves the boundaries
    are identical — a divergence of one cut in ten thousand would be invisible
    in tests and would quietly halve deduplication in production.
    """
    view = memoryview(data)
    length = len(view)
    if length <= params.min_size:
        return length

    end = min(length, params.max_size)
    normal = min(params.avg_size, end)

    gear = GEAR  # hoisted: this loop runs once per byte of the corpus
    mask_short = params.mask_short
    mask_long = params.mask_long

    fingerprint = 0
    index = params.min_size

    while index < normal:
        fingerprint = ((fingerprint << 1) + gear[view[index]]) & _MASK64
        if not fingerprint & mask_short:
            return index
        index += 1

    while index < end:
        fingerprint = ((fingerprint << 1) + gear[view[index]]) & _MASK64
        if not fingerprint & mask_long:
            return index
        index += 1

    return end


def chunk_bytes(data: Buffer, params: ChunkParams) -> Iterator[memoryview]:
    """Split an in-memory buffer into chunks.

    Yields memoryviews rather than copies, so chunking a large buffer does not
    double its memory. Callers that keep a chunk beyond the next iteration must
    materialise it with ``bytes()``.
    """
    view = memoryview(data)
    offset = 0
    total = len(view)
    while offset < total:
        size = cut_point(view[offset:], params)
        yield view[offset : offset + size]
        offset += size


def chunk_stream(
    source: ByteReader, params: ChunkParams, *, read_size: int | None = None
) -> Iterator[bytes]:
    """Split a stream of unknown length into chunks, without holding it in memory.

    The buffer is refilled whenever it holds less than ``max_size``, because a
    cut point can never be found beyond that — so a full buffer is always enough
    to decide the next boundary correctly. That is what makes the streaming
    output byte-identical to chunking the whole input at once, which
    ``test_stream_matches_whole_buffer`` pins.
    """
    window = read_size if read_size is not None else max(params.max_size * 4, 1 << 20)
    buffer = bytearray()
    exhausted = False

    while True:
        while not exhausted and len(buffer) < params.max_size:
            block = source.read(window)
            if not block:
                exhausted = True
                break
            buffer += block

        if not buffer:
            return

        # Once the source is exhausted the tail is emitted as-is, however short:
        # a final chunk below the minimum is normal and is not a boundary
        # decision, it is simply the end of the file.
        size = cut_point(buffer, params)
        yield bytes(buffer[:size])
        del buffer[:size]

        if exhausted and not buffer:
            return
