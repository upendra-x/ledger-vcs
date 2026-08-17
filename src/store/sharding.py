"""Partitioning object names by hash prefix.

Both sides of the collection diff are sharded the same way, and that is the whole
reason the diff can be exact with no Bloom filter anywhere: an object in shard *n*
of the write catalog can only be kept alive by shard *n* of the keep-sets, so one
shard of each fits in memory and the comparison is a set difference.

It lives in its own module because the catalog and the keep-set store both need
it and neither owns it. Previously the keep-set store reached into the catalog for
a private function, which is the sort of coupling that survives right up until
somebody changes one side's definition of a shard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.ids import ObjectName

__all__ = ["MAX_PREFIX_BITS", "shard_bounds", "shard_of"]

#: Three bytes of the digest are read, so twenty-four bits is the ceiling. More
#: than sixteen million shards would be a different problem than this one.
MAX_PREFIX_BITS: Final = 24

_PREFIX_BYTES: Final = 3


def shard_of(name: ObjectName, prefix_bits: int) -> int:
    """Which shard a hash falls in.

    Deterministic, and independent of the shard *count* beyond the bit width — so
    raising the count later re-partitions cleanly instead of reshuffling.
    """
    _check_bits(prefix_bits)
    leading = int.from_bytes(name.digest[:_PREFIX_BYTES], "big")
    return leading >> (MAX_PREFIX_BITS - prefix_bits)


def shard_bounds(prefix_bits: int, shard: int) -> tuple[bytes, bytes]:
    """Half-open digest range covering one shard.

    Byte bounds rather than a computed predicate so the query is a plain index
    range scan — the collector reads a shard sequentially, and a function call
    per row would dominate.
    """
    _check_bits(prefix_bits)
    shard_count = 1 << prefix_bits
    if not 0 <= shard < shard_count:
        raise ValueError(f"shard {shard} out of range for {shard_count} shards")
    width = MAX_PREFIX_BITS - prefix_bits
    low = shard << width
    high = (shard + 1) << width
    return low.to_bytes(_PREFIX_BYTES, "big"), (
        high.to_bytes(_PREFIX_BYTES, "big")
        if high < (1 << MAX_PREFIX_BITS)
        # Past the end of the prefix space. A four-byte sentinel sorts above every
        # three-byte prefix *and* above every full digest that starts with one, so
        # the last shard's scan reaches the end of the index.
        else b"\xff\xff\xff\xff"
    )


def _check_bits(prefix_bits: int) -> None:
    if not 0 < prefix_bits <= MAX_PREFIX_BITS:
        raise ValueError(f"prefix_bits out of range: {prefix_bits}")
