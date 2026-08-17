"""Chunking: the properties that make "an edit costs what changed" true.

The headline behaviour is *resynchronisation*. Insert bytes into the middle of a
file and a content-defined chunker must recover the original boundaries within
one or two chunks, so the tail of the file is unchanged and is not re-stored.
That single property is what turns the cost table from a claim into a
measurement, and it is what every test here is ultimately about.

Tests run at a scaled-down parameter set — the same derivation as production,
1/1024th the size — so they exercise multi-chunk and multi-level behaviour in
kilobytes rather than gigabytes.
"""

from __future__ import annotations

import io
import random

import pytest

from src.format import constants as C
from src.format.cdc import PRODUCTION_PARAMS, ChunkParams, chunk_bytes, chunk_stream, cut_point

#: 1 KiB average: min 256 B, max 4 KiB. Same shape as production, 1024x smaller.
TEST_PARAMS = ChunkParams.for_average(1024)


def random_bytes(size: int, seed: int = 0) -> bytes:
    """Incompressible, reproducible data. A fixed seed keeps failures debuggable."""
    return random.Random(seed).randbytes(size)


def chunk_sizes(data: bytes, params: ChunkParams = TEST_PARAMS) -> list[int]:
    return [len(c) for c in chunk_bytes(data, params)]


def chunk_list(data: bytes, params: ChunkParams = TEST_PARAMS) -> list[bytes]:
    return [bytes(c) for c in chunk_bytes(data, params)]


class TestParameterDerivation:
    def test_production_params_match_the_frozen_constants(self) -> None:
        """One derivation for production and tests, so they cannot drift apart."""
        assert PRODUCTION_PARAMS.min_size == C.MIN_CHUNK_BYTES
        assert PRODUCTION_PARAMS.avg_size == C.AVG_CHUNK_BYTES
        assert PRODUCTION_PARAMS.max_size == C.MAX_CHUNK_BYTES
        assert PRODUCTION_PARAMS.mask_short == C.CUT_MASK_SHORT
        assert PRODUCTION_PARAMS.mask_long == C.CUT_MASK_LONG

    def test_scaled_params_have_the_same_shape(self) -> None:
        assert TEST_PARAMS.min_size == TEST_PARAMS.avg_size // 4
        assert TEST_PARAMS.max_size == TEST_PARAMS.avg_size * 4
        assert TEST_PARAMS.mask_short.bit_count() > TEST_PARAMS.mask_long.bit_count()

    def test_normalization_makes_the_pre_average_mask_harder(self) -> None:
        """Inverted masks would widen the distribution instead of tightening it,
        and no other test would notice.
        """
        assert TEST_PARAMS.mask_short.bit_count() == TEST_PARAMS.mask_long.bit_count() + 4

    @pytest.mark.parametrize("avg", [7, 1000, 0, -8])
    def test_rejects_non_power_of_two_averages(self, avg: int) -> None:
        with pytest.raises(ValueError, match="power of two"):
            ChunkParams.for_average(avg)

    def test_rejects_inverted_masks(self) -> None:
        with pytest.raises(ValueError, match="strictly harder"):
            ChunkParams(min_size=1, avg_size=2, max_size=4, mask_short=0b11, mask_long=0b111)


class TestChunkBoundaries:
    def test_sizes_stay_within_the_configured_bounds(self) -> None:
        sizes = chunk_sizes(random_bytes(200_000))
        assert all(TEST_PARAMS.min_size <= s <= TEST_PARAMS.max_size for s in sizes[:-1])
        assert sizes[-1] <= TEST_PARAMS.max_size

    def test_chunks_reassemble_into_the_original(self) -> None:
        data = random_bytes(200_000)
        assert b"".join(chunk_list(data)) == data

    def test_mean_size_is_near_the_target(self) -> None:
        """Normalization exists to make this true; without it the mean drifts and
        the object count for a given corpus drifts with it.
        """
        sizes = chunk_sizes(random_bytes(2_000_000, seed=1))
        mean = sum(sizes) / len(sizes)
        assert TEST_PARAMS.avg_size * 0.6 < mean < TEST_PARAMS.avg_size * 1.6, mean

    def test_chunking_is_deterministic(self) -> None:
        data = random_bytes(100_000)
        assert chunk_sizes(data) == chunk_sizes(data)

    def test_boundaries_depend_only_on_local_content(self) -> None:
        """The property the whole design rests on: identical regions of two
        different files chunk identically, which is what lets unrelated
        environments share chunks.
        """
        shared = random_bytes(50_000, seed=7)
        first = random_bytes(3_000, seed=1) + shared
        second = random_bytes(5_000, seed=2) + shared

        chunks_a = chunk_list(first)
        chunks_b = chunk_list(second)
        assert set(chunks_a) & set(chunks_b), "no chunk was shared between the two files"


class TestResynchronisation:
    """An insertion must shift only the chunk it landed in."""

    def test_insertion_in_the_middle_changes_few_chunks(self) -> None:
        original = random_bytes(400_000, seed=3)
        midpoint = len(original) // 2
        edited = original[:midpoint] + b"INSERTED PAYLOAD" + original[midpoint:]

        before = chunk_list(original)
        after = chunk_list(edited)

        changed = len(set(after) - set(before))
        assert changed <= 3, f"{changed} new chunks for a 16-byte insertion"

    def test_the_tail_after_an_insertion_is_untouched(self) -> None:
        """Fixed-size blocks would re-store everything after the edit. This is
        the test that would fail if chunking silently reverted to fixed blocks.
        """
        original = random_bytes(400_000, seed=4)
        edited = original[:1000] + b"x" * 64 + original[1000:]

        before = chunk_list(original)
        after = chunk_list(edited)

        shared = set(before) & set(after)
        shared_bytes = sum(len(c) for c in before if c in shared)
        assert shared_bytes > len(original) * 0.9, (
            f"only {shared_bytes}/{len(original)} bytes were reused after a "
            f"64-byte insertion near the start"
        )

    def test_appending_reuses_every_earlier_chunk(self) -> None:
        """Appending 8 MiB to a 40 GiB file costs ~9 MiB.

        Every chunk except the last is bit-for-bit reusable, because nothing
        before the append moved.
        """
        original = random_bytes(300_000, seed=5)
        extended = original + random_bytes(50_000, seed=6)

        before = chunk_list(original)
        after = chunk_list(extended)

        assert after[: len(before) - 1] == before[:-1]

    def test_overwrite_in_place_touches_only_the_local_region(self) -> None:
        original = random_bytes(400_000, seed=8)
        offset = 200_000
        edited = bytearray(original)
        edited[offset : offset + 2000] = random_bytes(2000, seed=9)

        before = chunk_list(original)
        after = chunk_list(bytes(edited))

        new_bytes = sum(len(c) for c in after if c not in set(before))
        assert new_bytes < 20_000, f"{new_bytes} new bytes for a 2 KB overwrite"


class TestEdgeCases:
    def test_empty_input_yields_no_chunks(self) -> None:
        assert chunk_list(b"") == []

    def test_input_below_the_minimum_is_one_chunk(self) -> None:
        data = random_bytes(TEST_PARAMS.min_size - 1)
        assert chunk_list(data) == [data]

    def test_input_exactly_at_the_minimum_is_one_chunk(self) -> None:
        data = random_bytes(TEST_PARAMS.min_size)
        assert chunk_list(data) == [data]

    def test_the_maximum_is_enforced_when_no_boundary_is_ever_found(self) -> None:
        """The maximum bounds the cost of one fetch and the memory a
        single chunk can occupy, and it has to hold even if the mask never matches.

        Constant input does *not* exercise this — a run of identical bytes still
        drives the gear hash through a deterministic sequence that hits the mask.
        So the clamp is tested with masks that can essentially never match, which
        is the only way to reach the fallback return.
        """
        unmatchable = ChunkParams(
            min_size=TEST_PARAMS.min_size,
            avg_size=TEST_PARAMS.avg_size,
            max_size=TEST_PARAMS.max_size,
            mask_short=(1 << 64) - 1,  # requires the fingerprint to be exactly 0
            mask_long=(1 << 63) - 1,
        )
        data = random_bytes(TEST_PARAMS.max_size * 3, seed=31)
        assert cut_point(data, unmatchable) == unmatchable.max_size
        assert all(s == unmatchable.max_size for s in chunk_sizes(data, unmatchable)[:-1])

    def test_compressible_input_still_respects_the_bounds(self) -> None:
        zeros = b"\x00" * (TEST_PARAMS.max_size * 3)
        sizes = chunk_sizes(zeros)
        assert all(TEST_PARAMS.min_size <= s <= TEST_PARAMS.max_size for s in sizes[:-1])
        assert sum(sizes) == len(zeros)

    def test_uniform_data_chunks_identically_everywhere(self) -> None:
        """A zero-filled region produces identical chunks, so a sparse dataset
        deduplicates against itself — a real property for ML artifacts.
        """
        chunks = chunk_list(b"\x00" * (TEST_PARAMS.max_size * 4))
        assert len(set(chunks[:-1])) == 1

    def test_single_byte_input(self) -> None:
        assert chunk_list(b"z") == [b"z"]

    def test_cut_point_on_an_empty_buffer(self) -> None:
        assert cut_point(b"", TEST_PARAMS) == 0


class TestStreaming:
    def test_stream_matches_whole_buffer(self) -> None:
        """Streaming must not change a single boundary — otherwise a file chunked
        by upload and the same file chunked from disk would get different names.
        """
        data = random_bytes(500_000, seed=11)
        streamed = list(chunk_stream(io.BytesIO(data), TEST_PARAMS))
        assert streamed == chunk_list(data)

    @pytest.mark.parametrize("read_size", [1, 7, 256, 4096, 1 << 20])
    def test_stream_is_independent_of_the_read_size(self, read_size: int) -> None:
        """A short read from a socket must not move a boundary."""
        data = random_bytes(120_000, seed=12)
        streamed = list(chunk_stream(io.BytesIO(data), TEST_PARAMS, read_size=read_size))
        assert streamed == chunk_list(data)

    def test_stream_reassembles(self) -> None:
        data = random_bytes(300_000, seed=13)
        assert b"".join(chunk_stream(io.BytesIO(data), TEST_PARAMS)) == data

    def test_empty_stream(self) -> None:
        assert list(chunk_stream(io.BytesIO(b""), TEST_PARAMS)) == []

    def test_stream_does_not_hold_the_whole_input(self) -> None:
        """Chunking a 40 GiB dataset must not need 40 GiB of memory.

        Asserted by construction rather than by measurement: the reader is only
        ever asked for bounded blocks, and the buffer is drained as chunks are
        emitted.
        """
        size = 400_000
        reads: list[int] = []

        class CountingReader(io.RawIOBase):
            def __init__(self) -> None:
                self._remaining = size

            def read(self, n: int = -1, /) -> bytes:
                take = min(n if n > 0 else 0, self._remaining)
                reads.append(take)
                self._remaining -= take
                return random_bytes(take, seed=14) if take else b""

        list(chunk_stream(CountingReader(), TEST_PARAMS))
        assert max(reads) <= max(TEST_PARAMS.max_size * 4, 1 << 20)


class TestProductionScale:
    """A few assertions at the real parameters, so the frozen values are exercised."""

    def test_production_chunking_of_a_realistic_buffer(self) -> None:
        data = random_bytes(8 * 1024 * 1024, seed=21)
        sizes = chunk_sizes(data, PRODUCTION_PARAMS)
        assert sum(sizes) == len(data)
        assert all(s <= C.MAX_CHUNK_BYTES for s in sizes)
        assert all(s >= C.MIN_CHUNK_BYTES for s in sizes[:-1])

    def test_production_mean_is_near_one_mebibyte(self) -> None:
        data = random_bytes(32 * 1024 * 1024, seed=22)
        sizes = chunk_sizes(data, PRODUCTION_PARAMS)
        mean = sum(sizes) / len(sizes)
        assert C.AVG_CHUNK_BYTES * 0.6 < mean < C.AVG_CHUNK_BYTES * 1.6, mean
