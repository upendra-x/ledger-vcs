"""Compression at rest, and the one thing it must never touch.

*"The name is the hash of the uncompressed bytes, so the compression
codec can change without renaming anything."* That sentence is the entire reason
compression is a decorator over a backend rather than a step inside the store,
and ``test_the_codec_does_not_change_any_name`` is the sentence as an assertion.

Getting it wrong would be the worst class of bug this system has: names would
depend on a deployment setting, two deployments would stop deduplicating against
each other, and nothing would report an error — the corpus would just quietly
double.
"""

from __future__ import annotations

import random
import zlib
from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.errors import CorruptObject
from src.format.codec import encode, name_of
from src.format.model import Chunk
from src.store.backend import InMemoryBackend
from src.store.cas import ObjectStore
from src.store.catalog import InMemoryWriteCatalog
from src.store.compress import (
    CODEC_NONE,
    CODEC_ZSTD,
    CompressedBackend,
    compress,
    decompress,
)
from src.store.tombstone import SqliteTombstoneStore

if TYPE_CHECKING:
    from src.store.backend import RawBlobBackend

#: Text-like, so it genuinely compresses.
COMPRESSIBLE = b"the quick brown fox jumps over the lazy dog\n" * 500
#: Random, so it genuinely does not — which is what container layers and model
#: weights look like, and they are most of this corpus.
INCOMPRESSIBLE = random.Random(4).randbytes(200_000)


def store_over(backend: RawBlobBackend) -> ObjectStore:
    return ObjectStore(
        backend,
        catalog=InMemoryWriteCatalog(),
        tombstones=SqliteTombstoneStore.open(":memory:"),
        clock=ManualClock(),
    )


class TestFraming:
    def test_round_trips(self) -> None:
        assert decompress(compress(COMPRESSIBLE)) == COMPRESSIBLE
        assert decompress(compress(INCOMPRESSIBLE)) == INCOMPRESSIBLE
        assert decompress(compress(b"")) == b""

    def test_compressible_content_is_compressed(self) -> None:
        packed = compress(COMPRESSIBLE)
        assert packed[0] == CODEC_ZSTD
        assert len(packed) < len(COMPRESSIBLE) // 2

    def test_incompressible_content_is_stored_raw(self) -> None:
        """Storing a *larger* object to claim compression would lose on both axes.

        This is the common case here, not the exotic one: container layers and
        model weights are already compressed.
        """
        packed = compress(INCOMPRESSIBLE)
        assert packed[0] == CODEC_NONE
        assert len(packed) == len(INCOMPRESSIBLE) + 1

    def test_tiny_objects_skip_the_codec(self) -> None:
        """A tree node is small enough that the codec's own framing dominates."""
        assert compress(b"tiny")[0] == CODEC_NONE

    def test_every_object_says_how_it_was_packed(self) -> None:
        """Per object, not per deployment.

        Without it, changing the codec means rewriting the corpus — and a corpus
        is not a thing anyone rewrites.
        """
        assert compress(COMPRESSIBLE)[0] != compress(INCOMPRESSIBLE)[0]

    def test_an_unreadable_frame_is_an_error_not_a_value(self) -> None:
        with pytest.raises(CorruptObject, match="empty"):
            decompress(b"")
        with pytest.raises(CorruptObject, match="codec"):
            decompress(bytes([99]) + b"whatever")
        with pytest.raises(CorruptObject, match="decompress"):
            decompress(bytes([CODEC_ZSTD]) + b"not a zstd frame")


class TestTheNamesAreUntouched:
    def test_the_codec_does_not_change_any_name(self) -> None:
        """**The load-bearing sentence.**

        The same content stored through a compressing backend and a plain one
        must get the same name. If it did not, two deployments configured
        differently would stop deduplicating against each other — silently, since
        both would still work perfectly on their own.
        """
        plain = store_over(InMemoryBackend())
        packed = store_over(CompressedBackend(InMemoryBackend()))

        # A zero-length chunk is not a valid object at all — an empty file is
        # a blob with no entries — so the codec rejects it before we get here.
        for payload in (COMPRESSIBLE, INCOMPRESSIBLE, b"tiny"):
            chunk = Chunk(payload)
            assert plain.put_object(chunk).name == packed.put_object(chunk).name
            assert plain.put_object(chunk).name == name_of(chunk)

    def test_content_survives_the_round_trip_byte_for_byte(self) -> None:
        store = store_over(CompressedBackend(InMemoryBackend()))
        for payload in (COMPRESSIBLE, INCOMPRESSIBLE, b"x"):
            outcome = store.put_object(Chunk(payload))
            assert store.get_as(outcome.name, Chunk).data == payload

    def test_a_compressed_object_still_verifies_on_delivery(self) -> None:
        """The decorator is transparent *because* the store checks it.

        A codec that returned anything other than what was written fails the
        ordinary delivery check rather than quietly serving damaged content.
        """
        backend = InMemoryBackend()
        store = store_over(CompressedBackend(backend))
        outcome = store.put_object(Chunk(COMPRESSIBLE))

        key = outcome.name.hex
        backend.corrupt(key, compress(b"different content entirely"))
        with pytest.raises(CorruptObject):
            store.get(outcome.name)

    def test_a_ranged_read_still_reads_the_right_bytes(self) -> None:
        store = store_over(CompressedBackend(InMemoryBackend()))
        outcome = store.put_object(Chunk(COMPRESSIBLE))
        assert store.get_range(outcome.name, 10, 20) == COMPRESSIBLE[10:30]


class TestItIsMeasured:
    def test_the_store_reports_both_sizes(self) -> None:
        """One is a fact about the content, the other about this deployment.

        Conflating them makes "is compression paying for itself" unanswerable,
        and makes deduplication look better than it is.
        """
        store = store_over(CompressedBackend(InMemoryBackend()))
        outcome = store.put_object(Chunk(COMPRESSIBLE))

        assert outcome.size == len(encode(Chunk(COMPRESSIBLE)))
        assert outcome.stored_size < outcome.size

    def test_incompressible_content_reports_no_saving(self) -> None:
        store = store_over(CompressedBackend(InMemoryBackend()))
        outcome = store.put_object(Chunk(INCOMPRESSIBLE))
        assert outcome.stored_size >= outcome.size

    def test_the_catalog_counts_medium_bytes(self) -> None:
        """What a sweep returns is medium bytes, so that is what it records."""
        catalog = InMemoryWriteCatalog()
        store = ObjectStore(
            CompressedBackend(InMemoryBackend()),
            catalog=catalog,
            tombstones=SqliteTombstoneStore.open(":memory:"),
            clock=ManualClock(),
        )
        store.put_object(Chunk(COMPRESSIBLE))
        _, total = catalog.total()
        assert 0 < total < len(COMPRESSIBLE)

    def test_it_actually_saves_something_on_realistic_content(self) -> None:
        """Source code, prompts and manifests — what an environment is made of
        besides its datasets.
        """
        text = b"".join(
            f"def verify_{n}(result):\n    return result.score > 0.5\n".encode() for n in range(400)
        )
        packed = compress(text)
        assert len(packed) < len(text) // 4
        # And the codec is doing better than a naive one would.
        assert len(packed) <= len(zlib.compress(text)) * 1.1
