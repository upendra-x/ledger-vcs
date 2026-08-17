"""The corpus cannot be renamed by accident.

Content addressing, the canonical encoding and the chunking parameters are the
decisions that cannot be deferred or retrofitted. What makes
them dangerous is that getting one wrong *later* fails silently: objects written
under the old constants stay valid and readable, new writes simply stop matching
them, deduplication halves, and nothing raises.

So the fingerprint is pinned. A red test with an explanation is the only warning
anyone will get, and it has to arrive before a single object is written under
the new value.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from src.format import constants as C

# ─────────────────────────────────────────────────────────────────────────────
# If this test fails, read this before touching the expected value.
#
#   Changing a constant in ledger/format/constants.py starts a NEW CORPUS.
#   Every object already stored keeps its old name; nothing you write from now
#   on will deduplicate against it. That is occasionally the right call — but it
#   is a migration, not a diff. Updating the literal below to make the build
#   green is how a corpus silently forks.
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_FORMAT_FINGERPRINT = "73b6afacd5d6c2f2e05394263b921470315cffd99fba85b64d5b4ae495ba5d84"
EXPECTED_GEAR_TABLE_DIGEST = "428c4d2a5e10e678c9f366acd4fe6885e3d7970d9eafde3e9280344ac21f5411"


def test_format_constants_are_frozen() -> None:
    assert C.FORMAT_FINGERPRINT == EXPECTED_FORMAT_FINGERPRINT, (
        "A name-affecting constant changed. This renames the entire corpus and "
        "silently splits deduplication — see the comment above."
    )


def test_gear_table_is_frozen() -> None:
    """The gear table decides every chunk boundary, so it decides every blob name.

    Pinned separately from the fingerprint because it is the one constant that is
    *derived* rather than written down, so it can drift without anyone editing a
    literal — a BLAKE3 change or a domain-string typo would do it.
    """
    assert C.GEAR_TABLE_DIGEST == EXPECTED_GEAR_TABLE_DIGEST
    assert len(C.GEAR) == 256
    assert all(0 <= g < 2**64 for g in C.GEAR)
    assert len(set(C.GEAR)) == 256, "a collision in the gear table would bias boundaries"


def test_gear_table_derivation_is_reproducible_across_processes() -> None:
    """Derived at import, so it must not depend on anything process-local.

    A dict ordering, a hash seed or an environment variable leaking into the
    derivation would produce a table that differs between the API process and a
    worker — which would silently write two chunkings of the same file.
    """
    script = "from src.format.constants import GEAR_TABLE_DIGEST; print(GEAR_TABLE_DIGEST)"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"PYTHONHASHSEED": str(seed)},
        ).stdout.strip()
        for seed in (0, 1, 42)
    }
    assert outputs == {EXPECTED_GEAR_TABLE_DIGEST}


class TestObjectKinds:
    """Tag values are frozen because they are the first byte of every preimage."""

    def test_object_kind_values(self) -> None:
        assert {k.name: k.value for k in C.ObjectKind} == {
            "CHUNK": 1,
            "BLOB": 2,
            "TREE": 3,
            "COMMIT": 4,
        }

    def test_entry_kind_values(self) -> None:
        assert {k.name: k.value for k in C.EntryKind} == {
            "TREE": 1,
            "BLOB": 2,
            "SYMLINK": 3,
            "CONFLICT": 4,
        }

    def test_interior_nodes_did_not_get_their_own_tags(self) -> None:
        """There are "exactly four types", and a blob index node or an
        interior directory node is *not* a fifth one — it is a BLOB or a TREE
        carrying ``level > 0``.

        Two sibling designs independently invented an INDEX_NODE tag here, and
        one of them numbered it 3, colliding with TREE. Asserting the closed set
        is what stops that from being re-litigated in a way that renames objects.
        """
        assert len(C.ObjectKind) == 4
        assert max(k.value for k in C.ObjectKind) == 4


class TestChunkParameters:
    def test_size_ordering(self) -> None:
        assert C.MIN_CHUNK_BYTES < C.AVG_CHUNK_BYTES < C.MAX_CHUNK_BYTES

    def test_sizes_match_the_design(self) -> None:
        assert C.MIN_CHUNK_BYTES == 256 * 1024
        assert C.AVG_CHUNK_BYTES == 1024 * 1024
        assert C.MAX_CHUNK_BYTES == 4 * 1024 * 1024

    def test_normalized_chunking_level_2_bit_counts(self) -> None:
        """The short mask must be *harder* than the long one.

        Normalized chunking works by making a cut unlikely before the average
        size and likely after it. Inverting these two would widen the size
        distribution instead of tightening it, and would do so without any test
        failing anywhere else.
        """
        short_bits = bin(C.CUT_MASK_SHORT).count("1")
        long_bits = bin(C.CUT_MASK_LONG).count("1")
        assert short_bits == 22
        assert long_bits == 18
        assert short_bits > long_bits

    def test_masks_use_high_bits(self) -> None:
        """In h_i = Sum_j G[b_{i-j}] << j, bit k depends only on bytes within
        distance k. Masking low bits would make a boundary depend on a handful of
        recent bytes instead of the full 64-byte window.
        """
        for mask in (C.CUT_MASK_SHORT, C.CUT_MASK_LONG):
            assert mask & (1 << 63), "top bit must participate"
            assert mask & 0xFF == 0, "low bits must not participate"

    def test_standalone_threshold_matches_min_chunk(self) -> None:
        """Essentially every chunk is stored standalone, so its name is its address
        and it needs no location-index entry at all.
        """
        assert C.STANDALONE_THRESHOLD_BYTES == C.MIN_CHUNK_BYTES


class TestSplitParameters:
    def test_clamps_bracket_the_period(self) -> None:
        assert C.SPLIT_MIN_ENTRIES < C.SPLIT_PERIOD < C.SPLIT_MAX_ENTRIES

    def test_split_domain_is_versioned(self) -> None:
        """The domain string is what a successor split rule changes, so it has to
        carry a version — otherwise the only way to introduce one is to renumber
        something that is already frozen.
        """
        assert C.SPLIT_DOMAIN.endswith(b".v1")


@pytest.mark.parametrize(
    "name",
    [
        "DIGEST_BYTES",
        "FORMAT_VERSION",
        "MIN_CHUNK_BYTES",
        "AVG_CHUNK_BYTES",
        "MAX_CHUNK_BYTES",
        "CUT_MASK_SHORT",
        "CUT_MASK_LONG",
        "SPLIT_DOMAIN",
        "SPLIT_PERIOD",
        "SPLIT_MIN_ENTRIES",
        "SPLIT_MAX_ENTRIES",
        "MAX_NODE_BYTES",
        "MAX_ENTRY_NAME_BYTES",
        "MODE_REGULAR",
        "MODE_EXEC",
    ],
)
def test_every_name_affecting_constant_is_in_the_fingerprint(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constant added to this module but omitted from ``format_fingerprint``
    would be *unprotected*: someone could change it, rename the corpus, and the
    frozen test above would still pass.

    So perturb each one and assert the fingerprint actually moves. This is the
    test that keeps the freeze honest as the module grows.
    """
    original = getattr(C, name)
    perturbed = original + b"!" if isinstance(original, bytes) else original + 1

    monkeypatch.setattr(C, name, perturbed)
    assert C.format_fingerprint() != EXPECTED_FORMAT_FINGERPRINT, (
        f"{name} does not participate in the fingerprint, so changing it would "
        f"rename the corpus without failing test_format_constants_are_frozen"
    )

    monkeypatch.undo()
    assert C.format_fingerprint() == EXPECTED_FORMAT_FINGERPRINT
