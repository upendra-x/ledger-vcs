"""FROZEN. Every constant that participates in an object's name lives here.

    ┌───────────────────────────────────────────────────────────────────────┐
    │  Changing any value in this file renames the entire corpus.           │
    │                                                                       │
    │  The failure is SILENT. Objects written under the old constants stay  │
    │  valid and readable forever, because they are immutable. New writes    │
    │  simply stop matching them: the same bytes acquire a second name,      │
    │ deduplication quietly halves, and nothing raises. The specification
    lists │
    │  exactly this as the decision that cannot be retrofitted.              │
    │                                                                       │
    │  `test_format_constants_are_frozen` pins FORMAT_FINGERPRINT. If you    │
    │  changed something here on purpose, you are starting a new corpus.     │
    └───────────────────────────────────────────────────────────────────────┘

Why one file. Eight subsystems were designed against this specification
independently and produced four incompatible type-tag tables, two endiannesses
and three hash preimages between them. Every one of those changes every name. A
single owning module, fingerprinted by a test, is what makes that class of drift
a red build instead of a slow corpus fork.

What is *not* here: chunk parameters are the production defaults, and tests
inject scaled-down ones through ``ChunkParams`` rather than mutating these. Two
different parameter sets produce different chunk boundaries and therefore
different names — which is correct and expected, and is why the production
values are pinned here while remaining injectable at the call site.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final, final

from blake3 import blake3

__all__ = [
    "AVG_CHUNK_BYTES",
    "CUT_MASK_LONG",
    "CUT_MASK_SHORT",
    "DIGEST_BYTES",
    "FORMAT_FINGERPRINT",
    "FORMAT_VERSION",
    "GEAR",
    "GEAR_TABLE_DIGEST",
    "MAX_CHUNK_BYTES",
    "MAX_ENTRY_NAME_BYTES",
    "MAX_NODE_BYTES",
    "MIN_CHUNK_BYTES",
    "MODE_EXEC",
    "MODE_REGULAR",
    "SPLIT_DOMAIN",
    "SPLIT_MAX_ENTRIES",
    "SPLIT_MIN_ENTRIES",
    "SPLIT_PERIOD",
    "STANDALONE_THRESHOLD_BYTES",
    "EntryKind",
    "ObjectKind",
    "format_fingerprint",
]

KIB: Final = 1024
MIB: Final = 1024 * 1024

# ─────────────────────────────────────────────────────────────────────────────
# 1. Hashing and framing
# ─────────────────────────────────────────────────────────────────────────────

#: BLAKE3-256. Chosen for one property doing three jobs: dedup,
#: integrity and cacheability all fall out of the name being the content hash.
DIGEST_BYTES: Final = 32

#: Bumped only when the *encoding* of an object changes. Because the version
#: byte is inside the framing and therefore inside the hash, a re-encoded object
#: gets a new name and old objects remain valid and readable forever.
FORMAT_VERSION: Final = 1


@final
class ObjectKind(IntEnum):
    """The four object types. These integers are frozen forever.

    Interior nodes are deliberately *not* new kinds. A blob index node is a
    ``BLOB`` with ``level > 0``; an interior directory node is a ``TREE`` with
    ``level > 0``. That keeps the "exactly four types" literally true and
    gives depth validation for free, where a fifth tag would have bought nothing
    and would have collided with a sibling design's numbering.
    """

    CHUNK = 1
    BLOB = 2
    TREE = 3
    COMMIT = 4


@final
class EntryKind(IntEnum):
    """What a tree entry points at. Frozen forever.

    ``CONFLICT`` is reserved but rejected by every v1 decoder. The type list keeps
    jj-style stored conflicts as scoped future work: a conflicted commit cannot
    be materialised or built, so admitting one would force every downstream
    consumer to invent a rule for it. Reserving the value now means adding it
    later is a decoder change, not a renumbering.
    """

    TREE = 1
    BLOB = 2
    SYMLINK = 3
    CONFLICT = 4


#: The only two file modes Ledger records. Git's precedent: permissions beyond
#: "is it executable" are not portable and not worth putting inside a hash.
#: A redundant field is a hazard — two encoders choosing differently for the
#: same logical entry split dedup silently.
MODE_REGULAR: Final = 0o644
MODE_EXEC: Final = 0o755

#: A single path component. 255 matches every filesystem we materialize onto.
MAX_ENTRY_NAME_BYTES: Final = 255

# ─────────────────────────────────────────────────────────────────────────────
# 2. Content-defined chunking
# ─────────────────────────────────────────────────────────────────────────────

#: Below this, three costs converge: a ~40-byte reference in the
#: parent manifest, one object-store request per chunk on a cold read, and the
#: 128 KiB minimum billable size of infrequent-access tiers.
MIN_CHUNK_BYTES: Final = 256 * KIB

#: ~44,000 chunks for a 43 GiB environment, referenced by ~1.8 MiB of manifests.
AVG_CHUNK_BYTES: Final = 1 * MIB

#: Bounds the cost of one fetch and the memory a single chunk can occupy.
MAX_CHUNK_BYTES: Final = 4 * MIB

#: Objects at or above this size are stored standalone — their name is their
#: address, so they need no location-index entry at all. Smaller objects are
#: packed. Equal to the minimum chunk size, so essentially every chunk from
#: every large file takes the standalone path.
STANDALONE_THRESHOLD_BYTES: Final = MIN_CHUNK_BYTES

_GEAR_DOMAIN: Final = b"ledger.cdc.gear.v1"


def _derive_gear_table() -> tuple[int, ...]:
    """Derive the 256-entry gear table deterministically from a pinned domain.

    Published FastCDC tables are arbitrary random constants that an
    implementation must copy byte-for-byte. Deriving ours from a domain string
    means any language can reproduce it from this one line of specification —
    which matters because the Rust jj backend must chunk identically or it will
    write objects the Python side cannot deduplicate against.

    Pinned by ``GEAR_TABLE_DIGEST``: derivation is a convenience, the *values*
    are the contract.
    """
    return tuple(
        int.from_bytes(blake3(_GEAR_DOMAIN + bytes([i])).digest(length=8), "big")
        for i in range(256)
    )


GEAR: Final = _derive_gear_table()

#: Pins the derived table. If the derivation, the domain string or the BLAKE3
#: implementation ever changes, this catches it before a single object is named.
GEAR_TABLE_DIGEST: Final = blake3(b"".join(g.to_bytes(8, "big") for g in GEAR)).hexdigest()

# Normalized chunking, level 2: before the average size a *harder*
# mask makes cuts unlikely; after it an *easier* mask makes them likely. That
# pulls the chunk-size distribution towards the average and away from FastCDC's
# characteristic long tail, which is what keeps the max-truncation fraction low.
#
# Bit counts follow the FastCDC paper: log2(avg) ± normalization level.
_AVG_BITS: Final = AVG_CHUNK_BYTES.bit_length() - 1  # 20
_NORMALIZATION_LEVEL: Final = 2

# High bits, not low. In the gear hash h_i = Σ_{j<64} G[b_{i-j}] << j, bit k
# receives contributions only from bytes at distance j <= k — so the *high* bits
# are the ones that depend on the full 64-byte window. Masking low bits would
# make a boundary depend on only a handful of recent bytes.
_MASK_SHORT_BITS: Final = _AVG_BITS + _NORMALIZATION_LEVEL  # 22
_MASK_LONG_BITS: Final = _AVG_BITS - _NORMALIZATION_LEVEL  # 18

CUT_MASK_SHORT: Final = ((1 << _MASK_SHORT_BITS) - 1) << (64 - _MASK_SHORT_BITS)
CUT_MASK_LONG: Final = ((1 << _MASK_LONG_BITS) - 1) << (64 - _MASK_LONG_BITS)

assert CUT_MASK_SHORT == 0xFFFFFC0000000000, "cut mask drifted"
assert CUT_MASK_LONG == 0xFFFFC00000000000, "cut mask drifted"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Canonical shape — how trees and blob index nodes split
# ─────────────────────────────────────────────────────────────────────────────
#
# A wide directory becomes a "B-tree ordered by entry name that splits past a fanout
# threshold" without saying *where* the split points go, and the choice decides
# whether the cost table is true.
#
# Filling nodes left to right at a fixed fanout is deterministic, but inserting
# one entry near the start shifts every later boundary — so a one-file change in
# a wide directory rewrites the whole directory, and Diff's prune, merge's prune
# and cross-version dedup all fail together.
#
# So boundaries are content-defined on the entry *name*: the same trick as CDC,
# applied to keys. The shape is a pure function of the sorted entry set (two
# writers building the same directory get the same name) and is
# stable under insertion (only the containing node and its ancestors change).
#
# THE LEVEL SALT IS MANDATORY. Without it, every level-1 key is by construction
# a key that was already a boundary at level 0, so the predicate fires on every
# one of them and the tree degenerates into fixed positional grouping — silently,
# because SPLIT_MIN_ENTRIES clamps it back into looking correct.

SPLIT_DOMAIN: Final = b"ledger.tree.split.v1"

#: Expected entries per node. Chosen so the mean node lands near the stated design's
#: fanout of 512 once the min/max clamps are accounted for.
SPLIT_PERIOD: Final = 448

#: Clamps on the geometric distribution the predicate produces. Without a
#: minimum, a run of unlucky keys yields single-entry nodes and the depth blows
#: up; without a maximum, a lucky run yields one enormous node.
SPLIT_MIN_ENTRIES: Final = 64
SPLIT_MAX_ENTRIES: Final = 2048

#: A hard ceiling on the encoded size of one node, independent of entry count.
#: Long names would otherwise let SPLIT_MAX_ENTRIES produce a multi-megabyte
#: object, and "no object is ever large" is what keeps every other property true.
MAX_NODE_BYTES: Final = 128 * KIB


# ─────────────────────────────────────────────────────────────────────────────
# 4. The fingerprint
# ─────────────────────────────────────────────────────────────────────────────


def format_fingerprint() -> str:
    """A BLAKE3 over every name-affecting constant in this module.

    Pinned by a test. This exists so that "the encoding is frozen" is a fact the
    build checks rather than a convention people remember.
    """
    parts: list[str] = [
        f"digest_bytes={DIGEST_BYTES}",
        f"format_version={FORMAT_VERSION}",
        "object_kinds=" + ",".join(f"{k.name}:{k.value}" for k in ObjectKind),
        "entry_kinds=" + ",".join(f"{k.name}:{k.value}" for k in EntryKind),
        f"mode_regular={MODE_REGULAR:o}",
        f"mode_exec={MODE_EXEC:o}",
        f"max_entry_name_bytes={MAX_ENTRY_NAME_BYTES}",
        f"min_chunk={MIN_CHUNK_BYTES}",
        f"avg_chunk={AVG_CHUNK_BYTES}",
        f"max_chunk={MAX_CHUNK_BYTES}",
        f"cut_mask_short={CUT_MASK_SHORT:#018x}",
        f"cut_mask_long={CUT_MASK_LONG:#018x}",
        f"gear_table_digest={GEAR_TABLE_DIGEST}",
        f"split_domain={SPLIT_DOMAIN.decode()}",
        f"split_period={SPLIT_PERIOD}",
        f"split_min_entries={SPLIT_MIN_ENTRIES}",
        f"split_max_entries={SPLIT_MAX_ENTRIES}",
        f"max_node_bytes={MAX_NODE_BYTES}",
    ]
    return blake3("\n".join(parts).encode()).hexdigest()


FORMAT_FINGERPRINT: Final = format_fingerprint()
