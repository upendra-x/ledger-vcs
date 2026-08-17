"""The canonical encoding: naming, strictness, and the rules that give one
content exactly one name.

The organising idea of this file is that *every rejected spelling is a
deduplication bug that did not happen*. Each test below names a way one logical
object could have acquired a second encoding, and asserts the decoder refuses it.
"""

from __future__ import annotations

import pytest

from src.errors import CorruptObject, MalformedObject, NotCanonical, UnsupportedFormatVersion
from src.format import constants as C
from src.format.codec import (
    CHANGE_ID_BYTES,
    HEADER_BYTES,
    decode,
    decode_as,
    encode,
    is_canonical,
    name_of,
    name_of_encoded,
    peek_kind,
    verify,
)
from src.format.model import Blob, BlobEntry, Chunk, Commit, Tree, TreeEntry
from src.format.wire import Writer
from src.ids import ChangeId, ObjectName

NAME_A = ObjectName(b"\xaa" * 32)
NAME_B = ObjectName(b"\xbb" * 32)
NAME_C = ObjectName(b"\xcc" * 32)
CHANGE = ChangeId("ab" * CHANGE_ID_BYTES)


def a_commit(**overrides: object) -> Commit:
    defaults: dict[str, object] = {
        "tree": NAME_A,
        "parents": (),
        "change_id": CHANGE,
        "author": "agent-17",
        "committer": "agent-17",
        "timestamp_us": 1_700_000_000_000_000,
        "message": "add the verifier",
        "metadata": (),
    }
    return Commit(**(defaults | overrides))  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Framing and naming
# ─────────────────────────────────────────────────────────────────────────────


class TestFramingAndNaming:
    def test_the_type_tag_is_the_first_byte_and_the_version_the_second(self) -> None:
        framed = encode(Chunk(b"hello"))
        assert framed[0] == C.ObjectKind.CHUNK.value
        assert framed[1] == C.FORMAT_VERSION
        assert framed[HEADER_BYTES:] == b"hello"

    def test_name_is_blake3_of_the_framed_bytes(self) -> None:
        """The rule that makes verification 'rehash the stored bytes'."""
        from blake3 import blake3

        chunk = Chunk(b"hello")
        framed = encode(chunk)
        assert name_of(chunk) == ObjectName(blake3(framed).digest())

    def test_chunks_are_framed_like_everything_else(self) -> None:
        """An unframed chunk would fail its own verification on the way out.

        ``get()`` verifies by rehashing the stored bytes. If a chunk were stored
        raw while its name covered a header, every chunk fetched by a process
        that did not write it would raise CorruptObject.
        """
        chunk = Chunk(b"hello")
        stored = encode(chunk)
        verify(name_of(chunk), stored)  # must not raise

    def test_identical_payloads_of_different_kinds_get_different_names(self) -> None:
        """The type tag is inside the hash, so no object can ever be
        confused for one of another type.
        """
        payload = b"\x00" * 16
        chunk_name = name_of(Chunk(payload))
        # Same 16 bytes, offered as the tail of a differently-tagged object.
        forged = Writer().u8(C.ObjectKind.BLOB.value).u8(C.FORMAT_VERSION).raw(payload).finish()
        assert name_of_encoded(forged) != chunk_name

    def test_peek_kind_does_not_decode(self) -> None:
        for obj in (Chunk(b"x"), Blob(0, (BlobEntry(NAME_A, 1),)), a_commit()):
            assert peek_kind(encode(obj)) is obj.KIND

    def test_peek_kind_rejects_an_unknown_tag(self) -> None:
        with pytest.raises(MalformedObject, match="unknown object kind"):
            peek_kind(bytes([99, 1]))

    def test_peek_kind_rejects_a_stub(self) -> None:
        with pytest.raises(MalformedObject, match="shorter than its header"):
            peek_kind(b"\x01")

    def test_verify_rejects_a_single_flipped_bit(self) -> None:
        chunk = Chunk(b"the quick brown fox")
        framed = bytearray(encode(chunk))
        framed[-1] ^= 0x01
        with pytest.raises(CorruptObject, match="do not match the name"):
            verify(name_of(chunk), bytes(framed))

    def test_a_future_format_version_is_refused_not_guessed(self) -> None:
        framed = bytearray(encode(Chunk(b"x")))
        framed[1] = 99
        with pytest.raises(UnsupportedFormatVersion):
            decode(bytes(framed))


# ─────────────────────────────────────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────────────────────────────────────


class TestChunk:
    def test_round_trip(self) -> None:
        chunk = Chunk(bytes(range(256)))
        assert decode(encode(chunk)) == chunk

    def test_zero_length_chunk_is_rejected(self) -> None:
        """An empty file is a blob with no entries, not a blob holding an empty
        chunk. Allowing both would be two encodings of one thing.
        """
        with pytest.raises(NotCanonical, match="zero-length chunk"):
            encode(Chunk(b""))

    def test_oversize_chunk_is_rejected(self) -> None:
        with pytest.raises(NotCanonical, match="maximum chunk size"):
            encode(Chunk(b"\x00" * (C.MAX_CHUNK_BYTES + 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Blobs
# ─────────────────────────────────────────────────────────────────────────────


class TestBlob:
    def test_round_trip(self) -> None:
        blob = Blob(level=0, entries=(BlobEntry(NAME_A, 100), BlobEntry(NAME_B, 200)))
        assert decode(encode(blob)) == blob

    def test_size_is_derived_by_summing(self) -> None:
        """Not stored. A stored total would be a second way to express the same
        fact, and two encoders disagreeing about a redundant field is how one
        logical object acquires two names.
        """
        blob = Blob(level=0, entries=(BlobEntry(NAME_A, 100), BlobEntry(NAME_B, 200)))
        assert blob.size == 300

    def test_empty_blob_is_the_empty_file(self) -> None:
        empty = Blob(level=0, entries=())
        assert decode(encode(empty)) == empty
        assert empty.size == 0

    def test_zero_size_entry_is_rejected(self) -> None:
        with pytest.raises(NotCanonical, match="at least one byte"):
            encode(Blob(level=0, entries=(BlobEntry(NAME_A, 0),)))

    def test_empty_interior_node_is_rejected(self) -> None:
        with pytest.raises(NotCanonical, match="must be at level 0"):
            encode(Blob(level=1, entries=()))

    def test_single_child_interior_node_is_rejected(self) -> None:
        """It says nothing its only child does not already say — so admitting it
        would give one logical structure two encodings.
        """
        with pytest.raises(NotCanonical, match="at least two children"):
            encode(Blob(level=1, entries=(BlobEntry(NAME_A, 10),)))


# ─────────────────────────────────────────────────────────────────────────────
# Trees
# ─────────────────────────────────────────────────────────────────────────────


class TestTree:
    def test_round_trip(self) -> None:
        tree = Tree(
            level=0,
            entries=(
                TreeEntry(b"README.md", C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 11),
                TreeEntry(b"data", C.EntryKind.TREE, NAME_B, 0, 0),
                TreeEntry(b"run.sh", C.EntryKind.BLOB, NAME_C, C.MODE_EXEC, 42),
            ),
        )
        assert decode(encode(tree)) == tree

    def test_entries_must_be_strictly_ascending(self) -> None:
        """The single most likely way for two writers to name one directory
        differently — and the reason ingest re-encodes rather than only rehashing.
        """
        out_of_order = Tree(
            level=0,
            entries=(
                TreeEntry(b"zebra", C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 1),
                TreeEntry(b"apple", C.EntryKind.BLOB, NAME_B, C.MODE_REGULAR, 1),
            ),
        )
        with pytest.raises(NotCanonical, match="strictly ascending"):
            encode(out_of_order)

    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(NotCanonical, match="strictly ascending"):
            encode(
                Tree(
                    level=0,
                    entries=(
                        TreeEntry(b"a", C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 1),
                        TreeEntry(b"a", C.EntryKind.BLOB, NAME_B, C.MODE_REGULAR, 1),
                    ),
                )
            )

    def test_ordering_is_unsigned_byte_order(self) -> None:
        """UTF-8 is order-preserving, so a str-sorting implementation and a
        bytes-sorting one produce the same sequence. This pins that they must.
        """
        names = [b"Z", b"a", b"\xc3\xa9", b"\xef\xbc\xa1"]  # 'Z', 'a', 'é', fullwidth 'A'
        assert names == sorted(names)
        tree = Tree(
            level=0,
            entries=tuple(TreeEntry(n, C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 1) for n in names),
        )
        assert decode(encode(tree)) == tree

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"a/b", id="contains-separator"),
            pytest.param(b"a\x00b", id="contains-nul"),
            pytest.param(b".", id="dot"),
            pytest.param(b"..", id="dotdot"),
            pytest.param(b"\xff\xfe", id="not-utf8"),
            pytest.param(b"x" * 256, id="too-long"),
        ],
    )
    def test_rejects_dangerous_names(self, name: bytes) -> None:
        tree = Tree(
            level=0, entries=(TreeEntry(name, C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 1),)
        )
        with pytest.raises(NotCanonical):
            encode(tree)

    def test_unicode_names_are_not_normalized(self) -> None:
        """The identity of a name is its bytes. Normalizing here would make two
        distinct names collide; the filesystem hazard belongs to the materializer.
        """
        composed = "\u00e9".encode()  # single code point U+00E9
        decomposed = "e\u0301".encode()  # 'e' + U+0301 combining acute
        assert composed != decomposed, "fixture is wrong: these must differ as bytes"
        entries = sorted([composed, decomposed])
        tree = Tree(
            level=0,
            entries=tuple(
                TreeEntry(n, C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 1) for n in entries
            ),
        )
        assert decode(encode(tree)) == tree

    @pytest.mark.parametrize("mode", [0, 0o600, 0o777, 0o444])
    def test_file_mode_is_a_closed_set(self, mode: int) -> None:
        """Design: beyond 'is it executable', permissions are neither portable
        nor worth putting inside a hash. A redundant field with several valid
        values is a hazard: two encoders choosing differently for the same logical
        entry split dedup silently.
        """
        with pytest.raises(NotCanonical, match="0o644 or 0o755"):
            encode(
                Tree(
                    level=0,
                    entries=(TreeEntry(b"f", C.EntryKind.BLOB, NAME_A, mode, 1),),
                )
            )

    def test_subtree_entries_carry_no_mode_or_size(self) -> None:
        with pytest.raises(NotCanonical, match="carries no mode and no size"):
            encode(Tree(level=0, entries=(TreeEntry(b"d", C.EntryKind.TREE, NAME_A, 0, 99),)))

    def test_conflict_entries_are_reserved_and_rejected(self) -> None:
        """A conflicted commit cannot be materialised or built, so
        v1 refuses rather than pretending to support it.
        """
        with pytest.raises(NotCanonical, match="conflict entries are reserved"):
            encode(Tree(level=0, entries=(TreeEntry(b"c", C.EntryKind.CONFLICT, NAME_A, 0, 0),)))

    def test_interior_entries_are_bare_routing_keys(self) -> None:
        with pytest.raises(NotCanonical, match="bare subtree reference"):
            encode(
                Tree(
                    level=1,
                    entries=(
                        TreeEntry(b"m", C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 1),
                        TreeEntry(b"z", C.EntryKind.TREE, NAME_B, 0, 0),
                    ),
                )
            )

    def test_interior_node_round_trips(self) -> None:
        interior = Tree(
            level=1,
            entries=(
                TreeEntry(b"mmm", C.EntryKind.TREE, NAME_A, 0, 0),
                TreeEntry(b"zzz", C.EntryKind.TREE, NAME_B, 0, 0),
            ),
        )
        assert decode(encode(interior)) == interior

    def test_oversized_node_is_rejected(self) -> None:
        """No object is ever large — that is what keeps every other property true."""
        entries = tuple(
            TreeEntry(f"{i:08d}".encode() + b"x" * 200, C.EntryKind.BLOB, NAME_A, C.MODE_REGULAR, 1)
            for i in range(1000)
        )
        with pytest.raises(NotCanonical, match="maximum node size"):
            encode(Tree(level=0, entries=entries))


# ─────────────────────────────────────────────────────────────────────────────
# Commits
# ─────────────────────────────────────────────────────────────────────────────


class TestCommit:
    def test_round_trip(self) -> None:
        commit = a_commit(parents=(NAME_B, NAME_C), metadata=(("git_sha", "deadbeef"),))
        assert decode(encode(commit)) == commit

    def test_parent_order_is_meaningful_and_preserved(self) -> None:
        """The first parent is the branch that was being advanced, so parents are
        not sorted — two orderings are two different commits, deliberately.
        """
        forward = a_commit(parents=(NAME_B, NAME_C))
        reversed_ = a_commit(parents=(NAME_C, NAME_B))
        assert name_of(forward) != name_of(reversed_)
        assert decode_as(encode(forward), Commit).parents == (NAME_B, NAME_C)

    def test_duplicate_parents_are_rejected(self) -> None:
        with pytest.raises(NotCanonical, match="duplicate parent"):
            encode(a_commit(parents=(NAME_B, NAME_B)))

    def test_metadata_keys_must_be_strictly_ascending(self) -> None:
        with pytest.raises(NotCanonical, match="strictly ascending"):
            encode(a_commit(metadata=(("zebra", "1"), ("apple", "2"))))

    def test_duplicate_metadata_keys_are_rejected(self) -> None:
        with pytest.raises(NotCanonical, match="strictly ascending"):
            encode(a_commit(metadata=(("k", "1"), ("k", "2"))))

    def test_author_and_committer_are_required(self) -> None:
        with pytest.raises(NotCanonical, match="author and committer"):
            encode(a_commit(author=""))

    def test_change_id_must_be_16_bytes(self) -> None:
        with pytest.raises(NotCanonical, match="16 bytes"):
            encode(a_commit(change_id=ChangeId("ab" * 8)))

    def test_change_id_survives_a_rewrite_while_the_name_changes(self) -> None:
        """An automation refers to 'the change that adds the
        verifier' across an amendment, instead of chasing a hash that moves.
        """
        original = a_commit(message="add the verifier")
        amended = a_commit(message="add the verifier (typo fixed)")
        assert name_of(original) != name_of(amended)
        assert original.change_id == amended.change_id

    def test_a_root_commit_has_no_parents(self) -> None:
        root = a_commit(parents=())
        assert decode_as(encode(root), Commit).parents == ()
        assert not root.is_merge

    def test_merge_commits_are_recognised(self) -> None:
        assert a_commit(parents=(NAME_B, NAME_C)).is_merge

    def test_negative_timestamps_round_trip(self) -> None:
        """Imported git history predates the epoch in real repositories."""
        commit = a_commit(timestamp_us=-1_000_000)
        assert decode_as(encode(commit), Commit).timestamp_us == -1_000_000

    def test_unicode_message_round_trips(self) -> None:
        commit = a_commit(message="修复 verifier — 🎯")
        assert decode_as(encode(commit), Commit).message == commit.message


# ─────────────────────────────────────────────────────────────────────────────
# Strictness of the decoder itself
# ─────────────────────────────────────────────────────────────────────────────


class TestDecoderStrictness:
    def test_trailing_bytes_are_rejected(self) -> None:
        """Tolerating them would let two byte strings decode to the same object,
        so one logical object would have two valid names.
        """
        with pytest.raises(MalformedObject, match="trailing bytes"):
            decode(encode(a_commit()) + b"\x00")

    def test_truncation_is_rejected(self) -> None:
        framed = encode(a_commit())
        with pytest.raises(MalformedObject, match="truncated"):
            decode(framed[:-4])

    def test_a_declared_count_larger_than_the_payload_is_rejected(self) -> None:
        """The classic decoder bug: allocate on an attacker-controlled count."""
        framed = Writer().u8(C.ObjectKind.BLOB.value).u8(C.FORMAT_VERSION)
        framed.u8(0).u32(1_000_000)
        with pytest.raises(MalformedObject, match="truncated"):
            decode(framed.finish())

    def test_unknown_entry_kind_is_rejected(self) -> None:
        framed = (
            Writer()
            .u8(C.ObjectKind.TREE.value)
            .u8(C.FORMAT_VERSION)
            .u8(0)
            .u32(1)
            .bytes_u8(b"f")
            .u8(200)
            .name(NAME_A)
            .u16(C.MODE_REGULAR)
            .u64(1)
            .finish()
        )
        with pytest.raises(MalformedObject, match="unknown tree entry kind"):
            decode(framed)

    def test_is_canonical_rejects_a_valid_hash_over_wrong_order(self) -> None:
        """The specification gap, stated as a test.

        These bytes hash honestly to their own name — the hash check passes — but they are
        not the canonical encoding of the directory they describe. Accepting them
        gives that directory a second name.
        """
        out_of_order = (
            Writer()
            .u8(C.ObjectKind.TREE.value)
            .u8(C.FORMAT_VERSION)
            .u8(0)
            .u32(2)
            .bytes_u8(b"zebra")
            .u8(C.EntryKind.BLOB.value)
            .name(NAME_A)
            .u16(C.MODE_REGULAR)
            .u64(1)
            .bytes_u8(b"apple")
            .u8(C.EntryKind.BLOB.value)
            .name(NAME_B)
            .u16(C.MODE_REGULAR)
            .u64(1)
            .finish()
        )
        # It is self-consistent: it hashes to its own name.
        verify(name_of_encoded(out_of_order), out_of_order)
        # But it is not canonical, so ingest must refuse it.
        assert not is_canonical(out_of_order)
        with pytest.raises(NotCanonical):
            decode(out_of_order)
