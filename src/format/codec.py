"""The canonical encoding, and the naming rule. Frozen with the constants.

**The naming rule, in one line:**

    name = BLAKE3-256( kind_byte ‖ version_byte ‖ payload )

and the object's *stored bytes are that same framed string*, uniformly for all
four types including chunks. Verification is therefore the simplest statement it
can be — *rehash the stored bytes* — with no per-type reconstruction step that a
caller could get wrong, and the type tag is inside the hash by construction, so
a chunk and a blob with identical payloads cannot collide.

The two-byte header costs 0.0002% at a 1 MiB average chunk size. Framing chunks
like everything else is worth far more than that: an unframed chunk would make
``get()`` fail its own verification for every chunk it did not personally write.

**Canonicality is enforced, not assumed — and this is a gap in the
specification.** One logical object must have exactly one encoding, and every
object is rehashed against the name it was
sent under. But that check
alone is not enough: a client can build a tree whose entries are out of order,
hash *those* bytes honestly, and offer them under that hash. The hash check
passes. Now one logical tree has two valid names and deduplication has silently
split — precisely the failure this guards against.

So decoding rejects every non-canonical spelling, and ingest re-encodes and
compares (``src.store.cas``). The property that makes this airtight is:

    for every byte string b that decodes, encode(decode(b)) == b

There is no third outcome — either the bytes are rejected, or they are the one
encoding of what they mean.
"""

from __future__ import annotations

from typing import Final

from blake3 import blake3

from src.errors import (
    CorruptObject,
    MalformedObject,
    NotCanonical,
    UnsupportedFormatVersion,
)
from src.format.constants import (
    FORMAT_VERSION,
    MAX_CHUNK_BYTES,
    MAX_ENTRY_NAME_BYTES,
    MAX_NODE_BYTES,
    MODE_EXEC,
    MODE_REGULAR,
    EntryKind,
    ObjectKind,
)
from src.format.model import Blob, BlobEntry, Chunk, Commit, LedgerObject, Tree, TreeEntry
from src.format.wire import Cursor, Writer
from src.ids import ChangeId, ObjectName

__all__ = [
    "CHANGE_ID_BYTES",
    "HEADER_BYTES",
    "decode",
    "decode_as",
    "encode",
    "is_canonical",
    "name_of",
    "name_of_encoded",
    "peek_kind",
    "verify",
]

#: kind byte + version byte.
HEADER_BYTES: Final = 2

#: jj uses 16 bytes for a change id, and we match it so the mapping is
#: the identity function rather than a translation.
CHANGE_ID_BYTES: Final = 16

#: Versions this build can read. Readers accept every version ever emitted;
#: writers emit the newest. Old objects never need rewriting
#: because they are immutable.
_READABLE_VERSIONS: Final = frozenset({1})

_FORBIDDEN_NAME_BYTES: Final = frozenset({0x2F, 0x00})  # '/' and NUL
_RESERVED_NAMES: Final = frozenset({b".", b".."})
_VALID_FILE_MODES: Final = frozenset({MODE_REGULAR, MODE_EXEC})


# ─────────────────────────────────────────────────────────────────────────────
# Naming
# ─────────────────────────────────────────────────────────────────────────────


def name_of_encoded(framed: bytes) -> ObjectName:
    """The name of an object, given its encoded bytes."""
    return ObjectName(blake3(framed).digest())


def name_of(obj: LedgerObject) -> ObjectName:
    return name_of_encoded(encode(obj))


def verify(name: ObjectName, framed: bytes) -> None:
    """Raise unless ``framed`` really is the object called ``name``.

    Verification, in one function (phase 4 of the write protocol). Called on ingest and
    again on delivery, so a damaged medium, a poisoned cache entry or a lying
    client is an error at the point of use rather than a value handed to a rollout.
    """
    actual = name_of_encoded(framed)
    if actual != name:
        raise CorruptObject(
            "stored bytes do not match the name they were requested by",
            requested=str(name),
            actual=str(actual),
            length=len(framed),
        )


def peek_kind(framed: bytes) -> ObjectKind:
    """Read the type tag without decoding the payload.

    The store uses this to route a fetch, and the write path uses it to decide
    whether an object has children that must already exist.
    """
    if len(framed) < HEADER_BYTES:
        raise MalformedObject("object shorter than its header", length=len(framed))
    try:
        return ObjectKind(framed[0])
    except ValueError as exc:
        raise MalformedObject("unknown object kind", tag=framed[0]) from exc


def is_canonical(framed: bytes) -> bool:
    """Whether these exact bytes are the canonical encoding of what they mean."""
    try:
        return encode(decode(framed)) == framed
    except MalformedObject, NotCanonical, UnsupportedFormatVersion:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────


def encode(obj: LedgerObject) -> bytes:
    """Encode an object, validating it on the way out.

    Validation happens here as well as on decode so that a builder bug surfaces
    as an exception at the point of construction, rather than as an object that
    stores fine and fails to decode later.
    """
    writer = Writer().u8(obj.KIND.value).u8(FORMAT_VERSION)
    match obj:
        case Chunk():
            _encode_chunk(writer, obj)
        case Blob():
            _encode_blob(writer, obj)
        case Tree():
            _encode_tree(writer, obj)
        case Commit():
            _encode_commit(writer, obj)
    return writer.finish()


def _encode_chunk(writer: Writer, chunk: Chunk) -> None:
    _validate_chunk(chunk)
    writer.raw(chunk.data)


def _encode_blob(writer: Writer, blob: Blob) -> None:
    _validate_node_level(blob.level, len(blob.entries), "blob")
    writer.u8(blob.level).u32(len(blob.entries))
    for entry in blob.entries:
        if entry.size <= 0:
            raise NotCanonical("a blob entry must cover at least one byte", size=entry.size)
        writer.name(entry.target).u64(entry.size)
    _validate_node_size(writer, "blob")


def _encode_tree(writer: Writer, tree: Tree) -> None:
    _validate_node_level(tree.level, len(tree.entries), "tree")
    _validate_tree_entries(tree)
    writer.u8(tree.level).u32(len(tree.entries))
    for entry in tree.entries:
        writer.bytes_u8(entry.name).u8(entry.kind.value).name(entry.target)
        writer.u16(entry.mode).u64(entry.size)
    _validate_node_size(writer, "tree")


def _encode_commit(writer: Writer, commit: Commit) -> None:
    _validate_commit(commit)
    writer.name(commit.tree).u16(len(commit.parents))
    for parent in commit.parents:
        writer.name(parent)
    writer.raw(bytes.fromhex(commit.change_id.value))
    writer.bytes_u8(commit.author.encode())
    writer.bytes_u8(commit.committer.encode())
    writer.i64(commit.timestamp_us)
    writer.bytes_u32(commit.message.encode())
    writer.u32(len(commit.metadata))
    for key, value in commit.metadata:
        writer.bytes_u8(key.encode()).bytes_u32(value.encode())


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────


def decode(framed: bytes) -> LedgerObject:
    """Decode, rejecting anything that is not the canonical encoding.

    Strictness here is not fussiness — it is what makes ``encode ∘ decode`` the
    identity on every byte string this function accepts, which is what gives one
    content exactly one name.
    """
    kind = peek_kind(framed)
    version = framed[1]
    if version not in _READABLE_VERSIONS:
        raise UnsupportedFormatVersion(
            "object format version is not readable by this build",
            version=version,
            readable=sorted(_READABLE_VERSIONS),
        )

    cursor = Cursor(framed[HEADER_BYTES:])
    obj: LedgerObject
    match kind:
        case ObjectKind.CHUNK:
            obj = _decode_chunk(cursor)
        case ObjectKind.BLOB:
            obj = _decode_blob(cursor, len(framed))
        case ObjectKind.TREE:
            obj = _decode_tree(cursor, len(framed))
        case ObjectKind.COMMIT:
            obj = _decode_commit(cursor)
    cursor.expect_exhausted()
    return obj


def decode_as[T: LedgerObject](framed: bytes, expected: type[T]) -> T:
    """Decode, requiring a particular object type.

    Callers almost always know what they asked for — ``get_tree`` wants a tree,
    the commit walker wants a commit. Without this they would either narrow with
    an ``isinstance`` at every call site or, more likely, not narrow at all and
    discover the mismatch as an ``AttributeError`` several frames away.

    Being wrong here is a real possibility rather than a theoretical one: a tree
    entry with the wrong ``kind`` points at an object of an unexpected type, and
    turning that into a clear error at the fetch is much better than a confusing
    failure during materialization.
    """
    obj = decode(framed)
    if not isinstance(obj, expected):
        raise MalformedObject(
            "object is not of the expected kind",
            expected=expected.__name__.lower(),
            actual=type(obj).__name__.lower(),
        )
    return obj


def _decode_chunk(cursor: Cursor) -> Chunk:
    chunk = Chunk(cursor.rest())
    _validate_chunk(chunk)
    return chunk


def _decode_blob(cursor: Cursor, framed_length: int) -> Blob:
    level = cursor.u8()
    count = cursor.u32()
    entries: list[BlobEntry] = []
    for _ in range(count):
        target = cursor.name()
        size = cursor.u64()
        if size <= 0:
            raise NotCanonical("a blob entry must cover at least one byte", size=size)
        entries.append(BlobEntry(target, size))

    _validate_node_level(level, count, "blob")
    _reject_oversized_node(framed_length, "blob")
    return Blob(level=level, entries=tuple(entries))


def _decode_tree(cursor: Cursor, framed_length: int) -> Tree:
    level = cursor.u8()
    count = cursor.u32()
    entries: list[TreeEntry] = []
    for _ in range(count):
        name = cursor.bytes_u8()
        kind_byte = cursor.u8()
        try:
            kind = EntryKind(kind_byte)
        except ValueError as exc:
            raise MalformedObject("unknown tree entry kind", kind=kind_byte) from exc
        entries.append(
            TreeEntry(
                name=name,
                kind=kind,
                target=cursor.name(),
                mode=cursor.u16(),
                size=cursor.u64(),
            )
        )

    _validate_node_level(level, count, "tree")
    tree = Tree(level=level, entries=tuple(entries))
    _validate_tree_entries(tree)
    _reject_oversized_node(framed_length, "tree")
    return tree


def _decode_commit(cursor: Cursor) -> Commit:
    tree = cursor.name()
    parents = tuple(cursor.name() for _ in range(cursor.u16()))
    change_id = ChangeId(cursor.raw(CHANGE_ID_BYTES).hex())
    author = _decode_utf8(cursor.bytes_u8(), "author")
    committer = _decode_utf8(cursor.bytes_u8(), "committer")
    timestamp_us = cursor.i64()
    message = _decode_utf8(cursor.bytes_u32(), "message")
    metadata = tuple(
        (
            _decode_utf8(cursor.bytes_u8(), "metadata key"),
            _decode_utf8(cursor.bytes_u32(), "metadata value"),
        )
        for _ in range(cursor.u32())
    )

    commit = Commit(
        tree=tree,
        parents=parents,
        change_id=change_id,
        author=author,
        committer=committer,
        timestamp_us=timestamp_us,
        message=message,
        metadata=metadata,
    )
    _validate_commit(commit)
    return commit


def _decode_utf8(raw: bytes, what: str) -> str:
    """Strict UTF-8. Surrogates and overlong forms are two spellings of one string."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NotCanonical(f"{what} is not valid UTF-8") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Validation — one implementation, called from both directions
# ─────────────────────────────────────────────────────────────────────────────


def _validate_chunk(chunk: Chunk) -> None:
    if not chunk.data:
        raise NotCanonical(
            "a zero-length chunk is not a valid object; an empty file is a blob with no entries"
        )
    if len(chunk.data) > MAX_CHUNK_BYTES:
        raise NotCanonical(
            "chunk exceeds the maximum chunk size",
            size=len(chunk.data),
            maximum=MAX_CHUNK_BYTES,
        )


def _validate_node_level(level: int, entry_count: int, what: str) -> None:
    """Depth invariants shared by blob index nodes and interior tree nodes.

    Both rules exist to remove a redundant spelling. An interior node with one
    child says nothing its child does not already say, and an interior node with
    no children says nothing at all — so admitting either would mean two
    encodings for one logical structure.
    """
    if entry_count == 0 and level != 0:
        raise NotCanonical(f"an empty {what} node must be at level 0", level=level)
    if level > 0 and entry_count < 2:
        raise NotCanonical(
            f"an interior {what} node needs at least two children, otherwise it "
            f"is a redundant spelling of its only child",
            level=level,
            entries=entry_count,
        )


def _validate_node_size(writer: Writer, what: str) -> None:
    _reject_oversized_node(len(writer), what)


def _reject_oversized_node(encoded_length: int, what: str) -> None:
    """No object is ever large — that is what keeps every other property true.

    The split rule already bounds node size, so exceeding this means either a
    builder bug or a client constructing objects by hand.
    """
    if encoded_length > MAX_NODE_BYTES:
        raise NotCanonical(
            f"{what} node exceeds the maximum node size",
            size=encoded_length,
            maximum=MAX_NODE_BYTES,
        )


def _validate_entry_name(name: bytes) -> None:
    if not 1 <= len(name) <= MAX_ENTRY_NAME_BYTES:
        raise NotCanonical(
            "tree entry name length out of range", length=len(name), maximum=MAX_ENTRY_NAME_BYTES
        )
    if _FORBIDDEN_NAME_BYTES & set(name):
        raise NotCanonical("tree entry name contains a separator or NUL", name=name.hex())
    if name in _RESERVED_NAMES:
        raise NotCanonical("tree entry name is reserved", name=name.decode())
    # Names are compared as bytes, but they must still *be* text: a name that is
    # not valid UTF-8 cannot be materialised onto a filesystem or rendered in a
    # listing, and admitting one would push the failure to the point of use.
    #
    # Unicode normalization is deliberately NOT required here. The identity of a
    # name is its bytes; normalizing would make two distinct names collide. The
    # hazard that creates on case-insensitive or normalizing filesystems is
    # handled where it actually bites, in the materializer.
    _decode_utf8(name, "tree entry name")


def _validate_tree_entries(tree: Tree) -> None:
    previous: bytes | None = None
    for entry in tree.entries:
        _validate_entry_name(entry.name)

        if previous is not None and entry.name <= previous:
            raise NotCanonical(
                "tree entries must be strictly ascending by unsigned byte order",
                previous=previous.decode(errors="replace"),
                current=entry.name.decode(errors="replace"),
            )
        previous = entry.name

        if entry.kind is EntryKind.CONFLICT:
            # A conflicted commit cannot be materialised or built,
            # so admitting one would force every downstream consumer to invent a
            # rule for it. The value is reserved so that adding support later is
            # a decoder change rather than a renumbering.
            raise NotCanonical("conflict entries are reserved and rejected in v1")

        if tree.level > 0:
            # Interior entries are routing keys: the name is the last key in the
            # subtree, and nothing else carries meaning.
            if entry.kind is not EntryKind.TREE or entry.mode != 0 or entry.size != 0:
                raise NotCanonical(
                    "an interior tree entry must be a bare subtree reference",
                    level=tree.level,
                    kind=entry.kind.name,
                    mode=entry.mode,
                    size=entry.size,
                )
            continue

        match entry.kind:
            case EntryKind.BLOB:
                if entry.mode not in _VALID_FILE_MODES:
                    raise NotCanonical("file mode must be 0o644 or 0o755", mode=f"{entry.mode:o}")
            case EntryKind.TREE:
                if entry.mode != 0 or entry.size != 0:
                    raise NotCanonical(
                        "a subtree entry carries no mode and no size; a recursive "
                        "size cannot be validated without fetching children",
                        mode=entry.mode,
                        size=entry.size,
                    )
            case EntryKind.SYMLINK:
                if entry.mode != 0:
                    raise NotCanonical("a symlink entry carries no mode", mode=entry.mode)
            case EntryKind.CONFLICT:  # pragma: no cover - rejected above
                raise AssertionError("unreachable")

        if entry.size < 0:
            raise NotCanonical("negative entry size", size=entry.size)


def _validate_commit(commit: Commit) -> None:
    if len(commit.change_id.value) != CHANGE_ID_BYTES * 2:
        raise NotCanonical("change id must be 16 bytes", length=len(commit.change_id.value) // 2)
    try:
        bytes.fromhex(commit.change_id.value)
    except ValueError as exc:
        raise NotCanonical("change id is not hex", change_id=commit.change_id.value) from exc

    # Parent *order* is meaningful — the first parent is the branch that was
    # being advanced — so parents are not sorted. Duplicates are still rejected:
    # naming the same parent twice says nothing the single reference does not.
    if len(set(commit.parents)) != len(commit.parents):
        raise NotCanonical("duplicate parent in commit", parents=[str(p) for p in commit.parents])

    if not commit.author or not commit.committer:
        raise NotCanonical("commit author and committer are required")

    previous_key: str | None = None
    for key, _ in commit.metadata:
        if not key:
            raise NotCanonical("empty metadata key")
        if previous_key is not None and key <= previous_key:
            raise NotCanonical(
                "commit metadata keys must be strictly ascending",
                previous=previous_key,
                current=key,
            )
        previous_key = key
