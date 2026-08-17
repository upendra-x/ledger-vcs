"""Byte-level reader and writer primitives for the canonical encoding.

Two decisions are made here and nowhere else.

**Fixed-width big-endian integers, no varints.** LEB128 and friends have
non-minimal forms — ``0x81 0x00`` and ``0x01`` both decode to 1 — so a varint
encoding is only canonical if every decoder rejects the non-minimal spelling,
and that is a check people forget. Fixed width makes canonicality free rather
than enforced. The cost is real and small: on a 43 GiB environment the extra
bytes total roughly 220 KB, or 0.0005%.

**Every read is bounds-checked, and the reader must be exhausted.** A decoder
that silently ignores trailing bytes admits two byte strings that decode to the
same object, which is the same corpus-forking failure as a non-minimal varint.
``Cursor.expect_exhausted`` is not optional politeness; it is half of what
makes one object have exactly one encoding.
"""

from __future__ import annotations

from typing import Final, final

from src.errors import MalformedObject
from src.ids import ObjectName

__all__ = ["Cursor", "Writer"]

_U8_MAX: Final = 0xFF
_U16_MAX: Final = 0xFFFF
_U32_MAX: Final = 0xFFFF_FFFF
_U64_MAX: Final = 0xFFFF_FFFF_FFFF_FFFF
_I64_MIN: Final = -(1 << 63)
_I64_MAX: Final = (1 << 63) - 1

_NAME_BYTES: Final = 32


@final
class Writer:
    """Accumulates the canonical encoding of one object.

    Range checks are assertions about program correctness, not input validation:
    reaching them means a builder constructed an object that cannot be
    represented, which is a bug here rather than bad data from a client.
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def u8(self, value: int) -> Writer:
        if not 0 <= value <= _U8_MAX:
            raise ValueError(f"u8 out of range: {value}")
        self._buf.append(value)
        return self

    def u16(self, value: int) -> Writer:
        if not 0 <= value <= _U16_MAX:
            raise ValueError(f"u16 out of range: {value}")
        self._buf += value.to_bytes(2, "big")
        return self

    def u32(self, value: int) -> Writer:
        if not 0 <= value <= _U32_MAX:
            raise ValueError(f"u32 out of range: {value}")
        self._buf += value.to_bytes(4, "big")
        return self

    def u64(self, value: int) -> Writer:
        if not 0 <= value <= _U64_MAX:
            raise ValueError(f"u64 out of range: {value}")
        self._buf += value.to_bytes(8, "big")
        return self

    def i64(self, value: int) -> Writer:
        if not _I64_MIN <= value <= _I64_MAX:
            raise ValueError(f"i64 out of range: {value}")
        self._buf += value.to_bytes(8, "big", signed=True)
        return self

    def raw(self, data: bytes) -> Writer:
        self._buf += data
        return self

    def name(self, value: ObjectName) -> Writer:
        """A fixed 32-byte digest. Never length-prefixed — the width is frozen."""
        self._buf += value.digest
        return self

    def bytes_u8(self, data: bytes) -> Writer:
        """Length-prefixed by one byte. For path components and metadata keys."""
        if len(data) > _U8_MAX:
            raise ValueError(f"value too long for a u8 length prefix: {len(data)}")
        self._buf.append(len(data))
        self._buf += data
        return self

    def bytes_u32(self, data: bytes) -> Writer:
        """Length-prefixed by four bytes. For messages and metadata values."""
        if len(data) > _U32_MAX:
            raise ValueError(f"value too long for a u32 length prefix: {len(data)}")
        self._buf += len(data).to_bytes(4, "big")
        self._buf += data
        return self

    def finish(self) -> bytes:
        return bytes(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


@final
class Cursor:
    """A bounds-checked reader over an encoded object.

    Every failure is a ``MalformedObject``: the bytes came from a client, or from
    storage that has been damaged, and either way the answer is an error rather
    than a partially-populated object.
    """

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def _take(self, n: int) -> bytes:
        if n < 0:
            raise ValueError(f"negative read length: {n}")
        end = self._pos + n
        if end > len(self._data):
            raise MalformedObject(
                "truncated object",
                needed=n,
                available=self.remaining,
                at_offset=self._pos,
            )
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self._take(2), "big")

    def u32(self) -> int:
        return int.from_bytes(self._take(4), "big")

    def u64(self) -> int:
        return int.from_bytes(self._take(8), "big")

    def i64(self) -> int:
        return int.from_bytes(self._take(8), "big", signed=True)

    def raw(self, n: int) -> bytes:
        return self._take(n)

    def rest(self) -> bytes:
        """Everything left. Used only by CHUNK, whose payload is opaque bytes."""
        return self._take(self.remaining)

    def name(self) -> ObjectName:
        return ObjectName(self._take(_NAME_BYTES))

    def bytes_u8(self) -> bytes:
        return self._take(self._take(1)[0])

    def bytes_u32(self) -> bytes:
        return self._take(int.from_bytes(self._take(4), "big"))

    def expect_exhausted(self) -> None:
        """Trailing bytes are an error, never ignored.

        Tolerating them would let two distinct byte strings decode to the same
        object — so the same logical object would have two valid names, and
        deduplication would depend on which one a writer happened to send.
        """
        if self.remaining:
            raise MalformedObject(
                "trailing bytes after a complete object",
                trailing=self.remaining,
                at_offset=self._pos,
            )
