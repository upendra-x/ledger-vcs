"""Identifier value types.

Every identifier in Ledger is a distinct type rather than a bare ``str``. That is
not ceremony: the system routinely holds an object name, a change id, an
environment id and a session id in the same scope, all of which render as hex,
and a mix-up between them is exactly the class of bug that content addressing
otherwise makes impossible to detect. A wrong ``ObjectName`` is a 404; a wrong
``str`` that happens to parse is a silent read of someone else's content.

``ObjectName`` is the load-bearing one. It carries its hash algorithm because
retiring BLAKE3 must be a new prefix rather than an ambiguity: old
objects stay valid and still verify under the algorithm they were named with.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Final, Self, final

__all__ = [
    "ChangeId",
    "EnvId",
    "EnvName",
    "EpochId",
    "ObjectName",
    "RefName",
    "SessionId",
]

# ─────────────────────────────────────────────────────────────────────────────
# Object names
# ─────────────────────────────────────────────────────────────────────────────

#: Rendered prefix for the one hash algorithm v1 emits. A successor algorithm
#: gets a *new* prefix, never a redefinition of this one.
BLAKE3_ALGO: Final = "b3"

#: BLAKE3-256 — 32 bytes of digest.
DIGEST_BYTES: Final = 32

_NAME_RE: Final = re.compile(rf"^{BLAKE3_ALGO}:([0-9a-f]{{{DIGEST_BYTES * 2}}})$")


@final
@dataclass(frozen=True, slots=True, order=True)
class ObjectName:
    """The name of an immutable object: ``b3:<64 hex chars>``.

    Ordering is by raw digest, which is what the garbage collector's sharded
    diff and the keep-set's sorted-array representation both rely on.
    """

    digest: bytes

    def __post_init__(self) -> None:
        if len(self.digest) != DIGEST_BYTES:
            raise ValueError(f"object digest must be {DIGEST_BYTES} bytes, got {len(self.digest)}")

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse the rendered form. Raises ``ValueError`` on anything else.

        Deliberately strict: lowercase hex only, exact length, algorithm prefix
        required. A permissive parser here would let two spellings of one name
        into the system, and dedup is spelling-sensitive.
        """
        match = _NAME_RE.match(text)
        if match is None:
            raise ValueError(f"not a valid object name: {text!r}")
        return cls(bytes.fromhex(match.group(1)))

    @property
    def hex(self) -> str:
        """The digest as 64 lowercase hex characters, without the prefix."""
        return self.digest.hex()

    @property
    def algo(self) -> str:
        return BLAKE3_ALGO

    def __str__(self) -> str:
        return f"{BLAKE3_ALGO}:{self.digest.hex()}"

    def __repr__(self) -> str:
        return f"ObjectName({self})"


# ─────────────────────────────────────────────────────────────────────────────
# Mutable-side identifiers
# ─────────────────────────────────────────────────────────────────────────────

_ULID_ALPHABET: Final = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
_ULID_RANDOM_BITS: Final = 80
_ULID_MAX_RANDOM: Final = (1 << _ULID_RANDOM_BITS) - 1

_ulid_lock = threading.Lock()
_ulid_last_ms = -1
_ulid_last_random = 0


def _new_ulid() -> str:
    """A lexicographically sortable, time-ordered 26-character identifier.

    Monotonic, in the sense the ULID specification uses: two ids minted inside
    the same millisecond still sort in the order they were created, because the
    second one increments the first one's random component rather than drawing
    fresh bits.

    That is not decoration. ``ListEnvs`` paginates by id with no secondary
    index, and automations create environments in tight bursts — so without
    monotonicity, "list environments in creation order" is wrong precisely when
    many are created at once, which is the case anyone would look at.

    Implemented here rather than taken as a dependency because it is twenty
    lines and because it has to stay stable forever.
    """
    global _ulid_last_ms, _ulid_last_random

    with _ulid_lock:
        now_ms = time.time_ns() // 1_000_000
        if now_ms > _ulid_last_ms:
            _ulid_last_ms = now_ms
            _ulid_last_random = secrets.randbits(_ULID_RANDOM_BITS)
        else:
            # Same millisecond, or a clock that stepped backwards. Keep the
            # previous timestamp and step the randomness, so ordering holds
            # either way. Overflow after 2^80 ids in one millisecond is not a
            # case worth branching on, but it must not silently wrap into a
            # duplicate, so it borrows a millisecond from the future.
            if _ulid_last_random == _ULID_MAX_RANDOM:
                _ulid_last_ms += 1
                _ulid_last_random = secrets.randbits(_ULID_RANDOM_BITS)
            else:
                _ulid_last_random += 1
        value = (_ulid_last_ms << _ULID_RANDOM_BITS) | _ulid_last_random

    chars = []
    for _ in range(26):
        chars.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


@final
@dataclass(frozen=True, slots=True, order=True)
class EnvId:
    """Immutable identity of an environment. Survives every rename."""

    value: str

    @classmethod
    def new(cls) -> Self:
        return cls(f"env_{_new_ulid()}")

    def __str__(self) -> str:
        return self.value


@final
@dataclass(frozen=True, slots=True, order=True)
class ChangeId:
    """Stable identity of a *change*, surviving amendment and rebase.

    Distinct from an ``ObjectName``: a commit's name is its content hash and
    moves whenever anything changes, while a change id is assigned once and
    carried through rewrites so an automation can refer to "the change that adds
    the verifier" across a rebase.
    """

    value: str

    @classmethod
    def new(cls) -> Self:
        return cls(secrets.token_hex(16))

    def __str__(self) -> str:
        return self.value


_ENV_NAME_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,127}$")


@final
@dataclass(frozen=True, slots=True, order=True)
class EnvName:
    """A mutable, globally unique ``org/environment`` name.

    Mirrors GitHub's owner/repo so migration is one-to-one and so authorization
    has a natural prefix to scope grants against.
    """

    value: str

    def __post_init__(self) -> None:
        if _ENV_NAME_RE.match(self.value) is None:
            raise ValueError(f"not a valid environment name: {self.value!r}")

    @property
    def org(self) -> str:
        return self.value.split("/", 1)[0]

    def __str__(self) -> str:
        return self.value


_REF_NAME_RE: Final = re.compile(r"^refs/(heads|tags)/[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")


@final
@dataclass(frozen=True, slots=True, order=True)
class RefName:
    """``refs/heads/main``, ``refs/tags/v1``. The only mutable pointer at content."""

    value: str

    def __post_init__(self) -> None:
        if _REF_NAME_RE.match(self.value) is None:
            raise ValueError(f"not a valid ref name: {self.value!r}")
        if "//" in self.value or self.value.endswith("/") or ".." in self.value:
            raise ValueError(f"not a valid ref name: {self.value!r}")

    @property
    def is_tag(self) -> bool:
        """Tags are written once and then frozen — create-if-absent, not CAS."""
        return self.value.startswith("refs/tags/")

    def __str__(self) -> str:
        return self.value


@final
@dataclass(frozen=True, slots=True, order=True)
class SessionId:
    """A write session. Its lease keeps uploaded-but-unreferenced objects alive."""

    value: str

    @classmethod
    def new(cls) -> Self:
        return cls(f"ws_{_new_ulid()}")

    def __str__(self) -> str:
        return self.value


@final
@dataclass(frozen=True, slots=True, order=True)
class EpochId:
    """One garbage-collection cycle."""

    value: int

    def __str__(self) -> str:
        return f"epoch_{self.value:010d}"
