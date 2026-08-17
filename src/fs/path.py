"""Path parsing. Small, and worth its own module because it is a security boundary.

Every path that enters Ledger from outside — an API request, a CLI argument, a
git import — passes through here. The rules are the codec's rules:
a component is 1–255 bytes, no separator, no NUL, not ``.`` or ``..``), applied
one step earlier so a bad path is a 400 rather than a failure deep inside a walk.

Paths are handled as *bytes* rather than ``str`` for the same reason names are:
the identity of a path component is its bytes, and a locale- or
normalization-aware comparison would make two distinct entries collide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.errors import InvalidRequest
from src.format.constants import MAX_ENTRY_NAME_BYTES

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["format_path", "parse_path"]

_SEPARATOR: Final = b"/"
_RESERVED: Final = frozenset({b".", b".."})


def validate_component(component: bytes) -> bytes:
    """Check one path component, returning it unchanged.

    Returns rather than asserts so it composes into comprehensions without a
    separate statement — the call site reads as a filter, and forgetting to call
    it is then visible rather than silent.
    """
    if not 1 <= len(component) <= MAX_ENTRY_NAME_BYTES:
        raise InvalidRequest(
            "path component length out of range",
            length=len(component),
            maximum=MAX_ENTRY_NAME_BYTES,
        )
    if _SEPARATOR in component or b"\x00" in component:
        raise InvalidRequest("path component contains a separator or NUL")
    if component in _RESERVED:
        raise InvalidRequest("path component is reserved", component=component.decode())
    try:
        component.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidRequest("path component is not valid UTF-8") from exc
    return component


def parse_path(path: str | bytes) -> tuple[bytes, ...]:
    """Split a path into validated components.

    Leading and trailing separators are tolerated and repeated ones collapse,
    because callers write ``/data/train.bin`` and ``data/train.bin`` and mean the
    same thing. What is *not* tolerated is ``..`` — this is the one place a
    traversal attempt can enter, and it is rejected rather than normalised away,
    because normalising it would silently resolve to a different file than the
    caller asked for.
    """
    raw = path.encode() if isinstance(path, str) else path
    components = tuple(part for part in raw.split(_SEPARATOR) if part)
    for component in components:
        validate_component(component)
    return components


def format_path(components: Sequence[bytes]) -> str:
    """Render components back to a display path."""
    return "/".join(c.decode(errors="replace") for c in components)
