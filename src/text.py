"""Formatting humans read. Pure functions over values — no I/O, no state.

These live at the root rather than in a presentation layer because both the CLI
and the read-only browser render the same quantities, and two implementations of
"how big is that" drift into disagreeing about the same number. There were seven
copies of this before it existed.

Nothing here is inside a hash. That is the reason it may change freely, and the
reason it must never be imported by ``src.format``.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["exact_bytes", "human_bytes", "when"]

_UNITS: tuple[tuple[str, int], ...] = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


def human_bytes(count: int, *, precision: int = 2) -> str:
    """A byte count at the largest unit that leaves a digit in front of the point.

    ``precision`` exists because a measurement and a summary want different
    resolutions: a cost table earns two decimals, a directory listing does not.
    """
    for unit, size in _UNITS:
        if count >= size:
            return f"{count / size:.{precision}f} {unit}"
    return f"{count} B"


def exact_bytes(count: int) -> str:
    """A byte count as a whole number of units, or as plain bytes.

    For *constants* rather than measurements. ``256 KiB`` reads as a decision
    somebody made; ``256.00 KiB`` reads as something that was measured, which is
    exactly the wrong impression for a frozen parameter.
    """
    for unit, size in _UNITS[1:]:
        if count >= size and count % size == 0:
            return f"{count // size} {unit}"
    return f"{count} B"


def when(timestamp_us: int) -> str:
    """A microsecond timestamp as UTC, to the second.

    Always UTC: these are read next to values from other machines, and a local
    rendering makes two log lines silently incomparable.
    """
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
