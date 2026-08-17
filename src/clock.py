"""Time, injected rather than ambient.

Leases expire, idempotency records age out, ephemeral refs have a TTL, and the
garbage collector's grace period is the guard standing between a paused writer
and data loss. Every one of those is a *correctness* boundary, and none of them
can be tested against ``time.time()``.

So no module below the API layer calls the clock directly. They take a ``Clock``,
and tests hand them a ``ManualClock`` that only moves when the test moves it.
The garbage-collection tests in particular depend on this: proving the grace
period works means advancing seven days, not sleeping through them.

Microseconds are the unit throughout. Integers, because float seconds lose
precision at epoch scale and two events that must be ordered can compare equal.
"""

from __future__ import annotations

import time
from typing import Protocol, final, runtime_checkable

__all__ = ["Clock", "ManualClock", "SystemClock"]

MICROSECONDS_PER_SECOND = 1_000_000


@runtime_checkable
class Clock(Protocol):
    """Wall-clock time in microseconds since the Unix epoch."""

    def now_us(self) -> int: ...


@final
class SystemClock:
    """The real clock. The only implementation wired in production."""

    __slots__ = ()

    def now_us(self) -> int:
        return time.time_ns() // 1000

    def __repr__(self) -> str:
        return "SystemClock()"


@final
class ManualClock:
    """A clock that moves only when a test moves it.

    Not thread-safe by design — a test that needs concurrent time control has a
    more interesting problem than this class should hide.
    """

    __slots__ = ("_now_us",)

    def __init__(self, start_us: int = 0) -> None:
        self._now_us = start_us

    def now_us(self) -> int:
        return self._now_us

    def advance_us(self, delta_us: int) -> int:
        if delta_us < 0:
            raise ValueError("time does not run backwards")
        self._now_us += delta_us
        return self._now_us

    def advance_seconds(self, seconds: float) -> int:
        return self.advance_us(int(seconds * MICROSECONDS_PER_SECOND))

    def advance_days(self, days: float) -> int:
        return self.advance_seconds(days * 86_400)

    def __repr__(self) -> str:
        return f"ManualClock(now_us={self._now_us})"
