"""Housekeeping the system needs and nothing was scheduling.

Three reclamation mechanisms existed, were tested, and were never called by
anything in production. Each is small on its own and each leaks forever without a
caller:

* **Ephemeral refs never expired.** A branch created with a TTL kept its
  ``expires_at`` and nothing ever read it, so it stayed a live retention root
  indefinitely — which means the storage its content occupies is never returned.
  At the projected 0.1M actively-iterated environments running ten experiments
  each, that is a million refs pinning content nobody wants.

* **Orphan name claims were never reclaimed.** Creating an environment claims the
  name before writing the record, deliberately, so a crash between the two leaves
  a claimed-but-unused name — the safe direction, because the alternative is an
  environment nobody can address. Safe only if something later reclaims it;
  otherwise the name is permanently unusable and the error a caller gets is
  "already taken" for a name that belongs to nothing.

* **Tombstones never expired.** They exist to stop deduplication resurrecting a
  swept object, and they only need to outlive the longest in-flight write. Kept
  forever, the table grows without bound for no benefit.

Collection is where this belongs. All three answer the same question the
collector answers — *what is no longer needed* — and running them just before a
sweep is what makes the sweep's answer current: an ephemeral ref expired now is
content the very next diff can reclaim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from src.errors import LedgerError

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.clock import Clock
    from src.ids import EnvId, RefName
    from src.meta.models import Ref

__all__ = ["MaintenanceReport", "ReclaimSource", "run_maintenance"]


@final
@dataclass(frozen=True, slots=True)
class MaintenanceReport:
    """What one pass reclaimed. Reported rather than logged, so the CLI can print
    it and a test can assert on it.
    """

    expired_refs: int = 0
    orphan_names: int = 0
    purged_tombstones: int = 0

    @property
    def total(self) -> int:
        return self.expired_refs + self.orphan_names + self.purged_tombstones


@runtime_checkable
class ReclaimSource(Protocol):
    """The metadata reads a maintenance pass needs.

    Narrow for the same reason ``RootSource`` is: this package sits below
    ``meta``, and a housekeeping pass that could reach the whole repository could
    move a ref.
    """

    def expired_refs(self, *, limit: int = 1000) -> list[tuple[EnvId, Ref]]: ...

    def sweep_orphan_name_claims(self, *, limit: int = 100) -> int: ...


#: How an expired ref is discarded. Injected rather than called directly, because
#: deleting a ref is only half of a deletion — the environment's keep-set has to
#: shrink to match, and that lives in a layer above this one. Taking the whole
#: operation as a callable is what stops this module quietly doing the first half.
type Discard = Callable[[EnvId, RefName], object]


@runtime_checkable
class TombstoneSource(Protocol):
    def purge_expired(self, now_us: int) -> int: ...


def run_maintenance(
    roots: ReclaimSource,
    tombstones: TombstoneSource,
    *,
    clock: Clock,
    discard: Discard,
    limit: int = 1000,
) -> MaintenanceReport:
    """Expire what has expired. Safe to run at any time, and safe to interrupt.

    Every step is independently idempotent, so a pass that dies halfway leaves
    the system in a state the next pass finishes from. Nothing here is
    conditional on a previous step having run.
    """
    expired = 0
    for env, ref in roots.expired_refs(limit=limit):
        try:
            discard(env, ref.name)
        except LedgerError:
            # Someone moved or removed the ref between the scan and the delete,
            # so it is no longer the expired thing we read. Leaving it for the
            # next pass is strictly better than deleting a ref that just changed.
            continue
        expired += 1

    return MaintenanceReport(
        expired_refs=expired,
        orphan_names=roots.sweep_orphan_name_claims(limit=limit),
        purged_tombstones=tombstones.purge_expired(clock.now_us()),
    )
