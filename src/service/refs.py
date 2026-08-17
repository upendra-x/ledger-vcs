"""Creating and discarding refs — the operations that change what is reachable.

Both were previously written out at each call site, and the two copies had
already drifted in a way that mattered.

**Creating a branch** needs the environment's default ref resolved when no target
is given, and a TTL converted into an expiry. That was duplicated verbatim
between the HTTP route and the CLI.

**Discarding one** has to rebuild the environment's keep-set, and *nothing did*.
Keep-sets are maintained rather than recomputed: adding is free because a
commit's closure never changes, and only subtraction costs anything — which
happens exactly once, on ref deletion. The repository's ``delete_ref`` removes
the ref and nothing else, so through the API or the CLI a discarded branch stayed
in the keep-set forever and its content was never reclaimed. The demonstration
happened to be correct only because it called ``rebuild_keep_set`` by hand.

That is the whole reason this module exists: the rebuild is not an optimisation a
caller may skip, it is the second half of the deletion. Putting the two together
means there is one place to get it right, and no way to do half of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from src.meta.models import RefLifecycle

if TYPE_CHECKING:
    from src.ids import EnvId, ObjectName, RefName
    from src.instance import Ledger
    from src.meta.repository import RefUpdate

__all__ = ["DeleteOutcome", "RefService"]

_MICROSECONDS_PER_DAY = 86_400 * 1_000_000


@final
@dataclass(frozen=True, slots=True)
class DeleteOutcome:
    """What discarding a ref did, including the part that reclaims storage."""

    name: RefName
    #: Objects the environment's keep-set holds after the rebuild. The drop from
    #: its previous size is what a discarded branch actually freed.
    keep_set_size: int


@final
class RefService:
    """Ref lifecycle, with the storage consequences attached."""

    __slots__ = ("_ledger",)

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def create(
        self,
        env: EnvId,
        name: RefName,
        *,
        target: ObjectName | None = None,
        principal: str,
        ephemeral: bool = False,
        ttl_days: int = 14,
        idempotency_key: str | None = None,
    ) -> RefUpdate:
        """Open a branch. Costs no bytes — one metadata row over existing content.

        With no target it branches from the environment's *default ref*, which is
        what "branch from here" means to every caller that did not say otherwise.
        """
        repo = self._ledger.repo
        if target is None:
            default = repo.get_env(env).default_ref
            target = repo.get_ref(env, default).target
        return repo.create_ref(
            env,
            name,
            target,
            principal=principal,
            lifecycle=RefLifecycle.EPHEMERAL if ephemeral else RefLifecycle.PERMANENT,
            ttl_us=ttl_days * _MICROSECONDS_PER_DAY if ephemeral else None,
            idempotency_key=idempotency_key,
        )

    def delete(
        self,
        env: EnvId,
        name: RefName,
        *,
        expected_generation: int | None = None,
        principal: str,
        idempotency_key: str | None = None,
    ) -> DeleteOutcome:
        """Discard a branch, and shrink the environment's keep-set to match.

        Instant and synchronous from the caller's side — no automation ever waits
        on collection. The rebuild that follows decides what a *later* sweep may
        take; it frees nothing by itself, and it never takes content another ref
        or a retained operation-log entry still reaches.

        ``expected_generation`` defaults to whatever the ref currently holds,
        which is what a person at a terminal means by "delete this branch". An
        automation passes the generation it read, and gets the usual conflict if
        somebody moved the ref underneath it.
        """
        repo = self._ledger.repo
        if expected_generation is None:
            expected_generation = repo.get_ref(env, name).generation

        repo.delete_ref(
            env,
            name,
            expected_generation=expected_generation,
            principal=principal,
            idempotency_key=idempotency_key,
        )
        # The second half of the deletion, not an afterthought: without it the
        # branch's exclusive content stays in the keep-set and is never collected.
        remaining = self._ledger.gc.rebuild_keep_set(str(repo.get_env(env).name))
        return DeleteOutcome(name=name, keep_set_size=remaining)
