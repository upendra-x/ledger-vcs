"""Forking — permanent, and free.

A fork creates a new environment whose default ref points at an
existing commit. Because objects carry no environment identity, **nothing
is copied**::

    proximal/base-env    env_01J8…  refs/heads/main ──▶ c9f2…
                                                          ▲
    proximal/variant-a   env_01K2…  refs/heads/main ───────┤   2 rows, 0 bytes
    proximal/variant-b   env_01K3…  refs/heads/main ───────┘   2 rows, 0 bytes

From that moment the two are independent: a commit to the fork touches only the
fork's partition, so a busy fork cannot slow its parent.

One piece of asynchronous bookkeeping remains. The fork's *keep-set* has to know
it reaches this content, or the first ``DeleteRef`` in the source environment
would sweep objects the fork still needs. That walk covers trees, blob manifests
and index nodes — proportional to the environment's *shape* rather than its size.

``forked_from`` records provenance, and is deliberately **not** a storage
dependency: archiving or deleting the parent does not affect the fork, because
retention is decided per-ref over globally shared objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, final

from src.fs.closure import commit_closure
from src.meta.keys import ItemKind, env_sk
from src.meta.store import Key, Put

if TYPE_CHECKING:
    from src.ids import EnvId, EnvName, ObjectName, RefName
    from src.instance import Ledger
    from src.meta.models import Environment

__all__ = ["EnvironmentService", "ForkResult"]


@final
@dataclass(frozen=True, slots=True)
class ForkResult:
    environment: Environment
    source_commit: ObjectName
    #: Objects the fork's keep-set now reaches. **Zero of them were copied** —
    #: this is the count of names recorded, not of bytes moved.
    closure_size: int
    bytes_copied: int = 0


@final
class EnvironmentService:
    __slots__ = ("_ledger",)

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def fork(
        self,
        source: EnvId,
        name: EnvName,
        *,
        from_ref: RefName | None = None,
        owner: str = "",
        principal: str = "",
        idempotency_key: str | None = None,
    ) -> ForkResult:
        """Fork an environment. Copies no bytes."""
        ledger = self._ledger
        origin = ledger.repo.get_env(source)
        ref_name = from_ref or origin.default_ref
        head = ledger.repo.get_ref(source, ref_name)

        created = ledger.repo.create_env(
            name, owner=owner or origin.owner, idempotency_key=idempotency_key
        )
        provenance = replace(
            ledger.repo.get_env(created.env_id),
            forked_from_env=source,
            forked_from_commit=head.target,
        )
        ledger.meta.transact_write(
            [
                Put(
                    Key(str(created.env_id), env_sk()),
                    ItemKind.ENV,
                    provenance.to_body(),
                )
            ]
        )

        ledger.repo.create_ref(
            created.env_id,
            created.default_ref,
            head.target,
            principal=principal or owner or "ledger",
        )

        # Off the request path in production; done inline here
        # because at one-machine scale it is milliseconds, and skipping it would
        # leave the fork's keep-set empty while its ref reaches live content.
        closure = commit_closure(self._ledger.store, head.target)
        session = ledger.repo.begin_write(created.env_id, principal=principal or "ledger")
        try:
            ledger.repo.record_uploaded(created.env_id, session.session_id, closure)
            # Without this the fork's keep-set would be empty while its ref
            # reached live content, and the first DeleteRef in the source
            # environment would sweep objects the fork still needs.
            ledger.gc.graduate(str(created.env_id), closure)
        finally:
            ledger.repo.end_session(created.env_id, session.session_id)

        return ForkResult(
            environment=provenance, source_commit=head.target, closure_size=len(closure)
        )
