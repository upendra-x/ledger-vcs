"""Pushing a built version into the platform.

The contract is deliberately narrow: the commit hash as version identity, the
built image digests, the manifest metadata the platform indexes on, and a
completion callback. Nothing more, in either direction — **the platform never
reaches into Ledger's storage and Ledger never models the platform's schema.** A
platform outage stalls sync and nothing else.

Delivery is keyed by a **commit-derived idempotency key**, which is what makes
the at-least-once queue safe: a duplicated delivery updates the same platform
entry rather than creating a second one. There is nothing to deduplicate because
there is nothing that could be ambiguous — the key is a function of what is being
synced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, final

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["PlatformSync", "RecordingPlatform", "SyncOutcome", "SyncRequest", "sync_key"]


@final
@dataclass(frozen=True, slots=True)
class SyncRequest:
    """What the platform is told about a version."""

    commit: str
    env_id: str
    env_name: str
    images: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def idempotency_key(self) -> str:
        """Derived, never generated.

        A generated key would differ between the first delivery and its retry,
        which is precisely the case it exists to handle. Deriving it from the
        environment and the commit means a retry carries the same key by
        construction.
        """
        return sync_key(self.env_id, self.commit)


def sync_key(env_id: str, commit: str) -> str:
    return f"sync:{env_id}:{commit}"


@final
@dataclass(frozen=True, slots=True)
class SyncOutcome:
    platform_id: str
    synced_at_us: int
    #: True when the platform recognised the key and updated rather than
    #: created. Worth surfacing: it is the observable form of "at-least-once
    #: delivery, exactly-once effect".
    deduplicated: bool = False


class PlatformSync(Protocol):
    def sync(self, request: SyncRequest, *, now_us: int) -> SyncOutcome: ...


@final
class RecordingPlatform:
    """An in-process stand-in for the platform.

    It is a real implementation of the contract rather than a mock: it enforces
    idempotency the way the platform is required to, so a duplicated delivery
    updates an entry instead of creating a second one. Swapping it for an HTTP
    client changes this file and nothing else — which is the point of the
    contract being four fields wide.
    """

    __slots__ = ("_by_key", "deliveries")

    def __init__(self) -> None:
        self._by_key: dict[str, dict[str, Any]] = {}
        #: Every delivery, including duplicates, so a test can assert that a
        #: retry was *received* and still did not duplicate the entry.
        self.deliveries: list[SyncRequest] = []

    def sync(self, request: SyncRequest, *, now_us: int) -> SyncOutcome:
        self.deliveries.append(request)
        key = request.idempotency_key
        existing = self._by_key.get(key)

        entry = {
            "platform_id": existing["platform_id"] if existing else f"env-{len(self._by_key) + 1}",
            "commit": request.commit,
            "env_id": request.env_id,
            "env_name": request.env_name,
            "images": list(request.images),
            "metadata": dict(request.metadata),
            "synced_at_us": now_us,
        }
        self._by_key[key] = entry
        return SyncOutcome(
            platform_id=str(entry["platform_id"]),
            synced_at_us=now_us,
            deduplicated=existing is not None,
        )

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        return dict(self._by_key)

    def entry_for(self, env_id: str, commit: str) -> dict[str, Any] | None:
        return self._by_key.get(sync_key(env_id, commit))
