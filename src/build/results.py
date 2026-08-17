"""Build results, keyed by commit and shared by everyone.

This is the module that makes "a build is a pure function of a commit" true. Its
result belongs to the *commit* rather than to the environment that happened to
trigger it — and once it does, three things stop being problems:

* two refs pointing at the same commit build once;
* **a fork that has changed nothing inherits its parent's build**, which is the
  difference between forking being free and forking costing a full rebuild of a
  forty-gigabyte environment;
* re-triggering an unchanged commit returns the existing result.

The claim is durable rather than in-memory. A worker writes ``RUNNING`` before it
does any work, so a second worker that reaches the same commit sees the claim
instead of spending five minutes reproducing it. That claim can expire: a worker
that dies mid-build must not make a commit unbuildable forever.

Because results are global rather than buried per environment, "how many builds
failed today" is one scan rather than a walk over ten million partitions — which
is what makes a systemic build regression one signal instead of ten million
silent ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

from src.build.models import BuildResult, BuildStatus
from src.meta.keys import GlobalSpace

if TYPE_CHECKING:
    from src.clock import Clock
    from src.ids import ObjectName
    from src.meta.store import MetadataStore

__all__ = ["ABANDONED_CLAIM_US", "BuildResults"]

#: How long a ``RUNNING`` claim is honoured before another worker may take it
#: over. Generously longer than the manifest timeout ceiling, because the cost of
#: taking over too early is two workers building the same commit, while the cost
#: of waiting is a delay — and only one of those can corrupt anything.
ABANDONED_CLAIM_US: Final = 2 * 3600 * 1_000_000


@final
class BuildResults:
    """The commit-keyed result store."""

    __slots__ = ("_clock", "_store")

    def __init__(self, store: MetadataStore, *, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    # ── reading ──────────────────────────────────────────────────────────────

    def get(self, commit: ObjectName | str) -> BuildResult | None:
        item = self._store.read_global(GlobalSpace.BUILDS, _key(commit))
        return None if item is None else BuildResult.from_body(item.body)

    def failures(self, *, limit: int = 100) -> list[BuildResult]:
        """Every failed build. One scan, not ten million partition reads."""
        return [
            result
            for item in self._store.scan_global(GlobalSpace.BUILDS, limit=limit * 4)
            if (result := BuildResult.from_body(item.body)).status is BuildStatus.FAILED
        ][:limit]

    # ── claiming ─────────────────────────────────────────────────────────────

    def claim(self, commit: ObjectName | str, *, worker: str) -> BuildResult | None:
        """Take responsibility for building ``commit``.

        Returns ``None`` when the claim succeeded and this worker should build.
        Returns the existing result when there is nothing to do — a **cache hit**,
        which is the ordinary case for a fork, a second ref, or a retrigger.

        A ``RUNNING`` claim from a worker that has gone silent past
        ``ABANDONED_CLAIM_US`` is taken over rather than waited on: a worker that
        died mid-build must not make a commit permanently unbuildable.
        """
        key = _key(commit)
        now = self._clock.now_us()
        running = BuildResult(
            commit=key, status=BuildStatus.RUNNING, started_at_us=now, worker=worker
        )

        if self._store.claim_global(GlobalSpace.BUILDS, key, running.to_body()):
            return None

        existing = self.get(commit)
        if existing is None:  # pragma: no cover - claim lost then deleted
            return None
        if existing.status.terminal:
            return existing
        if now - existing.started_at_us < ABANDONED_CLAIM_US:
            return existing  # someone else is genuinely building it

        # Abandoned. Take it over, keeping the attempt count so a commit that
        # kills workers repeatedly is visible as such rather than looking new.
        self._store.put_global(
            GlobalSpace.BUILDS,
            key,
            BuildResult(
                commit=key,
                status=BuildStatus.RUNNING,
                started_at_us=now,
                attempts=existing.attempts + 1,
                worker=worker,
            ).to_body(),
        )
        return None

    def record(self, result: BuildResult) -> None:
        """Write a terminal result. Last writer wins, which is correct here:
        the value is a function of the commit, so two writers agree.
        """
        self._store.put_global(GlobalSpace.BUILDS, result.commit, result.to_body())

    def release(self, commit: ObjectName | str) -> None:
        """Drop an unfinished claim, so the commit can be built again.

        Used when a worker gives up cleanly — a lost lease, a shutdown — as
        opposed to failing, which records a failure someone should look at.
        """
        existing = self.get(commit)
        if existing is not None and not existing.status.terminal:
            self._store.delete_global(GlobalSpace.BUILDS, _key(commit))

    def invalidate(self, commit: ObjectName | str) -> bool:
        """Forget a finished result, so the next build genuinely rebuilds.

        The honest way to say "build it again". A build is a pure function of a
        commit, so asking for a rebuild is a statement that the previous *answer*
        is no longer trusted — and that has to be said where every consumer sees
        it, rather than by letting one caller quietly ignore a result the others
        still believe. Removing it is deliberate and rare; caching is the default
        because the purity of a build makes it correct.
        """
        return self._store.delete_global(GlobalSpace.BUILDS, _key(commit))


def _key(commit: ObjectName | str) -> str:
    return str(commit)
