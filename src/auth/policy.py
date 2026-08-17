"""The policy engine: decide every request from the token that carries it.

**Authority is resolved at mint time**, so a request costs a signature
verification and never a store read. A token states what it may do;
nothing here looks anything up to find out.

**Commit-to-environment binding is enforced here, and it is a gap in the
specification.** Possessing a hash must not be enough to fetch
bytes, because hashes leak. But that defence is defeated one level up unless a
*commit* is also checked against the environment a token names: otherwise a
commit hash observed from environment B, presented with a valid token for
environment A, reads B's content through A's authorization. So a read by commit
must show the commit is reachable from that environment's refs.

The check is a bounded ancestor walk, memoized. In the overwhelmingly common
case — reading the commit a ref currently points at — it is one comparison.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Final, final

from src.errors import Forbidden, LedgerError
from src.format.model import Commit

if TYPE_CHECKING:
    from src.auth.model import Operation
    from src.auth.tokens import Capability
    from src.ids import EnvId, ObjectName
    from src.instance import Ledger

__all__ = ["PolicyEngine", "ReachabilityChecker"]

#: How far back a commit may sit from a live ref and still be considered part of
#: an environment. Generous — the capacity plan projects fifty versions per environment —
#: and bounded so a hostile request cannot walk forever.
MAX_ANCESTRY_WALK: Final = 10_000


#: How many (environment, commit) answers one process keeps.
#:
#: Bounded because this lives on a long-running API plane and the corpus does
#: not. Every ancestry walk memoizes every commit it passes, so an unbounded set
#: grows with *traffic* — at the projected five hundred million commits, a
#: process that stayed up long enough would eventually hold a tuple per commit it
#: had ever been asked about. Least-recently-used, because the access pattern is
#: overwhelmingly "the commit a ref points at right now".
MAX_MEMOIZED_ANSWERS: Final = 100_000


@final
class ReachabilityChecker:
    """Is this commit part of this environment?

    Memoized per (environment, commit) because the answer cannot change in a way
    that *removes* reachability while a ref still points through it — history is
    append-only under a ref, and a ref moving on only adds ancestors. That
    one-directional property is what makes caching a *yes* safe and caching a
    *no* wrong: a commit unreachable now may be reachable after the next push, so
    only positives are kept.
    """

    __slots__ = ("_ledger", "_yes")

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger
        #: An ordered mapping used as a bounded set — the value is never read.
        self._yes: OrderedDict[tuple[str, str], None] = OrderedDict()

    def reaches(self, env: EnvId, commit: ObjectName) -> bool:
        if self._remembered(str(env), str(commit)):
            return True

        refs = self._ledger.repo.list_refs(env)
        frontier = [ref.target for ref in refs]
        # The common case, and the one worth being fast: reading the commit a
        # ref currently points at.
        if commit in frontier:
            self._remember(str(env), str(commit))
            return True

        seen: set[ObjectName] = set()
        walked = 0
        while frontier and walked < MAX_ANCESTRY_WALK:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            walked += 1
            self._remember(str(env), str(current))
            if current == commit:
                return True
            try:
                frontier.extend(self._ledger.store.get_as(current, Commit).parents)
            except LedgerError:
                # A parent this store does not hold — a partial replica, or a
                # commit whose ancestry was swept. It contributes no ancestors;
                # it must not stop the walk finding the answer down another edge.
                continue
        return False

    def _remembered(self, env: str, commit: str) -> bool:
        key = (env, commit)
        if key not in self._yes:
            return False
        self._yes.move_to_end(key)
        return True

    def _remember(self, env: str, commit: str) -> None:
        self._yes[(env, commit)] = None
        self._yes.move_to_end((env, commit))
        while len(self._yes) > MAX_MEMOIZED_ANSWERS:
            self._yes.popitem(last=False)


@final
class PolicyEngine:
    """Decides every request, before any resolution work happens."""

    __slots__ = ("_ledger", "_reachability")

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger
        self._reachability = ReachabilityChecker(ledger)

    # ── enforcement ──────────────────────────────────────────────────────────

    def authorize(self, capability: Capability, operation: Operation, env: EnvId) -> None:
        """Raise ``Forbidden`` unless this token may do this to this environment.

        The 403 deliberately carries no information about whether the target
        exists: a 403 that leaked existence would be an enumeration oracle over
        the corpus.
        """
        record = self._ledger.repo.get_env(env)
        if not capability.scope.permits(operation, env, record.name, record.labels):
            raise Forbidden(
                "this token does not permit that operation on that environment",
                operation=operation.label,
            )

    def authorize_commit(self, capability: Capability, env: EnvId, commit: ObjectName) -> None:
        """Bind a commit to the environment a token names.

        Without this, "no bare-hash read" is defeated one level up — a commit
        hash from environment B, presented with a valid token for A, would read
        B through A's authorization.
        """
        if capability.commit is not None and capability.commit != str(commit):
            raise Forbidden("this token is pinned to a different commit")
        if not self._reachability.reaches(env, commit):
            raise Forbidden(
                "that commit is not part of this environment",
                env_id=str(env),
            )
