"""The authorization model: principals, operations, selectors, scopes.

Access is provisioned along two axes — which operations a caller may
perform, and which environments it may perform them on — and one construct
covers both::

    scope  =  { operations }  ×  { environments }

*Read but not write* and *fork but not trigger a build* narrow the operations;
*these environments and not those* narrows the selector. There is one mechanism,
not two.

The stored form of this is a *grant*, and there is no ``Grant``
type here because there is no grant store: a token is minted by narrowing the
authority of the token that asked for it, never by resolving a principal's
grants. That is a real gap rather than a simplification — the
missing piece is the keyspace, and ``Scope`` is already the shape it would
resolve to.

Three decisions here are worth stating because each removes a whole class of
mistake:

**Operations are a bitmask, frozen forever.** A token carries its scope rather
than naming a principal whose grants must then be fetched, so the scope has to
be compact — at 14,000 reads per second a policy lookup per request would be the
most expensive thing in the system.

**No operation implies another.** ``env:admin`` does not grant ``env:read``;
``env:write`` does not grant ``env:read``. Effective
permission as the union of explicitly granted operations, and implication would
silently widen every grant while making "read but not write" harder to reason
about — the exact phrase the requirement uses.

**Selectors are additive with no deny rules.** Effective permission is the union
of what matched, so "who can write to this environment" is answered by looking at
what selected it, and no rule can be shadowed by another.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING, Self, final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.ids import EnvId, EnvName

__all__ = [
    "LabelSelector",
    "NamePrefixSelector",
    "Operation",
    "Principal",
    "Scope",
    "Selector",
    "SelectorSet",
    "parse_operations",
]


@final
class Operation(IntFlag):
    """What a caller may do. These bit values are frozen — a token minted today
    must still verify tomorrow.
    """

    READ = 1 << 0
    WRITE = 1 << 1
    ANNOTATE = 1 << 2
    FORK = 1 << 3
    CREATE = 1 << 4
    BUILD = 1 << 5
    ADMIN = 1 << 6

    @property
    def label(self) -> str:
        return f"env:{self.name.lower()}" if self.name else str(int(self))


#: Rendered names, in their canonical spelling.
_BY_LABEL = {op.label: op for op in Operation}


def parse_operations(labels: Sequence[str]) -> Operation:
    """``["env:read", "env:fork"]`` → a bitmask. Unknown labels are an error.

    Refusing an unknown label rather than ignoring it matters: a typo in a grant
    that silently granted nothing would look like a permissions bug for as long
    as it took someone to notice.
    """
    result = Operation(0)
    for label in labels:
        operation = _BY_LABEL.get(label)
        if operation is None:
            raise ValueError(f"unknown operation: {label!r} (known: {sorted(_BY_LABEL)})")
        result |= operation
    return result


def render_operations(operations: Operation) -> list[str]:
    return [op.label for op in Operation if operations & op]


@final
@dataclass(frozen=True, slots=True, order=True)
class Principal:
    """Who is acting.

    Humans through SSO, long-lived automations, and per-job identities. A rollout
    gets its own principal — one environment, read-only, expiring with the
    rollout — because the agent code running inside it is untrusted with respect
    to the corpus.
    """

    value: str

    def __str__(self) -> str:
        return self.value


# ─────────────────────────────────────────────────────────────────────────────
# Selectors
# ─────────────────────────────────────────────────────────────────────────────


class Selector:
    """Which environments a grant covers. A closed union.

    Closed rather than extensible: a selector has to be evaluated inside a token
    verification that must not touch a store, so anything requiring a lookup
    cannot be admitted here.
    """

    __slots__ = ()

    def matches(self, env_id: EnvId, name: EnvName, labels: Mapping[str, str]) -> bool:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


@final
@dataclass(frozen=True, slots=True)
class ExactSelector(Selector):
    env_id: str

    def matches(self, env_id: EnvId, name: EnvName, labels: Mapping[str, str]) -> bool:
        del name, labels
        return str(env_id) == self.env_id

    def render(self) -> str:
        return f"id:{self.env_id}"


@final
@dataclass(frozen=True, slots=True)
class NamePrefixSelector:
    """``proximal/swe-*``. Mirrors GitHub's owner/repo shape, which is what gives
    grants a natural prefix to be scoped against.
    """

    pattern: str

    def matches(self, env_id: EnvId, name: EnvName, labels: Mapping[str, str]) -> bool:
        del env_id, labels
        return self.matches_name(name)

    def matches_name(self, name: EnvName) -> bool:
        """The same decision, for a name that may not name anything yet.

        Split out because ``Scope.could_name`` needs it before an environment
        exists — see there for why that question has to be answerable at all.
        """
        return fnmatch.fnmatchcase(str(name), self.pattern)

    def render(self) -> str:
        return f"name:{self.pattern}"


@final
@dataclass(frozen=True, slots=True)
class LabelSelector:
    """``family=swe-bench AND team=rl-core``. Every pair must match."""

    required: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, **pairs: str) -> Self:
        return cls(tuple(sorted(pairs.items())))

    def matches(self, env_id: EnvId, name: EnvName, labels: Mapping[str, str]) -> bool:
        del env_id, name
        return all(labels.get(key) == value for key, value in self.required)

    def render(self) -> str:
        return "labels:" + ",".join(f"{k}={v}" for k, v in self.required)


type SelectorSet = tuple[ExactSelector | NamePrefixSelector | LabelSelector, ...]


def parse_selector(rendered: str) -> ExactSelector | NamePrefixSelector | LabelSelector:
    kind, _, rest = rendered.partition(":")
    match kind:
        case "id":
            return ExactSelector(rest)
        case "name":
            return NamePrefixSelector(rest)
        case "labels":
            pairs: list[tuple[str, str]] = []
            for pair in rest.split(","):
                if not pair:
                    continue
                key, _, value = pair.partition("=")
                pairs.append((key, value))
            return LabelSelector(tuple(sorted(pairs)))
        case _:
            raise ValueError(f"unknown selector: {rendered!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Scopes
# ─────────────────────────────────────────────────────────────────────────────


@final
@dataclass(frozen=True, slots=True)
class Scope:
    """A token's resolved authority.

    Grants are evaluated at *mint* time and the result is carried in the token,
    so verification is a signature check rather than a store read. The cost is
    that changing a grant changes what the *next* token can do — which is why
    tokens are short-lived.
    """

    operations: Operation
    selectors: SelectorSet
    #: When present, the token may only touch this environment. Per-rollout
    #: tokens pin it so a compromised agent cannot reach the corpus.
    env_id: str | None = None

    def permits(
        self,
        operation: Operation,
        env_id: EnvId,
        name: EnvName,
        labels: Mapping[str, str],
    ) -> bool:
        if self.env_id is not None and self.env_id != str(env_id):
            return False
        if not self.operations & operation:
            return False
        return any(selector.matches(env_id, name, labels) for selector in self.selectors)

    def could_name(self, name: EnvName) -> bool:
        """Whether this token has authority over the *namespace* ``name`` sits in.

        Answerable without the environment existing, which is the point: it
        decides what a caller is told about a name that resolves to nothing.
        Telling everyone "no such environment" while telling them "forbidden"
        for one that does exist is an oracle — a stranger enumerates the corpus
        one name at a time by reading status codes. So a caller who could not
        have touched this name either way is told the same thing in both cases,
        and only a caller already holding the namespace gets the honest 404.

        Only ``NamePrefixSelector`` can be evaluated here; the other two need an
        environment to match against, so a token carrying only those learns
        nothing. That asymmetry is deliberate rather than a limitation — a
        name-scoped grant *is* authority over names that do not exist yet, which
        is exactly what makes revealing their absence harmless.

        An environment pin ends the question before it starts. A pinned token may
        touch exactly one environment, and that environment exists — so every
        name it could ask about and *not* find is a name it has no authority
        over. Reading only the selectors here would leave the oracle open in its
        sharpest form: a per-rollout token is minted from a broad one and keeps
        its ``proximal/*`` selector, so it would learn the presence or absence of
        every name in the namespace while being permitted to read none of them.
        Those tokens are handed to agent code that is treated as untrusted, which makes this
        the one caller that most needs to learn nothing.
        """
        if self.env_id is not None:
            return False
        return any(
            isinstance(selector, NamePrefixSelector) and selector.matches_name(name)
            for selector in self.selectors
        )

    def narrow(self, *, operations: Operation | None = None, env_id: str | None = None) -> Scope:
        """Produce a strictly narrower scope.

        Only ever narrows: the intersection of operations, and an environment
        pin that cannot be removed once set. Widening is not expressible, which
        is what makes an attenuated token safe to hand to a rollout.
        """
        return Scope(
            operations=self.operations & operations if operations is not None else self.operations,
            selectors=self.selectors,
            env_id=env_id if self.env_id is None else self.env_id,
        )
