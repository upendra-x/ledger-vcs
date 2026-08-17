"""Dependency wiring, and the authorization gate.

Two properties are load-bearing here.

**Authorization runs before any resolution work.** An unauthorized
request should cost a signature verification and never reach the stores. So the
gate takes an environment *name*, resolves it, and checks the token — and the
403 it raises carries no information about whether the environment exists, since
a 403 that leaked existence would be an enumeration oracle over the corpus.

**Blocking work never runs on the event loop.** Every store call is synchronous
SQLite or filesystem I/O, and BLAKE3 hashing is CPU-bound. Routes are declared
``def`` rather than ``async def``, which makes Starlette run them in its
threadpool — the simplest correct answer, and the one that cannot be undone by a
future author forgetting to wrap a call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, final

from fastapi import Depends, Header, Request

from src.auth.model import Operation, Principal

# Imported at runtime, not under TYPE_CHECKING: FastAPI resolves dependency
# annotations when it builds the dependency graph, and an unresolvable
# forward reference is silently downgraded to a query parameter rather than
# raising — which turns an authorization dependency into a required query arg.
from src.auth.tokens import Capability, TicketSigner, TokenSigner, TokenVerifier
from src.errors import Unauthenticated

if TYPE_CHECKING:
    from src.auth.policy import PolicyEngine
    from src.ids import EnvId, EnvName
    from src.instance import Ledger
    from src.service.commits import CommitService

__all__ = [
    "AppState",
    "Authorized",
    "CurrentCapability",
    "LedgerDep",
    "authorize_env",
    "dev_capability",
    "require",
    "state_of",
]


@final
@dataclass(frozen=True, slots=True)
class AppState:
    """Everything a request handler may need, assembled once at startup."""

    ledger: Ledger
    policy: PolicyEngine
    commits: CommitService
    verifier: TokenVerifier
    signer: TokenSigner
    tickets: TicketSigner
    #: When true, requests without a token act as a fully-privileged local
    #: operator. Off by default and logged loudly at startup — a convenience for
    #: a single-machine demo, never a deployment mode.
    dev_mode: bool = False


def state_of(request: Request) -> AppState:
    return request.app.state.ledger_state  # type: ignore[no-any-return]


LedgerDep = Annotated[AppState, Depends(state_of)]


def dev_capability() -> Capability:
    """What an unauthenticated request gets when ``dev_mode`` is on.

    Defined once. Every surface that honours dev mode — the API and the registry
    — reads it from here, because two copies of "what dev mode grants" is two
    places for one of them to quietly grow more authority than the other.
    """
    from src.auth.model import NamePrefixSelector, Scope
    from src.auth.model import Operation as Op

    return Capability(
        principal=Principal("dev"),
        scope=Scope(operations=Op(~0) & ~Op(0), selectors=(NamePrefixSelector("*"),)),
        expires_at_us=2**62,
        not_before_us=0,
        token_id="dev",
    )


def current_capability(
    state: LedgerDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Capability:
    """Verify the bearer token. A signature check — no store, no network."""
    if authorization is None:
        if state.dev_mode:
            return dev_capability()
        raise Unauthenticated("a bearer token is required")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Unauthenticated("expected an Authorization: Bearer <token> header")
    return state.verifier.verify(token)


CurrentCapability = Annotated[Capability, Depends(current_capability)]


def interactive_capability(
    state: LedgerDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Capability:
    """Verify a credential from a client that cannot set arbitrary headers.

    A container runtime and a web browser have the same problem: neither can be
    told "send this bearer token". Both can do HTTP Basic — ``docker login``, and
    a browser's own prompt in response to ``WWW-Authenticate`` — so both surfaces
    accept the Ledger token as the *password*.

    It ends at the same verifier as everything else. Neither the registry nor the
    UI has its own idea of who anyone is, which is what is meant by a
    pull being authorized exactly like a read.
    """
    if authorization is None:
        if state.dev_mode:
            return dev_capability()
        raise Unauthenticated("a Ledger token is required")

    scheme, _, credential = authorization.partition(" ")
    match scheme.lower():
        case "bearer":
            token = credential
        case "basic":
            token = _password_from_basic(credential)
        case _:
            raise Unauthenticated("expected Bearer or Basic authorization")
    if not token:
        raise Unauthenticated("no credential supplied")
    return state.verifier.verify(token)


def _password_from_basic(credential: str) -> str:
    import base64

    try:
        decoded = base64.b64decode(credential, validate=True).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise Unauthenticated("malformed Basic credential") from exc
    # Usernames cannot contain a colon, so everything after the first one is the
    # password — which is where the token goes.
    _, _, password = decoded.partition(":")
    return password


@final
class Authorized:
    """A request that has already passed the gate.

    Carries the resolved environment id so a handler cannot accidentally act on
    a different one than was authorized.
    """

    __slots__ = ("capability", "env_id", "state")

    def __init__(self, state: AppState, capability: Capability, env_id: EnvId) -> None:
        self.state = state
        self.capability = capability
        self.env_id = env_id

    @property
    def principal(self) -> Principal:
        return self.capability.principal

    @property
    def ledger(self) -> Ledger:
        return self.state.ledger


def authorize_env(
    state: AppState, capability: Capability, name: EnvName, operation: Operation
) -> Authorized:
    """Resolve an environment name and authorize an operation on it.

    **The single gate.** Every surface goes through here — the API, the
    registry, the browser — because the interesting part is not the permitted
    case but what the refused ones are told, and two copies of that reasoning is
    one copy waiting to disagree.

    Resolution can fail, and *how* it fails is the whole problem. Answering 404
    for a name that resolves to nothing while answering 403 for one that does
    hands a stranger an oracle: try a name, read the status code, learn whether
    it exists. Repeat, and the corpus's environment names are enumerated by
    someone holding no authority over any of them.

    So absence is reported only to a caller whose token already covers the
    namespace; to everyone else it is spelled exactly like a refusal. The rule
    asks for a 403 that carries nothing, and a 403 carries nothing only if the
    404 beside it is equally uninformative.
    """
    from src.errors import NotFound

    try:
        env_id = state.ledger.repo.resolve_env_name(name)
    except NotFound:
        if capability.scope.could_name(name):
            raise
        raise _refused(operation) from None
    state.policy.authorize(capability, operation, env_id)
    return Authorized(state, capability, env_id)


def _refused(operation: Operation) -> Exception:
    """The refusal, worded identically wherever it comes from.

    Deliberately the same message and details the policy engine raises when an
    environment does exist. Two spellings of "no" are two thirds of an oracle.
    """
    from src.errors import Forbidden

    return Forbidden(
        "this token does not permit that operation on that environment",
        operation=operation.label,
    )


def require(operation: Operation) -> object:
    """Build a dependency that authorizes ``operation`` on the path's environment.

    Used as ``auth: Annotated[Authorized, Depends(require(Operation.READ))]`` so
    that every route states its own requirement, and
    ``test_every_route_declares_authorization`` can check none was forgotten.
    """

    def dependency(
        org: str,
        env: str,
        state: LedgerDep,
        capability: CurrentCapability,
    ) -> Authorized:
        from src.ids import EnvName

        # An environment name is org/environment, which is two
        # path segments — the same shape as GitHub's /repos/{owner}/{repo}, so
        # the migration is one-to-one.
        return authorize_env(state, capability, EnvName(f"{org}/{env}"), operation)

    return Depends(dependency)


def require_global(operation: Operation) -> object:
    """Authorize an operation that has no environment to select against.

    The operation set alone, because there is nothing to match a selector to.
    That makes this a **necessary and not a sufficient** check for any route
    whose request names something: ``create_env`` also has to ask
    ``Scope.could_name`` about the name in the body, or a namespace-scoped token
    would be able to claim a name anywhere in the corpus.
    """

    def dependency(state: LedgerDep, capability: CurrentCapability) -> Capability:
        from src.errors import Forbidden

        del state
        if not capability.scope.operations & operation:
            raise Forbidden("this token does not permit that operation", operation=operation.label)
        return capability

    return Depends(dependency)
