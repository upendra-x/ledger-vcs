"""Capability tokens, and the content tickets that carry the hot path.

**Two mechanisms, because the two paths have different budgets.**

*API routes* — a few thousand per second — carry an Ed25519 capability token.
It states its own authority, so verification is a signature check with no store
read: at 14,000 reads per second a policy lookup per request would be the single
most expensive thing in the system.

*Content bytes* — the bulk of the traffic — carry an HMAC ticket instead.
Ed25519 verification is roughly 50 µs; HMAC-SHA256 over a short string is under
a microsecond. The API plane hands out short-lived signed URLs
and the edge serve the bytes, and the ticket is that signature.

**No algorithm field.** JOSE's ``alg`` header is the most-exploited part of the
format, because a verifier that trusts it can be told to accept ``none`` or to
treat a public key as an HMAC secret. The version tag ``lg1`` fixes the suite
forever: there is no algorithm agility here, therefore no algorithm confusion.

**Possessing a hash is not authority.** Hashes leak through logs
and diffs, so a bare-hash read would make a hash a credential. A ticket binds
the hash to a principal and an expiry, and is issued only after a path-scoped
authorization has already succeeded.
"""

from __future__ import annotations

import base64
import hmac
import json
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Final, Self, final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from src.auth.model import Principal, Scope, parse_operations, parse_selector
from src.auth.model import render_operations as _render_operations
from src.errors import Unauthenticated

if TYPE_CHECKING:
    from src.clock import Clock
    from src.ids import ObjectName

__all__ = [
    "TOKEN_PREFIX",
    "Capability",
    "ContentTicket",
    "TicketSigner",
    "TokenSigner",
    "TokenVerifier",
]

#: Fixes the signature suite forever. A successor is a new prefix, never a
#: negotiated parameter.
TOKEN_PREFIX: Final = "lg1"

#: Ticket lifetime. Long enough for a large fetch, short enough that a leaked
#: URL in a log is not a standing grant.
DEFAULT_TICKET_TTL_US: Final = 15 * 60 * 1_000_000


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@final
@dataclass(frozen=True, slots=True)
class Capability:
    """A verified token's claims."""

    principal: Principal
    scope: Scope
    expires_at_us: int
    not_before_us: int
    token_id: str
    #: When set, the token may only read this commit. A per-rollout token pins
    #: it, so the agent inside cannot wander the corpus even by hash.
    commit: str | None = None

    def to_claims(self) -> dict[str, Any]:
        claims: dict[str, Any] = {
            "sub": str(self.principal),
            "scp": _render_operations(self.scope.operations),
            "sel": [s.render() for s in self.scope.selectors],
            "exp": self.expires_at_us,
            "nbf": self.not_before_us,
            "jti": self.token_id,
        }
        if self.scope.env_id is not None:
            claims["env"] = self.scope.env_id
        if self.commit is not None:
            claims["cmt"] = self.commit
        return claims

    @classmethod
    def from_claims(cls, claims: Any) -> Self:
        return cls(
            principal=Principal(claims["sub"]),
            scope=Scope(
                operations=parse_operations(claims["scp"]),
                selectors=tuple(parse_selector(s) for s in claims["sel"]),
                env_id=claims.get("env"),
            ),
            expires_at_us=int(claims["exp"]),
            not_before_us=int(claims["nbf"]),
            token_id=claims["jti"],
            commit=claims.get("cmt"),
        )


@final
class TokenSigner:
    """Mints capability tokens. Lives only where grants are resolved."""

    __slots__ = ("_clock", "_key", "_key_id")

    def __init__(self, key: Ed25519PrivateKey, *, key_id: str, clock: Clock) -> None:
        self._key = key
        self._key_id = key_id
        self._clock = clock

    @classmethod
    def generate(cls, *, key_id: str, clock: Clock) -> TokenSigner:
        return cls(Ed25519PrivateKey.generate(), key_id=key_id, clock=clock)

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._key.public_key()

    @property
    def key_id(self) -> str:
        return self._key_id

    def mint(
        self,
        principal: Principal,
        scope: Scope,
        *,
        ttl_us: int,
        commit: ObjectName | None = None,
        token_id: str | None = None,
    ) -> str:
        now = self._clock.now_us()
        capability = Capability(
            principal=principal,
            scope=scope,
            expires_at_us=now + ttl_us,
            not_before_us=now,
            token_id=token_id or _b64(sha256(f"{principal}{now}".encode()).digest()[:12]),
            commit=str(commit) if commit else None,
        )
        payload = json.dumps(
            {"kid": self._key_id, **capability.to_claims()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        body = _b64(payload)
        signature = _b64(self._key.sign(payload))
        return f"{TOKEN_PREFIX}.{body}.{signature}"


@final
class TokenVerifier:
    """Verifies a token by signature alone — no store, no network.

    Holds public keys by id so rotation is a second entry rather than a flag
    day: tokens minted under the old key keep verifying until they expire.
    """

    __slots__ = ("_clock", "_keys")

    def __init__(self, keys: dict[str, Ed25519PublicKey], *, clock: Clock) -> None:
        self._keys = keys
        self._clock = clock

    def verify(self, token: str) -> Capability:
        prefix, _, rest = token.partition(".")
        if prefix != TOKEN_PREFIX:
            raise Unauthenticated("unrecognised token format")

        body, _, signature = rest.partition(".")
        if not body or not signature:
            raise Unauthenticated("malformed token")

        try:
            payload = _unb64(body)
            claims = json.loads(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            raise Unauthenticated("malformed token payload") from exc

        key = self._keys.get(claims.get("kid", ""))
        if key is None:
            raise Unauthenticated("token was signed by an unknown key")

        try:
            key.verify(_unb64(signature), payload)
        except (InvalidSignature, ValueError) as exc:
            raise Unauthenticated("token signature is not valid") from exc

        try:
            capability = Capability.from_claims(claims)
        except (KeyError, ValueError) as exc:
            raise Unauthenticated("token claims are not well formed") from exc

        now = self._clock.now_us()
        if now >= capability.expires_at_us:
            raise Unauthenticated("token has expired")
        if now < capability.not_before_us:
            raise Unauthenticated("token is not yet valid")
        return capability


# ─────────────────────────────────────────────────────────────────────────────
# Content tickets
# ─────────────────────────────────────────────────────────────────────────────


@final
@dataclass(frozen=True, slots=True)
class ContentTicket:
    object_name: str
    principal: str
    expires_at_us: int


@final
class TicketSigner:
    """Signs and verifies content tickets.

    Symmetric on purpose. A ticket is verified by the same service that issued
    it — in production by the CDN or the object store's presigned-URL machinery —
    so there is no third party needing a public key, and HMAC is two orders of
    magnitude cheaper than a signature on a path that carries most of the bytes.
    """

    __slots__ = ("_clock", "_secret")

    def __init__(self, secret: bytes, *, clock: Clock) -> None:
        if len(secret) < 32:
            raise ValueError("ticket secret must be at least 32 bytes")
        self._secret = secret
        self._clock = clock

    def issue(self, name: ObjectName, principal: Principal, *, ttl_us: int | None = None) -> str:
        ticket = ContentTicket(
            object_name=str(name),
            principal=str(principal),
            expires_at_us=self._clock.now_us() + (ttl_us or DEFAULT_TICKET_TTL_US),
        )
        return f"{ticket.expires_at_us}.{_b64(self._mac(ticket))}"

    def verify(self, name: ObjectName, principal: Principal, ticket: str) -> None:
        expiry_text, _, mac = ticket.partition(".")
        if not mac:
            raise Unauthenticated("malformed content ticket")
        try:
            expires_at_us = int(expiry_text)
        except ValueError as exc:
            raise Unauthenticated("malformed content ticket") from exc

        expected = ContentTicket(
            object_name=str(name), principal=str(principal), expires_at_us=expires_at_us
        )
        # Constant time: a ticket check that leaked timing would let an attacker
        # forge one byte at a time.
        if not hmac.compare_digest(_b64(self._mac(expected)), mac):
            raise Unauthenticated("content ticket is not valid for this object")
        if self._clock.now_us() >= expires_at_us:
            raise Unauthenticated("content ticket has expired")

    def _mac(self, ticket: ContentTicket) -> bytes:
        message = f"{ticket.object_name}|{ticket.principal}|{ticket.expires_at_us}".encode()
        return hmac.new(self._secret, message, sha256).digest()
