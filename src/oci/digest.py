"""OCI digests — the *other* content-addressing scheme.

Ledger names objects by BLAKE3; OCI names them by SHA-256, and that is not a
detail we can paper over. Both are content addresses over the same
bytes, so they agree about identity and disagree only about spelling — which
means the mapping between them is a fact about content, never a registry lookup.

Keeping this a value type rather than a string is what makes the two namespaces
impossible to confuse. ``ObjectName`` and ``Digest`` are both "a hash of some
bytes", they are both hex, and passing one where the other belongs would fail
somewhere far away with a puzzling not-found.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self, final

from src.errors import InvalidRequest

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["Digest", "sha256_of"]

#: The only algorithm Ledger serves. OCI permits ``sha512`` as well, but a
#: second algorithm would mean a second blob directory and a second index for no
#: requirement anyone has — so it is refused loudly rather than half-supported.
SHA256: Final = "sha256"

_ENCODED = re.compile(r"^[a-f0-9]{64}$")


@final
@dataclass(frozen=True, slots=True, order=True)
class Digest:
    """``sha256:<64 hex>`` — an OCI content descriptor's identity.

    Ordered so that a set of digests has a canonical listing, which is what lets
    the blob directory of an image layout be built deterministically: the same
    image ingested twice must produce the same tree, or deduplication splits.
    """

    algorithm: str
    encoded: str

    def __post_init__(self) -> None:
        if self.algorithm != SHA256:
            raise InvalidRequest(
                f"unsupported digest algorithm {self.algorithm!r}; Ledger's registry "
                f"serves {SHA256} only",
                algorithm=self.algorithm,
            )
        if not _ENCODED.match(self.encoded):
            raise InvalidRequest(
                "a sha256 digest must be 64 lowercase hex characters",
                encoded=self.encoded[:80],
            )

    @classmethod
    def parse(cls, text: str) -> Self:
        algorithm, separator, encoded = text.partition(":")
        if not separator:
            raise InvalidRequest("a digest must be written <algorithm>:<hex>", digest=text[:80])
        return cls(algorithm=algorithm, encoded=encoded)

    @classmethod
    def of(cls, data: bytes) -> Self:
        return cls(algorithm=SHA256, encoded=hashlib.sha256(data).hexdigest())

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.encoded}"


def sha256_of(blocks: Iterable[bytes]) -> tuple[Digest, int]:
    """Digest and length of a stream, in one pass.

    Used while a layer is being decompressed and chunked: the uncompressed bytes
    flow past exactly once, and hashing them a second time would mean either
    buffering gigabytes or reading them twice.
    """
    hasher = hashlib.sha256()
    total = 0
    for block in blocks:
        hasher.update(block)
        total += len(block)
    return Digest(algorithm=SHA256, encoded=hasher.hexdigest()), total
