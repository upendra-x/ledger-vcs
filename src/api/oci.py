"""The OCI registry endpoint.

Container images are ordinary content (``oci.layout``), but nothing that
consumes them speaks Ledger's API: a container runtime speaks the OCI
distribution protocol. So Ledger speaks it too, read-only, and this is how images
actually reach a rollout host::

    GET /v2/<org>/<env>/<image>/manifests/<ref>    resolves through the commit's
                                                  images/index.json
    GET /v2/<org>/<env>/<image>/blobs/<digest>     digest → path → chunks → bytes

Three properties fall out of the object model rather than being built here.

**A pull is authorized exactly like a read.** The token that grants ``env:read``
grants the pull, so there is no registry credential anywhere. A blob is found by
resolving ``images/blobs/sha256/<hex>`` *within a commit of this environment* —
never by looking the SHA-256 up in a global index, which would hand any caller
any layer in the corpus by hash alone.

**Pulling a tag yields that version's images.** A registry tag is a Ledger ref, so
``…/app:main`` resolves through ``refs/heads/main`` to a commit, and the commit
pins the manifest digest. Move the ref back and the same pull yields the old
image — there is no tag that could have moved underneath it.

**Layers are the same objects everything else reads.** A base layer shared by ten
thousand environments is one object and one cache entry, which is what an
ordinary registry works hard for.

The one place the OCI mapping is not lossless is compression, and ``oci.media``
states that trade exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final

from fastapi import APIRouter, Depends, Header, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.deps import Authorized, LedgerDep, authorize_env, interactive_capability
from src.auth.model import Operation
from src.auth.tokens import Capability
from src.errors import InvalidRequest, LedgerError, NotFound
from src.format.model import Commit, TreeEntry
from src.fs.blob import BlobReader
from src.ids import EnvName, ObjectName, RefName
from src.meta.models import Ref
from src.oci.digest import Digest
from src.oci.layout import find_blob, find_manifest
from src.oci.media import MEDIA_LAYER_TAR
from src.oci.model import Descriptor

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["REGISTRY_PREFIX", "oci_error_response", "router"]

REGISTRY_PREFIX: Final = "/v2"

#: What an unauthenticated pull is told to do. Basic rather than a token dance
#: because the credential *is* a Ledger token — ``docker login <host> -u <who>
#: -p <token>`` — and inventing a second token endpoint to hand back a token the
#: caller already has would be ceremony with no security property attached.
WWW_AUTHENTICATE: Final = 'Basic realm="ledger",service="ledger"'

#: The header every distribution client reads to learn a manifest's digest
#: without hashing the body itself.
CONTENT_DIGEST: Final = "Docker-Content-Digest"

router = APIRouter(prefix=REGISTRY_PREFIX, tags=["registry"])


# ─────────────────────────────────────────────────────────────────────────────
# Authorization
# ─────────────────────────────────────────────────────────────────────────────


#: The registry's credential handling is the API's — see ``deps``. A container
#: runtime sends HTTP Basic with the Ledger token as the password, because that
#: is the only shape ``docker login`` can be told to produce, and it ends at the
#: same verifier as a bearer token.
registry_capability = interactive_capability


def require_pull() -> object:
    """Authorize a pull as a read of the environment in the repository name.

    Through ``authorize_env`` rather than resolving and checking here, so the
    registry cannot answer a stranger differently from the API — a pull against
    an unknown repository and a pull against one the caller may not read have to
    look the same, or the registry becomes the enumeration oracle the API is not.
    """

    def dependency(
        org: str,
        env: str,
        state: LedgerDep,
        capability: Annotated[Capability, Depends(registry_capability)],
    ) -> Authorized:
        return authorize_env(state, capability, EnvName(f"{org}/{env}"), Operation.READ)

    return Depends(dependency)


PullAuth = Annotated[Authorized, require_pull()]


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────


def _tree_for_reference(auth: Authorized, reference: str) -> ObjectName:
    """The root tree a tag refers to.

    A registry tag is a Ledger ref: bare names resolve under ``refs/heads/``, and
    a fully qualified ref is accepted as written so a rollout can pin
    ``refs/tags/v4`` if it wants to.
    """
    ref = RefName(reference if reference.startswith("refs/") else f"refs/heads/{reference}")
    try:
        commit = auth.ledger.repo.get_ref(auth.env_id, ref).target
    except NotFound as exc:
        raise NotFound(
            "no such tag in this environment",
            oci_code="MANIFEST_UNKNOWN",
            reference=reference,
        ) from exc
    return auth.ledger.store.get_as(commit, Commit).tree


def _current_trees(auth: Authorized) -> Iterator[tuple[Ref, ObjectName]]:
    """Every ref and the root tree of the commit it points at, now.

    This is what a request that names no tag — a blob fetch, or a manifest fetch
    by digest — is resolved against. Bounding it to what refs point at *now* is
    deliberate and is the same rule collection uses: content no ref names is
    content the registry stops serving, so the endpoint can never hand back an
    image that garbage collection is entitled to reclaim.

    A ref naming a commit this store does not hold is skipped rather than raised.
    That is a restore-from-backup or half-replicated state, and one unreadable
    ref must not take down every pull from the environment.
    """
    store = auth.ledger.store
    for ref in auth.ledger.repo.list_refs(auth.env_id):
        try:
            yield ref, store.get_as(ref.target, Commit).tree
        except LedgerError:  # pragma: no cover - a ref naming a missing commit
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/", include_in_schema=False)
@router.get("")
def version_check(state: LedgerDep) -> Response:
    """The handshake every client makes before anything else.

    Deliberately unauthenticated. It advertises that this *is* a registry and
    nothing more; a 401 here would only teach the client to retry with a
    credential it will be asked for on the very next request anyway.
    """
    del state
    return JSONResponse(
        content={},
        headers={"Docker-Distribution-Api-Version": "registry/2.0"},
    )


@router.get("/{org}/{env}/{image}/tags/list")
def list_tags(org: str, env: str, image: str, auth: PullAuth) -> Response:
    """Which tags this image can be pulled at — that is, which refs exist.

    Only refs whose current version actually holds the image are listed, because
    a tag that resolves to a version without it would 404 on the very next
    request.

    Resolution goes through ``_current_trees`` rather than reading each ref here,
    so a ref naming a commit this store does not hold is skipped in both places.
    Two spellings of the same walk had drifted: this one raised, turning one bad
    ref into a 500 for the whole listing, while the other quietly moved on.
    """
    tags: list[str] = []
    for ref, tree in _current_trees(auth):
        if find_manifest(auth.ledger.store, tree, image) is not None:
            tags.append(str(ref.name).removeprefix("refs/heads/"))
    return JSONResponse(content={"name": _repository(org, env, image), "tags": sorted(tags)})


@router.get("/{org}/{env}/{image}/manifests/{reference}")
@router.head("/{org}/{env}/{image}/manifests/{reference}")
def get_manifest(
    org: str, env: str, image: str, reference: str, request: Request, auth: PullAuth
) -> Response:
    """Serve an image manifest, by tag or by digest.

    The bytes are the ones stored, never re-serialized: a manifest's digest is
    the hash of its bytes, and every client re-hashes what it received.
    """
    del org, env
    found = _lookup_manifest(auth, image, reference)
    if found is None:
        raise NotFound(
            "no such manifest in this environment",
            oci_code="MANIFEST_UNKNOWN",
            image=image,
            reference=reference,
        )
    descriptor, body = found
    headers = {
        CONTENT_DIGEST: str(descriptor.digest),
        "Content-Length": str(len(body)),
        "Docker-Distribution-Api-Version": "registry/2.0",
    }
    # A HEAD carries the same headers and no body — which is exactly how a
    # client checks whether it already has an image without downloading it.
    if request.method == "HEAD":
        return Response(
            status_code=status.HTTP_200_OK, headers=headers, media_type=descriptor.media_type
        )
    return Response(content=body, media_type=descriptor.media_type, headers=headers)


def _lookup_manifest(
    auth: Authorized, image: str, reference: str
) -> tuple[Descriptor, bytes] | None:
    store = auth.ledger.store
    if reference.startswith("sha256:"):
        wanted = Digest.parse(reference)
        for _ref, tree in _current_trees(auth):
            found = find_manifest(store, tree, image)
            if found is not None and found[0].digest == wanted:
                return found
        return None
    return find_manifest(store, _tree_for_reference(auth, reference), image)


@router.get("/{org}/{env}/{image}/blobs/{digest}")
@router.head("/{org}/{env}/{image}/blobs/{digest}")
def get_blob(
    org: str,
    env: str,
    image: str,
    digest: str,
    request: Request,
    auth: PullAuth,
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> Response:
    """Serve a layer or config blob.

    Found by *path* within a version of this environment, which is what makes a
    pull authorized exactly like a read. Streamed rather than buffered, so a
    2 GiB layer costs the service a few megabytes of memory and reaches the
    client while it is still being assembled.
    """
    del org, env, image
    wanted = Digest.parse(digest)
    entry = _find_blob_entry(auth, wanted)
    if entry is None:
        raise NotFound("no such blob in this environment", oci_code="BLOB_UNKNOWN", digest=digest)

    headers = {CONTENT_DIGEST: str(wanted), "Accept-Ranges": "bytes"}
    if request.method == "HEAD":
        return Response(
            status_code=status.HTTP_200_OK,
            headers={**headers, "Content-Length": str(entry.size)},
            media_type=MEDIA_LAYER_TAR,
        )

    reader = BlobReader(auth.ledger.store, entry.target)
    if range_header is None:
        return StreamingResponse(
            reader.stream(),
            media_type=MEDIA_LAYER_TAR,
            headers={**headers, "Content-Length": str(entry.size)},
        )

    start, end = _parse_range(range_header, entry.size)
    return StreamingResponse(
        reader.stream(start, end - start + 1),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=MEDIA_LAYER_TAR,
        headers={
            **headers,
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{entry.size}",
        },
    )


def _find_blob_entry(auth: Authorized, digest: Digest) -> TreeEntry | None:
    for _ref, tree in _current_trees(auth):
        entry = find_blob(auth.ledger.store, tree, digest)
        if entry is not None:
            return entry
    return None


def _parse_range(header: str, size: int) -> tuple[int, int]:
    """One byte range, the only form a distribution client sends.

    Both spellings are handled, and the difference between them matters:
    ``bytes=100-`` is *from* 100, while ``bytes=-100`` is the **last** hundred
    bytes. Reading the second as the first is the worst kind of bug this route
    can have — it answers 206 with entirely the wrong bytes, and the client has
    no way to tell.

    Multi-range requests are refused rather than partially honoured, for the same
    reason: answering only the first range looks like success.
    """
    units, separator, spec = header.partition("=")
    if not separator or units.strip().lower() != "bytes" or "," in spec:
        raise InvalidRequest("only a single byte range is supported", range=header)

    first, dash, last = spec.strip().partition("-")
    if not dash:
        raise InvalidRequest("malformed Range header", range=header)
    try:
        if not first:
            # A suffix range. Clamped rather than refused when it asks for more
            # than exists, which is what the HTTP specification requires.
            suffix = int(last)
            if suffix <= 0:
                raise InvalidRequest("a suffix range must be positive", range=header)
            return max(0, size - suffix), size - 1
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError as exc:
        raise InvalidRequest("malformed Range header", range=header) from exc

    if not 0 <= start <= end < size:
        raise InvalidRequest("range is outside the blob", range=header, size=size)
    return start, end


def _repository(org: str, env: str, image: str) -> str:
    return f"{org}/{env}/{image}"


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

#: How a Ledger error is spelled in the distribution specification's vocabulary,
#: when the raising site did not say. Kept as a fallback rather than the primary
#: mapping so the specific code — MANIFEST_UNKNOWN versus BLOB_UNKNOWN, which a
#: client genuinely branches on — comes from the site that knows which it is.
_FALLBACK_CODES: Final = {
    400: "UNSUPPORTED",
    401: "UNAUTHORIZED",
    403: "DENIED",
    404: "NAME_UNKNOWN",
    429: "TOOMANYREQUESTS",
}


def oci_error_response(exc: LedgerError) -> JSONResponse:
    """Render a Ledger error the way a distribution client expects to read it.

    The registry has its own error contract — an ``errors`` array of coded
    objects — and a client that gets Ledger's shape instead reports "unknown
    error" with no detail. One translation, in one place, so no route can forget.
    """
    code = exc.details.get("oci_code") or _FALLBACK_CODES.get(exc.status_code, "UNKNOWN")
    details = {k: v for k, v in exc.details.items() if k != "oci_code"}
    headers = {"WWW-Authenticate": WWW_AUTHENTICATE} if exc.status_code == 401 else {}
    return JSONResponse(
        status_code=exc.status_code,
        content={"errors": [{"code": code, "message": exc.message, "detail": details}]},
        headers=headers,
    )
