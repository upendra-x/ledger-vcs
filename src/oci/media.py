"""OCI media types, and a compression trade made deliberately.

**Ledger stores layers uncompressed.** A layer's OCI digest is the SHA-256 of the
*gzipped* tar, and gzip is the enemy of content-defined chunking: a one-byte
change near the start of a layer perturbs every compressed byte after it, so two
builds of a nearly identical layer would share almost no chunks. Storing the
compressed bytes preserves the legacy digest and destroys the deduplication that
motivates the entire design.

So the tar is decompressed on ingest and stored as ordinary content, where CDC
works normally. What that costs is stated exactly:

* the byte-identical gzip stream cannot be handed back — only the layer
  *content* is restored byte for byte, not its compression envelope;
* the layer descriptor's digest changes, and therefore so does the manifest's.

What it does **not** cost is the image's *configuration* identity. A layer's
uncompressed digest is precisely its ``diff_id``, which the config blob already
lists — so decompressing changes no value inside the config, the config is stored
byte for byte, and its digest is the same one the source build produced. The
rootfs a runtime unpacks is identical too, layer for layer, because the tar
inside the gzip is what was always going to be extracted.

So: same config digest, same layer content, different manifest digest. A client
that pins the manifest digest must re-pin; a client that cares what the image
*is* gets exactly what it had.
"""

from __future__ import annotations

import enum
import gzip
from typing import TYPE_CHECKING, Final, final

from src.errors import InvalidRequest

if TYPE_CHECKING:
    from typing import IO

    from src.format.cdc import ByteReader

__all__ = [
    "MEDIA_IMAGE_CONFIG",
    "MEDIA_IMAGE_INDEX",
    "MEDIA_IMAGE_MANIFEST",
    "MEDIA_LAYER_TAR",
    "Compression",
    "compression_of",
    "decompressed",
    "is_index",
    "is_manifest",
    "uncompressed_layer_type",
]

# ── OCI ──────────────────────────────────────────────────────────────────────
MEDIA_IMAGE_INDEX: Final = "application/vnd.oci.image.index.v1+json"
MEDIA_IMAGE_MANIFEST: Final = "application/vnd.oci.image.manifest.v1+json"
MEDIA_IMAGE_CONFIG: Final = "application/vnd.oci.image.config.v1+json"
MEDIA_LAYER_TAR: Final = "application/vnd.oci.image.layer.v1.tar"
MEDIA_LAYER_TAR_GZIP: Final = "application/vnd.oci.image.layer.v1.tar+gzip"
MEDIA_LAYER_TAR_ZSTD: Final = "application/vnd.oci.image.layer.v1.tar+zstd"
MEDIA_EMPTY: Final = "application/vnd.oci.empty.v1+json"

# ── Docker, which registries and daemons still emit ──────────────────────────
MEDIA_DOCKER_MANIFEST_LIST: Final = "application/vnd.docker.distribution.manifest.list.v2+json"
MEDIA_DOCKER_MANIFEST: Final = "application/vnd.docker.distribution.manifest.v2+json"
MEDIA_DOCKER_CONFIG: Final = "application/vnd.docker.container.image.v1+json"
MEDIA_DOCKER_LAYER: Final = "application/vnd.docker.image.rootfs.diff.tar"
MEDIA_DOCKER_LAYER_GZIP: Final = "application/vnd.docker.image.rootfs.diff.tar.gzip"

#: Layers whose bytes live somewhere else and are fetched by URL. Ledger cannot
#: store what it was never given, and silently dropping such a layer would
#: produce an image that pulls and then fails to run.
FOREIGN_LAYER_TYPES: Final = frozenset(
    {
        "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip",
        "application/vnd.oci.image.layer.nondistributable.v1.tar",
        "application/vnd.oci.image.layer.nondistributable.v1.tar+gzip",
        "application/vnd.oci.image.layer.nondistributable.v1.tar+zstd",
    }
)

_INDEX_TYPES: Final = frozenset({MEDIA_IMAGE_INDEX, MEDIA_DOCKER_MANIFEST_LIST})
_MANIFEST_TYPES: Final = frozenset({MEDIA_IMAGE_MANIFEST, MEDIA_DOCKER_MANIFEST})


def is_index(media_type: str) -> bool:
    return media_type in _INDEX_TYPES


def is_manifest(media_type: str) -> bool:
    return media_type in _MANIFEST_TYPES


@final
class Compression(enum.Enum):
    """How a layer's bytes are wrapped in the source we were handed."""

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"


def compression_of(media_type: str) -> Compression:
    """Read the compression out of a layer's media type.

    By suffix rather than by sniffing magic bytes: the media type is what the
    manifest *claims*, and a layer whose bytes disagree with its declared type is
    a corrupt image we want to fail on rather than quietly reinterpret.
    """
    if media_type in FOREIGN_LAYER_TYPES:
        raise InvalidRequest(
            "this image has a foreign (non-distributable) layer, whose bytes were "
            "never included; Ledger stores content and cannot version a pointer",
            media_type=media_type,
        )
    if media_type.endswith(("+gzip", ".tar.gzip")):
        return Compression.GZIP
    if media_type.endswith("+zstd"):
        return Compression.ZSTD
    if media_type.endswith((".tar", "+tar")) or media_type in {
        MEDIA_LAYER_TAR,
        MEDIA_DOCKER_LAYER,
    }:
        return Compression.NONE
    raise InvalidRequest(f"unrecognised layer media type {media_type!r}", media_type=media_type)


def uncompressed_layer_type(media_type: str) -> str:
    """The media type the same layer gets once Ledger has unwrapped it.

    Always an OCI type, even for a Docker-typed source layer: what we store is a
    plain tar, and describing it with Docker's vocabulary would claim a
    compression that is no longer there.
    """
    compression_of(media_type)  # rejects foreign and unknown types
    return MEDIA_LAYER_TAR


def decompressed(raw: IO[bytes], compression: Compression) -> ByteReader:
    """Wrap a layer stream so reads yield the uncompressed tar.

    Streaming, never buffered: a 6 GiB layer is decompressed and chunked in one
    pass, so ingesting a multi-gigabyte image costs a constant amount of memory.

    Returns the narrowest thing the chunker needs — something with ``read`` —
    rather than a file object, because that is genuinely all a decompressor has
    in common with a socket and with an open file.
    """
    match compression:
        case Compression.NONE:
            return raw
        case Compression.GZIP:
            return gzip.GzipFile(fileobj=raw, mode="rb")
        case Compression.ZSTD:
            from compression.zstd import ZstdFile

            return ZstdFile(raw, "rb")
