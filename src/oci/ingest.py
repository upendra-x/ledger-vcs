"""Turning a container image into ordinary Ledger content.

The whole of it is four steps:

1. **Decompress every layer.** The gzip envelope is discarded and the tar inside
   is chunked like any other large file, because content-defined chunking cannot
   see through gzip (``media`` explains the trade in full).
2. **Leave the config alone.** A layer's uncompressed digest *is* its
   ``diff_id``, which the config already lists — so decompressing changes no
   value inside the config, and its digest is the one the source build produced.
3. **Edit the manifest, don't rebuild it.** Only the layer list and the media
   types change; everything else survives byte for byte.
4. **Write the layout.** Manifest, config and layers become files in the
   commit's ``images/`` directory, named by digest (``layout``).

Two numbers come out of this and both are worth printing. The layers *reused*
count is deduplication measured rather than claimed: a second image sharing a
base layer decompresses nothing, because a layer's ``diff_id`` is known before
any byte of it is read. And the uncompressed-versus-compressed byte counts are
exactly what the trade in step 1 costs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final

from src.errors import InvalidRequest
from src.oci.digest import SHA256, Digest
from src.oci.layout import blob_entry
from src.oci.media import (
    MEDIA_DOCKER_CONFIG,
    MEDIA_IMAGE_CONFIG,
    MEDIA_IMAGE_MANIFEST,
    compression_of,
    decompressed,
    uncompressed_layer_type,
)
from src.oci.model import (
    Descriptor,
    canonical_json,
    parse_document,
)
from src.runtime.ingest import IngestStats
from src.store.digests import DigestEntry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.format.cdc import ByteReader
    from src.format.model import TreeEntry
    from src.ids import ObjectName
    from src.instance import Ledger
    from src.oci.model import Platform
    from src.oci.source import ImageSource, SourceBlob, SourceImage

__all__ = ["ImageIngester", "IngestedImage"]

#: Where a layer's original compression is recorded. The
#: original digest be kept as metadata: it is what lets a caller recognise the
#: layer they pushed, and what a future opt-in "retain the compressed bytes too"
#: would key on.
COMPRESSED_DIGEST: Final = "dev.ledger.oci.compressed.digest"
COMPRESSED_SIZE: Final = "dev.ledger.oci.compressed.size"
COMPRESSED_MEDIA_TYPE: Final = "dev.ledger.oci.compressed.mediaType"


@final
@dataclass(frozen=True, slots=True)
class IngestedImage:
    """One image, now stored as content."""

    image: str
    descriptor: Descriptor
    manifest_bytes: bytes
    blob_entries: tuple[TreeEntry, ...]
    digest_entries: tuple[DigestEntry, ...]
    stats: IngestStats
    layers: int
    layers_reused: int
    bytes_compressed: int
    bytes_uncompressed: int
    platform: Platform | None = None

    @property
    def digest(self) -> Digest:
        return self.descriptor.digest


@final
class ImageIngester:
    """Reads an image out of an archive and writes it into the object store.

    Writes content only — no ref moves, no commit. Publishing is
    ``service.images``' job, which keeps the part that can be retried freely
    separate from the part that is a state transition.
    """

    __slots__ = ("_ledger",)

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def ingest(
        self,
        source: ImageSource,
        *,
        image: str,
        platform: Platform,
        from_image: str | None = None,
    ) -> IngestedImage:
        selected = source.select(from_image, platform)
        config_bytes = source.read(selected.config)
        config = self._store_config(selected, config_bytes)

        diff_ids = _diff_ids(config_bytes)
        if len(diff_ids) != len(selected.layers):
            raise InvalidRequest(
                "this image's config lists a different number of layers than its "
                "manifest; it is internally inconsistent and cannot be trusted",
                config_layers=len(diff_ids),
                manifest_layers=len(selected.layers),
            )

        stats = config.stats
        entries = [config.entry]
        digest_entries: list[DigestEntry] = []
        layers: list[Descriptor] = []
        reused = 0
        compressed = 0
        uncompressed = 0

        for source_layer, diff_id in zip(selected.layers, diff_ids, strict=True):
            outcome = self._store_layer(source, source_layer, diff_id)
            stats = stats + outcome.stats
            entries.append(outcome.entry)
            digest_entries.append(outcome.digest_entry)
            layers.append(outcome.descriptor)
            reused += int(outcome.reused)
            compressed += source_layer.size
            uncompressed += outcome.descriptor.size

        manifest_bytes = _rewrite_manifest(selected, config.descriptor, layers)
        manifest_name, manifest_stats = self._ledger.ingester.ingest_bytes(manifest_bytes)
        manifest_digest = Digest.of(manifest_bytes)
        entries.append(blob_entry(manifest_digest, manifest_name, len(manifest_bytes)))

        return IngestedImage(
            image=image,
            descriptor=Descriptor(
                media_type=MEDIA_IMAGE_MANIFEST,
                digest=manifest_digest,
                size=len(manifest_bytes),
                platform=selected.platform,
            ),
            manifest_bytes=manifest_bytes,
            blob_entries=tuple(entries),
            digest_entries=tuple(digest_entries),
            stats=stats + manifest_stats,
            layers=len(layers),
            layers_reused=reused,
            bytes_compressed=compressed,
            bytes_uncompressed=uncompressed,
            platform=selected.platform,
        )

    # ── config ───────────────────────────────────────────────────────────────

    def _store_config(self, selected: SourceImage, config_bytes: bytes) -> _StoredBlob:
        """Store the config *verbatim*, and check it is what it claimed to be.

        Never re-serialized. The config's digest is the image's identity —
        ``docker images`` prints it as the IMAGE ID — and a re-serialization that
        merely reordered two keys would change it.
        """
        digest = Digest.of(config_bytes)
        if selected.config.digest is not None and selected.config.digest != digest:
            raise InvalidRequest(
                "this image's config does not hash to the digest its manifest "
                "gives; the archive is damaged",
                claimed=str(selected.config.digest),
                actual=str(digest),
            )
        name, stats = self._ledger.ingester.ingest_bytes(config_bytes)
        return _StoredBlob(
            descriptor=Descriptor(
                media_type=_config_media_type(selected.config.media_type),
                digest=digest,
                size=len(config_bytes),
            ),
            entry=blob_entry(digest, name, len(config_bytes)),
            stats=stats,
        )

    # ── layers ───────────────────────────────────────────────────────────────

    def _store_layer(self, source: ImageSource, layer: SourceBlob, diff_id: Digest) -> _StoredLayer:
        reused = self._reuse(diff_id)
        if reused is not None:
            return self._describe_layer(
                layer, diff_id, reused.name, reused.size, IngestStats(), reused=True
            )

        with source.open(layer) as raw:
            reader = _HashingReader(decompressed(raw, compression_of(layer.media_type)))
            name, stats = self._ledger.ingester.ingest_stream(reader)

        # The strongest check available, and it is free: a layer's uncompressed
        # digest must equal the diff_id the config already committed to. If it
        # does not, either the archive is damaged or the layer was not the
        # compression its media type declared — and storing it would produce an
        # image that pulls and then fails to unpack.
        if reader.digest != diff_id:
            raise InvalidRequest(
                "a layer's uncompressed content does not match the diff_id its "
                "config declares; the image is damaged",
                expected=str(diff_id),
                actual=str(reader.digest),
                media_type=layer.media_type,
            )
        return self._describe_layer(layer, diff_id, name, reader.size, stats, reused=False)

    def _reuse(self, diff_id: Digest) -> DigestEntry | None:
        """Have we already stored this exact layer content?

        Keyed on the *uncompressed* digest, which the config hands us before a
        single byte of the layer is read — so a rebuild that only changed the top
        layer decompresses only the top layer, whatever compression either build
        used.

        The index is a hint. A hit is confirmed against the store's own existence
        predicate, because collection can have swept the object since the row was
        written and a stale row must never resurrect it.
        """
        hit = self._ledger.digests.lookup(diff_id.algorithm, diff_id.encoded)
        if hit is None or self._ledger.store.missing([hit.name]):
            return None
        return hit

    def _describe_layer(
        self,
        layer: SourceBlob,
        diff_id: Digest,
        name: ObjectName,
        size: int,
        stats: IngestStats,
        *,
        reused: bool,
    ) -> _StoredLayer:
        annotations: dict[str, str] = {}
        if layer.digest is not None and layer.digest != diff_id:
            annotations = {
                COMPRESSED_DIGEST: str(layer.digest),
                COMPRESSED_SIZE: str(layer.size),
                COMPRESSED_MEDIA_TYPE: layer.media_type,
            }
        return _StoredLayer(
            descriptor=Descriptor(
                media_type=uncompressed_layer_type(layer.media_type),
                digest=diff_id,
                size=size,
                annotations=tuple(sorted(annotations.items())),
            ),
            entry=blob_entry(diff_id, name, size),
            digest_entry=DigestEntry(
                algorithm=SHA256, encoded=diff_id.encoded, name=name, size=size
            ),
            stats=stats,
            reused=reused,
        )


@final
@dataclass(frozen=True, slots=True)
class _StoredBlob:
    descriptor: Descriptor
    entry: TreeEntry
    stats: IngestStats


@final
@dataclass(frozen=True, slots=True)
class _StoredLayer:
    descriptor: Descriptor
    entry: TreeEntry
    digest_entry: DigestEntry
    stats: IngestStats
    reused: bool


@final
class _HashingReader:
    """Digests and counts a stream while something else consumes it.

    The uncompressed bytes of a 6 GiB layer flow past exactly once: chunking
    reads them, this hashes them on the way through. Hashing separately would
    mean either buffering the layer or decompressing it twice.
    """

    __slots__ = ("_hasher", "_source", "_total")

    def __init__(self, source: ByteReader) -> None:
        self._source = source
        self._hasher = hashlib.sha256()
        self._total = 0

    def read(self, size: int = -1, /) -> bytes:
        block = self._source.read(size)
        self._hasher.update(block)
        self._total += len(block)
        return block

    @property
    def digest(self) -> Digest:
        return Digest(algorithm=SHA256, encoded=self._hasher.hexdigest())

    @property
    def size(self) -> int:
        return self._total


def _diff_ids(config_bytes: bytes) -> tuple[Digest, ...]:
    """The uncompressed digest of every layer, straight from the config.

    This is the fact that makes the whole uncompressed-storage design work: the
    image already commits to what each layer looks like *decompressed*, so
    storing it that way is not a reinterpretation, it is storing the thing the
    config already named.
    """
    document = json.loads(config_bytes)
    if not isinstance(document, dict):
        raise InvalidRequest("an image config must be a JSON object")
    rootfs = document.get("rootfs")
    if not isinstance(rootfs, dict) or not isinstance(rootfs.get("diff_ids"), list):
        raise InvalidRequest("an image config must have rootfs.diff_ids")
    return tuple(Digest.parse(str(entry)) for entry in rootfs["diff_ids"])


def _config_media_type(source_type: str) -> str:
    """Docker's config type becomes OCI's.

    The bytes are untouched, so the digest is untouched; only the *label* on the
    descriptor changes, and it has to, because the manifest carrying it is now
    declared as OCI and mixing the vocabularies is what makes some clients balk.
    """
    return MEDIA_IMAGE_CONFIG if source_type == MEDIA_DOCKER_CONFIG else source_type


def _rewrite_manifest(
    selected: SourceImage, config: Descriptor, layers: Sequence[Descriptor]
) -> bytes:
    """The stored manifest: the source's own, with two fields replaced.

    When the source had no manifest at all — docker's legacy archive genuinely
    does not have one — a minimal OCI manifest is authored instead. That is the
    only case where Ledger invents manifest structure, and it is invention out of
    nothing rather than a rebuild that could lose a field.
    """
    if selected.manifest_bytes is None:
        return canonical_json(
            {
                "schemaVersion": 2,
                "mediaType": MEDIA_IMAGE_MANIFEST,
                "config": config.to_json(),
                "layers": [layer.to_json() for layer in layers],
            }
        )

    document = parse_document(selected.manifest_bytes)
    document["mediaType"] = MEDIA_IMAGE_MANIFEST
    document["config"] = config.to_json()
    document["layers"] = [layer.to_json() for layer in layers]
    return canonical_json(document)
