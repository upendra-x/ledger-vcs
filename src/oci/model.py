"""OCI descriptors, manifests and image indexes.

There is a deliberate asymmetry here, and it is the difference between an
importer that loses information and one that does not:

**Manifests are *edited*, never re-authored.** A manifest arrives from somebody
else's build. Ledger rewrites exactly one field — the layer list, because the
layers are now stored uncompressed (``media``) — and every other key survives
byte for byte, including ones this version has never heard of. Re-authoring from
a typed model would silently drop ``subject``, ``artifactType`` and any future
field, and the loss would only surface as a signature that no longer verifies.

**The image index is authored.** ``images/index.json`` is entirely Ledger's own:
it says which manifest each image name refers to, and nothing else has a claim
on its shape.

Serialization is compact and deterministic, because a manifest's digest is the
hash of its bytes: the same image ingested twice must produce identical bytes or
it deduplicates against nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, final

from src.errors import InvalidRequest
from src.oci.digest import Digest
from src.oci.media import MEDIA_IMAGE_INDEX, is_index, is_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "REF_NAME_ANNOTATION",
    "Descriptor",
    "ImageIndex",
    "Manifest",
    "Platform",
    "canonical_json",
    "parse_index",
    "parse_manifest",
]

type JsonDocument = dict[str, Any]

#: The standard annotation naming what a manifest is called. It is how an image
#: layout records "this manifest is `app`", and therefore how a pull of
#: ``…/proximal/demo/app:main`` finds the right manifest in a commit.
REF_NAME_ANNOTATION: Final = "org.opencontainers.image.ref.name"

SCHEMA_VERSION: Final = 2


def canonical_json(document: JsonDocument) -> bytes:
    """Serialize compactly and reproducibly.

    Key *order* is the document's own, not sorted: a manifest we are editing must
    keep the order it arrived in so that the only difference from the original is
    the field we actually changed.
    """
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()


def _annotations(raw: object) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    if not isinstance(raw, dict):
        raise InvalidRequest("annotations must be an object")
    return tuple(sorted((str(k), str(v)) for k, v in raw.items()))


@final
@dataclass(frozen=True, slots=True)
class Platform:
    """What a manifest in an index is built for."""

    architecture: str
    os: str
    variant: str | None = None

    def matches(self, other: Platform) -> bool:
        """Whether ``self`` satisfies a request for ``other``.

        A variant is only compared when the *request* names one, so asking for
        ``linux/arm64`` accepts ``linux/arm64/v8`` — which is what a host means
        by its own platform, and what every runtime does.
        """
        if (self.architecture, self.os) != (other.architecture, other.os):
            return False
        return other.variant is None or self.variant == other.variant

    def __str__(self) -> str:
        base = f"{self.os}/{self.architecture}"
        return f"{base}/{self.variant}" if self.variant else base

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Platform:
        return cls(
            architecture=str(raw.get("architecture", "")),
            os=str(raw.get("os", "")),
            variant=str(raw["variant"]) if raw.get("variant") else None,
        )

    def to_json(self) -> JsonDocument:
        document: JsonDocument = {"architecture": self.architecture, "os": self.os}
        if self.variant:
            document["variant"] = self.variant
        return document


@final
@dataclass(frozen=True, slots=True)
class Descriptor:
    """A pointer to content, by digest. The atom of the whole format."""

    media_type: str
    digest: Digest
    size: int
    annotations: tuple[tuple[str, str], ...] = ()
    platform: Platform | None = None

    @property
    def annotation_map(self) -> dict[str, str]:
        return dict(self.annotations)

    def annotated(self, **extra: str) -> Descriptor:
        merged = {**self.annotation_map, **extra}
        return Descriptor(
            media_type=self.media_type,
            digest=self.digest,
            size=self.size,
            annotations=tuple(sorted(merged.items())),
            platform=self.platform,
        )

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Descriptor:
        for required in ("mediaType", "digest", "size"):
            if required not in raw:
                raise InvalidRequest(f"descriptor is missing {required!r}", descriptor=sorted(raw))
        platform = raw.get("platform")
        return cls(
            media_type=str(raw["mediaType"]),
            digest=Digest.parse(str(raw["digest"])),
            size=int(raw["size"]),
            annotations=_annotations(raw.get("annotations")),
            platform=Platform.from_json(platform) if isinstance(platform, dict) else None,
        )

    def to_json(self) -> JsonDocument:
        document: JsonDocument = {
            "mediaType": self.media_type,
            "digest": str(self.digest),
            "size": self.size,
        }
        if self.platform is not None:
            document["platform"] = self.platform.to_json()
        if self.annotations:
            document["annotations"] = dict(self.annotations)
        return document


@final
@dataclass(frozen=True, slots=True)
class Manifest:
    """One image, for one platform: a config blob and an ordered layer list.

    Parsed for reading only. The bytes it came from are what gets stored and
    served, because a manifest's digest is the hash of those exact bytes.
    """

    media_type: str
    config: Descriptor
    layers: tuple[Descriptor, ...]
    annotations: tuple[tuple[str, str], ...] = ()


@final
@dataclass(frozen=True, slots=True)
class ImageIndex:
    """A set of named manifests — ``images/index.json``.

    Ledger authors this one, and uses it for a purpose an ordinary registry
    solves with tags: mapping an image *name* inside an environment to the
    manifest that version pinned.
    """

    manifests: tuple[Descriptor, ...] = ()
    annotations: tuple[tuple[str, str], ...] = ()
    media_type: str = MEDIA_IMAGE_INDEX

    def named(self, image: str) -> Descriptor | None:
        for descriptor in self.manifests:
            if descriptor.annotation_map.get(REF_NAME_ANNOTATION) == image:
                return descriptor
        return None

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for descriptor in self.manifests
            if (name := descriptor.annotation_map.get(REF_NAME_ANNOTATION)) is not None
        )

    def with_image(self, image: str, descriptor: Descriptor) -> ImageIndex:
        """Add or replace one image, keeping the list ordered by name.

        Ordered because the index is content that gets hashed: two environments
        that ended up with the same images must end up with the same index bytes,
        whatever order the images were added in.
        """
        annotated = descriptor.annotated(**{REF_NAME_ANNOTATION: image})
        kept = [d for d in self.manifests if d.annotation_map.get(REF_NAME_ANNOTATION) != image]
        kept.append(annotated)
        kept.sort(key=lambda d: (d.annotation_map.get(REF_NAME_ANNOTATION, ""), str(d.digest)))
        return ImageIndex(manifests=tuple(kept), annotations=self.annotations)

    def to_json(self) -> JsonDocument:
        document: JsonDocument = {
            "schemaVersion": SCHEMA_VERSION,
            "mediaType": self.media_type,
            "manifests": [d.to_json() for d in self.manifests],
        }
        if self.annotations:
            document["annotations"] = dict(self.annotations)
        return document

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_json())


def _load(data: bytes, what: str) -> JsonDocument:
    try:
        document = json.loads(data)
    except json.JSONDecodeError as exc:
        raise InvalidRequest(f"{what} is not valid JSON", error=str(exc)) from exc
    if not isinstance(document, dict):
        raise InvalidRequest(f"{what} must be a JSON object")
    return document


def parse_index(data: bytes) -> ImageIndex:
    document = _load(data, "an image index")
    raw = document.get("manifests")
    if not isinstance(raw, list):
        raise InvalidRequest("an image index must have a 'manifests' array")
    return ImageIndex(
        manifests=tuple(Descriptor.from_json(entry) for entry in raw),
        annotations=_annotations(document.get("annotations")),
        media_type=str(document.get("mediaType", MEDIA_IMAGE_INDEX)),
    )


def parse_manifest(data: bytes) -> Manifest:
    document = _load(data, "an image manifest")
    config = document.get("config")
    layers = document.get("layers")
    if not isinstance(config, dict) or not isinstance(layers, list):
        raise InvalidRequest("an image manifest must have 'config' and a 'layers' array")
    return Manifest(
        media_type=str(document.get("mediaType", "")),
        config=Descriptor.from_json(config),
        layers=tuple(Descriptor.from_json(entry) for entry in layers),
        annotations=_annotations(document.get("annotations")),
    )


def parse_document(data: bytes) -> JsonDocument:
    """The raw document, for editing rather than modelling."""
    return _load(data, "a manifest")


def select_platform(descriptors: Iterable[Descriptor], wanted: Platform) -> Descriptor | None:
    """Pick the manifest for a platform, ignoring attestations.

    Build attestations ride along in modern image indexes as manifests with
    platform ``unknown/unknown``. They are not runnable images, and a naive
    "first manifest" pick lands on one often enough to matter.
    """
    for descriptor in descriptors:
        platform = descriptor.platform
        if platform is None or platform.architecture == "unknown":
            continue
        if not (is_manifest(descriptor.media_type) or is_index(descriptor.media_type)):
            continue
        if platform.matches(wanted):
            return descriptor
    return None
