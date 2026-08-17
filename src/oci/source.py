"""Reading an image out of whatever a container tool handed us.

``docker save`` produces two different things depending on which snapshotter the
daemon runs: a modern **OCI image layout** (``oci-layout``, ``index.json``,
``blobs/sha256/…``) or the **legacy docker archive** (``manifest.json`` plus
per-layer directories). Both are supported, and an extracted directory of either
works as well as its tarball, because the difference is a detail of how a file
was delivered rather than anything about the image.

Everything below this module speaks one vocabulary: a config blob, an ordered
list of layer blobs, and a way to open each as a stream. That is deliberately
narrower than "an OCI layout" — the ingester must never need to know that a
layer is a tar member in one case and a file in another, or it acquires two code
paths where the interesting work happens.

Streams, never buffers. A layer is opened, decompressed, chunked and released;
nothing here ever holds a multi-gigabyte layer in memory.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, final

from src.errors import InvalidRequest, NotFound
from src.oci.digest import Digest
from src.oci.media import (
    MEDIA_DOCKER_LAYER,
    MEDIA_DOCKER_LAYER_GZIP,
    MEDIA_IMAGE_CONFIG,
    is_index,
)
from src.oci.model import (
    REF_NAME_ANNOTATION,
    Descriptor,
    ImageIndex,
    Platform,
    parse_index,
    parse_manifest,
    select_platform,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import IO

__all__ = [
    "ImageSource",
    "SourceBlob",
    "SourceImage",
    "open_image_source",
]

#: How many index levels to follow before deciding the archive is malformed. Two
#: is already generous — an index of indexes of manifests — and a bound is what
#: stops a self-referential index from looping forever.
_MAX_INDEX_DEPTH: Final = 4

#: Docker's own name annotation, which ``docker save`` writes and OCI does not
#: define. Read as a fallback so ``--image alpine:latest`` matches what the user
#: actually typed at the daemon.
_CONTAINERD_NAME: Final = "io.containerd.image.name"


@final
@dataclass(frozen=True, slots=True)
class SourceBlob:
    """One blob in the source, addressed however that source addresses blobs.

    ``reference`` is opaque outside the source that produced it: a digest for an
    OCI layout, a tar member path for a legacy archive. ``digest`` is present
    only when the source genuinely names the blob by hash — the legacy format
    does not, and inventing one by reading the whole layer would cost a pass over
    every byte for no benefit.
    """

    reference: str
    media_type: str
    size: int
    digest: Digest | None = None


@final
@dataclass(frozen=True, slots=True)
class SourceImage:
    """One platform's image: a config, ordered layers, and its own manifest."""

    config: SourceBlob
    layers: tuple[SourceBlob, ...]
    #: The manifest exactly as the source stored it, when the source had one.
    #: ``None`` for a legacy archive, which has no manifest to preserve — the
    #: only case in which Ledger authors one.
    manifest_bytes: bytes | None = None
    annotations: tuple[tuple[str, str], ...] = ()
    platform: Platform | None = None
    source_name: str | None = None


class ImageSource(Protocol):
    """Somewhere an image can be read from."""

    @property
    def image_names(self) -> tuple[str, ...]:
        """What this source calls the images it holds, for an error message."""
        ...

    def select(self, image: str | None, platform: Platform) -> SourceImage: ...

    def open(self, blob: SourceBlob) -> IO[bytes]: ...

    def read(self, blob: SourceBlob) -> bytes: ...

    def close(self) -> None: ...

    # A source owns an open file handle, so it is a context manager. Declared on
    # the protocol rather than left to each implementation, because a caller that
    # cannot write `with` around it will eventually forget to close one.
    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Byte access: a tarball and a directory, behind one interface
# ─────────────────────────────────────────────────────────────────────────────


class _Files(Protocol):
    """Named byte streams. The only thing the layouts below need."""

    def open(self, name: str) -> IO[bytes]: ...

    def size(self, name: str) -> int: ...

    def exists(self, name: str) -> bool: ...

    def close(self) -> None: ...

    def read(self, name: str) -> bytes: ...


@final
class _TarFiles:
    """A tarball, indexed once so members can be opened in any order."""

    __slots__ = ("_members", "_tar")

    def __init__(self, path: Path) -> None:
        # Deliberately not a `with`: the archive stays open for the lifetime of
        # the source, because a 6 GiB image is read layer by layer rather than
        # extracted. `close()` is the counterpart, and `open_image_source` calls
        # it on every failure path.
        self._tar = tarfile.open(path)  # noqa: SIM115
        self._members = {
            _normalise(member.name): member for member in self._tar.getmembers() if member.isfile()
        }

    def open(self, name: str) -> IO[bytes]:
        member = self._members.get(_normalise(name))
        if member is None:
            raise NotFound("this archive has no such member", member=name)
        stream = self._tar.extractfile(member)
        if stream is None:  # pragma: no cover - guarded by isfile() above
            raise NotFound("archive member is not a regular file", member=name)
        return stream

    def read(self, name: str) -> bytes:
        with self.open(name) as stream:
            return stream.read()

    def size(self, name: str) -> int:
        member = self._members.get(_normalise(name))
        if member is None:
            raise NotFound("this archive has no such member", member=name)
        return member.size

    def exists(self, name: str) -> bool:
        return _normalise(name) in self._members

    def close(self) -> None:
        self._tar.close()

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._members)


@final
class _DirectoryFiles:
    """An extracted layout on disk."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, name: str) -> Path:
        # Resolved and re-checked, so a crafted archive cannot name its way out
        # of the directory it was extracted into.
        candidate = (self._root / _normalise(name)).resolve()
        root = self._root.resolve()
        if not candidate.is_relative_to(root):
            raise InvalidRequest("archive member escapes the layout directory", member=name)
        return candidate

    def open(self, name: str) -> IO[bytes]:
        path = self._path(name)
        if not path.is_file():
            raise NotFound("this layout has no such file", member=name)
        return path.open("rb")

    def read(self, name: str) -> bytes:
        return self._path(name).read_bytes()

    def size(self, name: str) -> int:
        return self._path(name).stat().st_size

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def close(self) -> None:
        return None


def _normalise(name: str) -> str:
    return name.removeprefix("./").lstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
# The OCI image layout
# ─────────────────────────────────────────────────────────────────────────────


@final
class OciLayoutSource:
    """An OCI image layout — the format modern ``docker save`` writes.

    Also the format Ledger *stores* (``oci.layout``), which is not a coincidence:
    a round trip through Ledger and back out is the same shape, so the checkout
    of an environment is something ``skopeo`` can copy without conversion.
    """

    __slots__ = ("_files",)

    def __init__(self, files: _Files) -> None:
        self._files = files

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(sorted(_names_in(self._index())))

    def _index(self) -> ImageIndex:
        return parse_index(self._files.read("index.json"))

    def select(self, image: str | None, platform: Platform) -> SourceImage:
        index = self._index()
        descriptor = _pick_named(index.manifests, image, self.image_names)

        for _ in range(_MAX_INDEX_DEPTH):
            if not is_index(descriptor.media_type):
                break
            nested = parse_index(self.read(_blob(descriptor)))
            chosen = select_platform(nested.manifests, platform)
            if chosen is None:
                raise NotFound(
                    "this image has no manifest for that platform",
                    platform=str(platform),
                    available=sorted(
                        str(d.platform) for d in nested.manifests if d.platform is not None
                    ),
                )
            descriptor = chosen
        else:  # pragma: no cover - a four-deep index is already malformed
            raise InvalidRequest("image index nests too deeply", depth=_MAX_INDEX_DEPTH)

        manifest_bytes = self.read(_blob(descriptor))
        manifest = parse_manifest(manifest_bytes)
        return SourceImage(
            config=_blob(manifest.config),
            layers=tuple(_blob(layer) for layer in manifest.layers),
            manifest_bytes=manifest_bytes,
            annotations=manifest.annotations,
            platform=descriptor.platform,
            source_name=_name_of(descriptor),
        )

    def open(self, blob: SourceBlob) -> IO[bytes]:
        return self._files.open(_blob_path(blob.reference))

    def read(self, blob: SourceBlob) -> bytes:
        return self._files.read(_blob_path(blob.reference))

    def close(self) -> None:
        self._files.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _blob(descriptor: Descriptor) -> SourceBlob:
    return SourceBlob(
        reference=str(descriptor.digest),
        media_type=descriptor.media_type,
        size=descriptor.size,
        digest=descriptor.digest,
    )


def _blob_path(reference: str) -> str:
    digest = Digest.parse(reference)
    return f"blobs/{digest.algorithm}/{digest.encoded}"


def _name_of(descriptor: Descriptor) -> str | None:
    annotations = descriptor.annotation_map
    return annotations.get(REF_NAME_ANNOTATION) or annotations.get(_CONTAINERD_NAME)


def _names_in(index: object) -> Iterator[str]:
    manifests = getattr(index, "manifests", ())
    for descriptor in manifests:
        annotations = descriptor.annotation_map
        for key in (_CONTAINERD_NAME, REF_NAME_ANNOTATION):
            if key in annotations:
                yield annotations[key]


def _pick_named(
    descriptors: tuple[Descriptor, ...], image: str | None, available: tuple[str, ...]
) -> Descriptor:
    """Choose which of an archive's images was meant.

    A single-image archive needs no name, which is the overwhelmingly common
    case; anything else must say which one, and the error lists the choices
    rather than picking the first and hoping.
    """
    if not descriptors:
        raise InvalidRequest("this archive contains no images")
    if image is None:
        if len(descriptors) == 1:
            return descriptors[0]
        raise InvalidRequest(
            "this archive holds more than one image; name the one to ingest",
            available=list(available),
        )

    for descriptor in descriptors:
        annotations = descriptor.annotation_map
        candidates = {
            annotations.get(REF_NAME_ANNOTATION),
            annotations.get(_CONTAINERD_NAME),
        }
        if image in candidates or any(c and c.endswith(f"/{image}") for c in candidates):
            return descriptor
    raise NotFound("this archive holds no such image", image=image, available=list(available))


# ─────────────────────────────────────────────────────────────────────────────
# Docker's legacy archive
# ─────────────────────────────────────────────────────────────────────────────


@final
class DockerArchiveSource:
    """``docker save`` without the containerd snapshotter.

    Layers here are already *uncompressed* tars addressed by file path rather
    than by digest, which is why ``SourceBlob.digest`` is optional: the format
    genuinely does not name them by hash. Supporting it is worth the eighty lines
    because it is still what a large fraction of daemons emit, and an importer
    that only reads one of the two formats fails at the worst possible moment.
    """

    __slots__ = ("_entries", "_files")

    def __init__(self, files: _Files) -> None:
        self._files = files
        self._entries = _load_legacy_manifest(files)

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(sorted(tag for entry in self._entries for tag in entry.tags))

    def select(self, image: str | None, platform: Platform) -> SourceImage:
        del platform  # a legacy archive holds exactly one platform per entry
        entry = _pick_legacy(self._entries, image, self.image_names)
        config_bytes = self._files.read(entry.config)
        return SourceImage(
            config=SourceBlob(
                reference=entry.config,
                media_type=MEDIA_IMAGE_CONFIG,
                size=len(config_bytes),
                digest=Digest.of(config_bytes),
            ),
            layers=tuple(
                SourceBlob(
                    reference=layer,
                    media_type=(
                        MEDIA_DOCKER_LAYER_GZIP if layer.endswith(".gz") else MEDIA_DOCKER_LAYER
                    ),
                    size=self._files.size(layer),
                )
                for layer in entry.layers
            ),
            manifest_bytes=None,
            source_name=entry.tags[0] if entry.tags else None,
        )

    def open(self, blob: SourceBlob) -> IO[bytes]:
        return self._files.open(blob.reference)

    def read(self, blob: SourceBlob) -> bytes:
        return self._files.read(blob.reference)

    def close(self) -> None:
        self._files.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@final
@dataclass(frozen=True, slots=True)
class _LegacyEntry:
    config: str
    layers: tuple[str, ...]
    tags: tuple[str, ...]


def _load_legacy_manifest(files: _Files) -> tuple[_LegacyEntry, ...]:
    document: object = json.loads(files.read("manifest.json"))
    if not isinstance(document, list) or not document:
        raise InvalidRequest("a docker archive's manifest.json must be a non-empty array")

    entries: list[_LegacyEntry] = []
    for raw in document:
        if not isinstance(raw, dict) or "Config" not in raw or "Layers" not in raw:
            raise InvalidRequest("a docker archive entry needs 'Config' and 'Layers'")
        entries.append(
            _LegacyEntry(
                config=str(raw["Config"]),
                layers=tuple(str(layer) for layer in raw["Layers"]),
                tags=tuple(str(tag) for tag in (raw.get("RepoTags") or ())),
            )
        )
    return tuple(entries)


def _pick_legacy(
    entries: tuple[_LegacyEntry, ...], image: str | None, available: tuple[str, ...]
) -> _LegacyEntry:
    if image is None:
        if len(entries) == 1:
            return entries[0]
        raise InvalidRequest(
            "this archive holds more than one image; name the one to ingest",
            available=list(available),
        )
    for entry in entries:
        if image in entry.tags or any(tag.endswith(f"/{image}") for tag in entry.tags):
            return entry
    raise NotFound("this archive holds no such image", image=image, available=list(available))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────


def open_image_source(path: Path | str) -> ImageSource:
    """Open an image archive or layout, detecting which of the two it is.

    Detected by what the container actually holds rather than by extension: the
    same bytes arrive as ``image.tar``, ``image.tgz`` or an extracted directory,
    and the file name is the least reliable thing about them.
    """
    location = Path(path)
    if location.is_dir():
        return _dispatch(_DirectoryFiles(location), str(location))
    if not location.is_file():
        raise NotFound("no such image archive", path=str(location))

    tar_files = _TarFiles(location)
    try:
        return _dispatch(tar_files, str(location))
    except BaseException:
        # A tarball we could not make sense of still holds an open file handle,
        # and leaking one per failed import is the kind of thing that only shows
        # up as a puzzling limit hours later.
        tar_files.close()
        raise


def _dispatch(files: _Files, where: str) -> ImageSource:
    if files.exists("index.json"):
        return OciLayoutSource(files)
    if files.exists("manifest.json"):
        return DockerArchiveSource(files)
    raise InvalidRequest(
        "this is neither an OCI image layout nor a docker archive — no "
        "index.json and no manifest.json",
        path=where,
    )
