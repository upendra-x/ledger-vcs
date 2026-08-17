"""Where images live inside a version, and how a pull finds them.

An environment's ``images/`` directory **is an OCI image layout** — the same
``oci-layout`` / ``index.json`` / ``blobs/sha256/…`` arrangement the standard
defines::

    /                                   tree
    ├── env.yaml                        blob     the environment manifest
    ├── data/train.bin       40 GiB     blob ──▶ index ──▶ 40,960 chunks
    └── images/                         tree
        ├── oci-layout                  blob
        ├── index.json                  blob     which manifest each image is
        └── blobs/sha256/               tree
            ├── <manifest digest>       blob
            ├── <config digest>         blob
            └── <layer digest>  2.4 GiB blob ──▶ index ──▶ 2,458 chunks

Two things follow from choosing the standard's layout rather than inventing one.
A checkout is directly usable by ``skopeo`` and ``crane``, so nothing about
Ledger has to be installed to get an image back out. And the registry endpoint becomes a path
lookup: serving ``blobs/<digest>`` is
``resolve_path`` on ``images/blobs/sha256/<hex>``, which means **a pull is
authorized exactly like a read, by environment and path** — the token that grants
``env:read`` grants the pull, and there is no registry credential anywhere.

That path lookup is also the security boundary. Serving a blob by looking its
SHA-256 up in a global index would hand any caller any layer in the corpus by
hash alone, which is precisely the bare-hash read that is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, final

from src.errors import NotFound
from src.format.constants import MODE_REGULAR, EntryKind
from src.format.model import TreeEntry
from src.format.shape import PRODUCTION_SHAPE, build_tree
from src.fs.blob import BlobReader
from src.fs.tree import iter_entries, resolve_path
from src.oci.digest import SHA256, Digest
from src.oci.model import ImageIndex, parse_index

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from src.format.shape import Emit, ShapeParams
    from src.ids import ObjectName
    from src.oci.model import Descriptor
    from src.store.cas import ObjectStore, PutOutcome

__all__ = [
    "IMAGES_DIR",
    "ImageLayout",
    "build_images_entry",
    "find_blob",
    "find_manifest",
    "read_index",
]

IMAGES_DIR: Final = "images"
INDEX_FILE: Final = "index.json"
LAYOUT_FILE: Final = "oci-layout"
BLOBS_DIR: Final = "blobs"

#: The marker file the OCI image-layout specification requires. Its content is
#: fixed, so it is one deduplicated blob shared by every environment that holds
#: an image — a nice illustration of the general property rather than a special
#: case anyone had to arrange.
LAYOUT_BYTES: Final = b'{"imageLayoutVersion":"1.0.0"}'

type BlobWriter = Callable[[bytes], ObjectName]


def _blob_path(digest: Digest) -> str:
    return f"{IMAGES_DIR}/{BLOBS_DIR}/{digest.algorithm}/{digest.encoded}"


def _index_path() -> str:
    return f"{IMAGES_DIR}/{INDEX_FILE}"


@final
@dataclass(frozen=True, slots=True)
class ImageLayout:
    """A commit's image layout, read once for editing.

    Holds the blob *entries* rather than the blob bytes: adding an image to an
    environment that already has ten rebuilds one directory listing and copies
    no content at all.
    """

    index: ImageIndex
    blob_entries: tuple[TreeEntry, ...]


def read_index(store: ObjectStore, root_tree: ObjectName) -> ImageIndex:
    """The environment's image index, or an empty one if it holds no images."""
    data = _read_file(store, root_tree, _index_path())
    return ImageIndex() if data is None else parse_index(data)


def read_layout(store: ObjectStore, root_tree: ObjectName) -> ImageLayout:
    return ImageLayout(
        index=read_index(store, root_tree),
        blob_entries=tuple(iter_blob_entries(store, root_tree)),
    )


def iter_blob_entries(store: ObjectStore, root_tree: ObjectName) -> Iterator[TreeEntry]:
    """Every blob the layout holds, in digest order.

    Streamed rather than collected, because the blobs directory of an
    environment with many images is the one wide directory in this layout.
    """
    try:
        resolved = resolve_path(store, root_tree, f"{IMAGES_DIR}/{BLOBS_DIR}/{SHA256}")
    except NotFound:
        return
    if resolved.kind is not EntryKind.TREE:  # pragma: no cover - malformed layout
        return
    yield from iter_entries(store, resolved.target)


def find_manifest(
    store: ObjectStore, root_tree: ObjectName, image: str
) -> tuple[Descriptor, bytes] | None:
    """The manifest a version pinned for one image, and its exact bytes.

    The bytes are served verbatim rather than re-serialized: a manifest's digest
    is the hash of those bytes, and a client that re-hashes what it received —
    which every container runtime does — would reject anything else.
    """
    descriptor = read_index(store, root_tree).named(image)
    if descriptor is None:
        return None
    data = _read_file(store, root_tree, _blob_path(descriptor.digest))
    if data is None:  # pragma: no cover - index and blobs are written together
        return None
    return descriptor, data


def find_blob(store: ObjectStore, root_tree: ObjectName, digest: Digest) -> TreeEntry | None:
    """The tree entry for one blob of this version's images, by OCI digest."""
    try:
        resolved = resolve_path(store, root_tree, _blob_path(digest))
    except NotFound:
        return None
    if resolved.kind is not EntryKind.BLOB:  # pragma: no cover - malformed layout
        return None
    return resolved.entry


def _read_file(store: ObjectStore, root_tree: ObjectName, path: str) -> bytes | None:
    try:
        resolved = resolve_path(store, root_tree, path)
    except NotFound:
        return None
    if resolved.kind is not EntryKind.BLOB:  # pragma: no cover - malformed layout
        return None
    return BlobReader(store, resolved.target).read_all()


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────


def build_images_entry(
    store: ObjectStore,
    *,
    index: ImageIndex,
    blob_entries: Sequence[TreeEntry],
    write_bytes: BlobWriter,
    shape: ShapeParams = PRODUCTION_SHAPE,
    written: list[PutOutcome] | None = None,
) -> TreeEntry:
    """Build the ``images/`` subtree and return the entry naming it.

    Deterministic: the same set of images produces the same tree whatever order
    they were added in, because the blob directory is sorted by digest and the
    index is sorted by image name. Two environments that converged on the same
    images therefore share every node — the ordinary deduplication property,
    which would be lost if this appended in arrival order.
    """
    collected: list[PutOutcome] = [] if written is None else written
    emit = _emitter(store, collected)

    deduplicated = {entry.name: entry for entry in blob_entries}
    algorithm_tree = build_tree([deduplicated[key] for key in sorted(deduplicated)], emit, shape)
    blobs_tree = build_tree([_tree_entry(SHA256.encode(), algorithm_tree)], emit, shape)

    index_bytes = index.to_bytes()
    entries = [
        _tree_entry(BLOBS_DIR.encode(), blobs_tree),
        _file_entry(INDEX_FILE.encode(), write_bytes(index_bytes), len(index_bytes)),
        _file_entry(LAYOUT_FILE.encode(), write_bytes(LAYOUT_BYTES), len(LAYOUT_BYTES)),
    ]
    entries.sort(key=lambda entry: entry.name)
    return _tree_entry(IMAGES_DIR.encode(), build_tree(entries, emit, shape))


def blob_entry(digest: Digest, target: ObjectName, size: int) -> TreeEntry:
    """The layout entry for one stored blob. Its *name* is the digest hex.

    Naming the file by its content digest is what makes the registry's blob route
    a path lookup, and it is the layout specification's own rule rather than a
    convention Ledger chose.
    """
    return _file_entry(digest.encoded.encode(), target, size)


def _tree_entry(name: bytes, target: ObjectName) -> TreeEntry:
    return TreeEntry(name=name, kind=EntryKind.TREE, target=target, mode=0, size=0)


def _file_entry(name: bytes, target: ObjectName, size: int) -> TreeEntry:
    return TreeEntry(name=name, kind=EntryKind.BLOB, target=target, mode=MODE_REGULAR, size=size)


def _emitter(store: ObjectStore, written: list[PutOutcome]) -> Emit:
    def emit(name: ObjectName, framed: bytes) -> None:
        written.append(store.put_encoded(name, framed))

    return emit
