"""Adding a container image to an environment — as a version, not a tag.

The Multi-Container requirement is *"an old version should bring back the
same containers it was committed with"*, and this is where that is decided. An
image is added by writing its bytes into the commit's ``images/`` layout, so the
version **contains** the image. Restoring an old commit restores the same layers
because they are the same objects; there is no tag anywhere that could have moved
underneath it, and no registry that could have garbage-collected it.

An environment may hold any number of images, each under its own name, which is
the multi-container half of the requirement. Adding a second image rewrites one
directory listing and copies no layer bytes.

Publishing goes through the ordinary commit protocol — the same session, the same
generation compare-and-swap, the same keep-set graduation. There is no image-shaped
write path, because an image is not a special kind of content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from src.format.model import Commit
from src.fs.edit import empty_tree, set_path
from src.oci.ingest import ImageIngester
from src.oci.layout import IMAGES_DIR, build_images_entry, read_index, read_layout
from src.oci.model import Platform
from src.oci.source import open_image_source
from src.runtime.ingest import IngestStats
from src.service.commits import CommitService

if TYPE_CHECKING:
    from pathlib import Path

    from src.ids import EnvId, ObjectName, RefName
    from src.instance import Ledger
    from src.meta.models import Ref
    from src.oci.digest import Digest
    from src.oci.ingest import IngestedImage
    from src.oci.model import ImageIndex
    from src.store.cas import PutOutcome

__all__ = ["AddImageResult", "ImageService", "host_platform"]


def host_platform() -> Platform:
    """The platform an image should be selected for by default.

    The machine's own, because the overwhelmingly common intent is "the image I
    just built here". A multi-platform archive holds several; picking silently by
    position would produce an image that pulls and refuses to start.
    """
    import platform as _platform

    machine = _platform.machine().lower()
    architecture = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(
        machine, machine
    )
    return Platform(architecture=architecture, os="linux")


@final
@dataclass(frozen=True, slots=True)
class AddImageResult:
    """What adding an image cost, in the terms costs are reported in."""

    commit: ObjectName
    ref: RefName
    generation: int
    image: str
    manifest_digest: Digest
    layers: int
    layers_reused: int
    bytes_compressed: int
    bytes_uncompressed: int
    stats: IngestStats


@final
class ImageService:
    """Ingest an image and publish it as the next version of a ref."""

    __slots__ = ("_commits", "_ingester", "_ledger")

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger
        self._commits = CommitService(ledger)
        self._ingester = ImageIngester(ledger)

    def add(
        self,
        env: EnvId,
        ref: RefName,
        archive: Path,
        *,
        image: str,
        author: str,
        message: str = "",
        platform: Platform | None = None,
        from_image: str | None = None,
        idempotency_key: str | None = None,
    ) -> AddImageResult:
        """Add or replace one image in an environment, as a new commit."""
        wanted = platform or host_platform()
        ingested: IngestedImage | None = None

        def build(current: Ref | None) -> tuple[ObjectName, IngestStats]:
            nonlocal ingested
            root = self._root_tree(current)
            with open_image_source(archive) as source:
                ingested = self._ingester.ingest(
                    source, image=image, platform=wanted, from_image=from_image
                )

            # Recorded only after the content is stored, so a row can name a
            # missing object but never a name that was never written.
            self._ledger.digests.record(ingested.digest_entries)

            # Every write is tallied, including the layout's own small files and
            # the tree nodes above them. Counting only the layers would report
            # "0 objects created" for a commit that created several, which is
            # exactly the number a reader uses to judge deduplication.
            layout_stats = IngestStats()

            def write_bytes(data: bytes) -> ObjectName:
                nonlocal layout_stats
                name, stats = self._ledger.ingester.ingest_bytes(data)
                layout_stats = layout_stats + stats
                return name

            existing = read_layout(self._ledger.store, root)
            written: list[PutOutcome] = []
            entry = build_images_entry(
                self._ledger.store,
                index=existing.index.with_image(image, ingested.descriptor),
                blob_entries=(*existing.blob_entries, *ingested.blob_entries),
                write_bytes=write_bytes,
                shape=self._ledger.shape_params,
                written=written,
            )
            edit = set_path(
                self._ledger.store, root, IMAGES_DIR, entry, shape=self._ledger.shape_params
            )
            total = ingested.stats + layout_stats + IngestStats.counting((*written, *edit.written))
            return edit.tree, total

        result = self._commits.publish(
            env,
            ref,
            build,
            author=author,
            message=message or f"add image {image}",
            idempotency_key=idempotency_key,
        )
        if ingested is None:  # pragma: no cover - publish always runs the builder
            raise AssertionError("the tree builder did not run")
        return AddImageResult(
            commit=result.commit,
            ref=result.ref,
            generation=result.generation,
            image=image,
            manifest_digest=ingested.digest,
            layers=ingested.layers,
            layers_reused=ingested.layers_reused,
            bytes_compressed=ingested.bytes_compressed,
            bytes_uncompressed=ingested.bytes_uncompressed,
            stats=result.stats,
        )

    def list_images(self, env: EnvId, ref: RefName) -> ImageIndex:
        """The images a ref's current version holds."""
        commit = self._ledger.repo.get_ref(env, ref).target
        tree = self._ledger.store.get_as(commit, Commit).tree
        return read_index(self._ledger.store, tree)

    # ── internals ────────────────────────────────────────────────────────────

    def _root_tree(self, current: Ref | None) -> ObjectName:
        if current is None:
            return empty_tree(self._ledger.store, shape=self._ledger.shape_params)
        return self._ledger.store.get_as(current.target, Commit).tree
