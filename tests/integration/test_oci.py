"""Container images end to end.

This is where the **Multi-Container** requirement is demonstrated: *"an old
version should bring back the same containers it was committed with"*. The proof
is a pull, a ref moved backwards, and a second pull that yields the earlier
image — with no tag anywhere that could have moved underneath it.

Three claims are checked rather than asserted in prose:

* **layers are stored uncompressed**, so content-defined chunking works on them
  and a rebuild that changed one file shares the base layer;
* **the config survives byte for byte**, so decompressing changes what an image
  *weighs* and not what it *is*;
* **a pull is authorized exactly like a read** — a layer that exists in another
  environment is not served by knowing its hash.

Everything here builds its own images in memory (``tests/support/images.py``).
The real ``docker pull`` proof lives in ``tests/e2e/test_docker_pull.py``, which
skips when there is no daemon; this file must run anywhere.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import build_app
from src.api.deps import AppState
from src.auth.model import NamePrefixSelector, Operation, Principal, Scope
from src.clock import ManualClock
from src.errors import InvalidRequest
from src.format.cdc import ChunkParams
from src.format.model import Commit
from src.format.shape import ShapeParams
from src.fs.blob import BlobReader
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.oci.digest import Digest
from src.oci.layout import find_blob, read_index
from src.oci.media import MEDIA_IMAGE_MANIFEST
from src.oci.model import Platform
from src.oci.source import open_image_source
from src.service.images import ImageService
from support.images import build_docker_archive, build_oci_archive, layer_tar, sha256_hex

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.ids import EnvId

ORG = "proximal"
ENV = "proximal/demo"
MAIN = RefName("refs/heads/main")
LINUX_AMD64 = Platform("amd64", "linux")

#: Deliberately lopsided: the base layer is large and the layers stacked on
#: top of it are tiny, so "the base was not stored again" is a claim the byte
#: counts can actually distinguish rather than one lost in rounding.
BASE_LAYER = layer_tar({"etc/os-release": b"NAME=demo\n", "bin/sh": b"#!/bin/sh\n" * 20000})
TOP_LAYER = layer_tar({"app/main.py": b"print('hello')\n"})
OTHER_LAYER = layer_tar({"opt/other": b"x" * 200})


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(start_us=1_700_000_000_000_000)


@pytest.fixture
def ledger(tmp_path: Path, clock: ManualClock) -> Iterator[Ledger]:
    """Test-scale chunk and shape parameters, so multi-level structures appear
    in kilobytes rather than in gigabytes.
    """
    with Ledger(
        tmp_path / "ledger",
        clock=clock,
        chunk_params=ChunkParams.for_average(4096),
        shape_params=ShapeParams(
            domain=b"ledger.tree.split.v1",
            period=32,
            min_entries=4,
            max_entries=64,
            max_node_bytes=16 * 1024,
        ),
        shard_count=4,
    ) as opened:
        yield opened


@pytest.fixture
def env_id(ledger: Ledger) -> EnvId:
    return ledger.repo.create_env(EnvName(ENV)).env_id


@pytest.fixture
def images(ledger: Ledger) -> ImageService:
    return ImageService(ledger)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return build_oci_archive(tmp_path / "image.tar", [BASE_LAYER, TOP_LAYER])


def add(
    images: ImageService, env_id: EnvId, archive: Path, *, image: str = "app", ref: RefName = MAIN
) -> Any:
    return images.add(env_id, ref, archive, image=image, author="tester", platform=LINUX_AMD64)


def root_tree(ledger: Ledger, env_id: EnvId, ref: RefName = MAIN) -> Any:
    commit = ledger.repo.get_ref(env_id, ref).target
    return ledger.store.get_as(commit, Commit).tree


def stored_blob(ledger: Ledger, env_id: EnvId, digest: Digest) -> bytes:
    entry = find_blob(ledger.store, root_tree(ledger, env_id), digest)
    assert entry is not None, f"no blob {digest} in this version"
    return BlobReader(ledger.store, entry.target).read_all()


def rewrite_tar(source: Path, target: Path, replace: dict[str, bytes]) -> Path:
    """Copy an archive, substituting members. Damage, applied deliberately."""
    members: dict[str, bytes] = {}
    with tarfile.open(source) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            assert handle is not None
            members[member.name] = handle.read()
    members.update(replace)
    with tarfile.open(target, "w") as tar:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name])
            info.mtime = 0
            tar.addfile(info, io.BytesIO(members[name]))
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────────────────────


class TestIngest:
    def test_layers_are_stored_uncompressed(
        self, ledger: Ledger, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """The central trade, checked on the bytes.

        The stored blob is the *tar*, not the gzip stream: that is what makes
        content-defined chunking see through a rebuild, and it is why the layer's
        digest is now its ``diff_id``.
        """
        add(images, env_id, archive)

        index = read_index(ledger.store, root_tree(ledger, env_id))
        manifest = json.loads(stored_blob(ledger, env_id, index.named("app").digest))  # type: ignore[union-attr]

        for layer, source in zip(manifest["layers"], (BASE_LAYER, TOP_LAYER), strict=True):
            assert layer["mediaType"] == "application/vnd.oci.image.layer.v1.tar"
            assert layer["digest"] == f"sha256:{sha256_hex(source)}"
            assert stored_blob(ledger, env_id, Digest.parse(layer["digest"])) == source

    def test_the_original_compressed_digest_is_recorded(
        self, ledger: Ledger, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """The compressed digest is kept as metadata: it is how a caller recognises
        the layer they pushed, and what a future "retain the gzip too" option
        keys on.
        """
        add(images, env_id, archive)
        index = read_index(ledger.store, root_tree(ledger, env_id))
        manifest = json.loads(stored_blob(ledger, env_id, index.named("app").digest))  # type: ignore[union-attr]

        annotations = manifest["layers"][0]["annotations"]
        expected = sha256_hex(gzip.compress(BASE_LAYER, mtime=0))
        assert annotations["dev.ledger.oci.compressed.digest"] == f"sha256:{expected}"
        assert annotations["dev.ledger.oci.compressed.mediaType"].endswith("+gzip")

    def test_the_config_survives_byte_for_byte(
        self, ledger: Ledger, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """Decompressing changes what an image weighs, not what it is.

        A layer's uncompressed digest *is* its ``diff_id``, which the config
        already listed — so no value inside the config changes, and its digest is
        the one the source build produced.
        """
        with open_image_source(archive) as source:
            original = source.read(source.select(None, LINUX_AMD64).config)

        add(images, env_id, archive)
        index = read_index(ledger.store, root_tree(ledger, env_id))
        manifest = json.loads(stored_blob(ledger, env_id, index.named("app").digest))  # type: ignore[union-attr]

        assert manifest["config"]["digest"] == f"sha256:{sha256_hex(original)}"
        assert stored_blob(ledger, env_id, Digest.parse(manifest["config"]["digest"])) == original

    def test_manifest_fields_ledger_does_not_model_are_preserved(
        self, tmp_path: Path, ledger: Ledger, images: ImageService, env_id: EnvId
    ) -> None:
        """The manifest is *edited*, not re-authored.

        Rebuilding it from a typed model would drop any field this version has
        never heard of, and the loss would only surface as a signature that no
        longer verifies.
        """
        archive = build_oci_archive(
            tmp_path / "annotated.tar",
            [BASE_LAYER],
            extra_manifest={
                "artifactType": "application/vnd.example.thing",
                "annotations": {"org.opencontainers.image.source": "https://example.invalid"},
            },
        )
        add(images, env_id, archive)

        index = read_index(ledger.store, root_tree(ledger, env_id))
        manifest = json.loads(stored_blob(ledger, env_id, index.named("app").digest))  # type: ignore[union-attr]
        assert manifest["artifactType"] == "application/vnd.example.thing"
        assert manifest["annotations"]["org.opencontainers.image.source"].endswith(
            "example.invalid"
        )

    def test_the_layout_is_a_real_oci_image_layout(
        self, tmp_path: Path, ledger: Ledger, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """Checked out, ``images/`` is something other tools can read.

        The strongest available check without installing one: Ledger's own layout
        reader — which knows nothing about Ledger storage and only reads the
        standard's files — finds the same image in the materialized directory.
        """
        add(images, env_id, archive)
        commit = ledger.repo.get_ref(env_id, MAIN).target
        checkout = tmp_path / "checkout"
        ledger.materializer.materialize_commit(commit, checkout)

        assert (checkout / "images" / "oci-layout").read_bytes() == (
            b'{"imageLayoutVersion":"1.0.0"}'
        )
        with open_image_source(checkout / "images") as source:
            selected = source.select("app", LINUX_AMD64)
            assert len(selected.layers) == 2
            # Layers on the way back out are the uncompressed tars we stored.
            assert source.read(selected.layers[0]) == BASE_LAYER

    def test_a_second_identical_image_stores_no_layer_bytes(
        self, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """Deduplication as *work avoided*, not merely as storage saved.

        A layer's ``diff_id`` is known from the config before a single byte of it
        is read, so a layer already held is never decompressed at all.
        """
        add(images, env_id, archive)
        again = add(images, env_id, archive, image="app-copy")

        assert again.layers_reused == again.layers == 2
        assert again.stats.bytes_stored < 4096, (
            "a second copy of the same image should cost a directory listing, "
            f"not content — stored {again.stats.bytes_stored} bytes"
        )

    def test_a_rebuild_shares_its_base_layer(
        self, tmp_path: Path, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """The reason layers are stored uncompressed at all."""
        add(images, env_id, archive)
        rebuilt = build_oci_archive(tmp_path / "v2.tar", [BASE_LAYER, OTHER_LAYER])
        second = add(images, env_id, rebuilt, image="app2")

        assert (second.layers, second.layers_reused) == (2, 1)
        # Adding the rebuild costs about what its *new* layer costs. The base
        # layer is five times larger and was not stored again.
        assert second.stats.bytes_stored < 2 * len(OTHER_LAYER), (
            f"adding a rebuild stored {second.stats.bytes_stored} bytes for a "
            f"{len(OTHER_LAYER)}-byte new layer; the {len(BASE_LAYER)}-byte base "
            f"layer looks to have been stored a second time"
        )

    def test_a_nested_index_selects_the_platform_rather_than_the_attestation(
        self, tmp_path: Path, ledger: Ledger, images: ImageService, env_id: EnvId
    ) -> None:
        """What ``docker save`` actually emits.

        A build attestation sits alongside the real manifests at
        ``unknown/unknown``; picking it produces an image that pulls cleanly and
        cannot be run.
        """
        archive = build_oci_archive(tmp_path / "multi.tar", [BASE_LAYER], nested_index=True)
        add(images, env_id, archive)

        index = read_index(ledger.store, root_tree(ledger, env_id))
        manifest = json.loads(stored_blob(ledger, env_id, index.named("app").digest))  # type: ignore[union-attr]
        assert len(manifest["layers"]) == 1
        assert manifest["layers"][0]["digest"] == f"sha256:{sha256_hex(BASE_LAYER)}"

    def test_the_legacy_docker_archive_is_supported(
        self, tmp_path: Path, ledger: Ledger, images: ImageService, env_id: EnvId
    ) -> None:
        """A daemon without the containerd snapshotter writes this, and its
        layers are *already* uncompressed — the one path where nothing has to be
        unwrapped.
        """
        archive = build_docker_archive(tmp_path / "legacy.tar", [BASE_LAYER])
        result = add(images, env_id, archive)

        assert result.layers == 1
        assert stored_blob(ledger, env_id, Digest.of(BASE_LAYER)) == BASE_LAYER

    def test_a_layer_that_disagrees_with_its_diff_id_is_refused(
        self, tmp_path: Path, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """The free integrity check the format hands us.

        The config already committed to what each layer looks like decompressed.
        A layer that does not match it is either a damaged archive or not the
        compression it declared — and storing it produces an image that pulls and
        then fails to unpack, on the rollout host rather than here.
        """
        compressed = gzip.compress(BASE_LAYER, mtime=0)
        damaged = rewrite_tar(
            archive,
            tmp_path / "damaged.tar",
            {f"blobs/sha256/{sha256_hex(compressed)}": gzip.compress(OTHER_LAYER, mtime=0)},
        )
        with pytest.raises(InvalidRequest, match="diff_id"):
            add(images, env_id, damaged)

    def test_a_foreign_layer_is_refused(
        self, tmp_path: Path, images: ImageService, env_id: EnvId
    ) -> None:
        """Its bytes were never included; Ledger versions content, not pointers."""
        archive = build_oci_archive(
            tmp_path / "foreign.tar",
            [BASE_LAYER],
            layer_media_type="application/vnd.docker.image.rootfs.foreign.diff.tar.gzip",
        )
        with pytest.raises(InvalidRequest, match="foreign"):
            add(images, env_id, archive)

    def test_a_multi_image_archive_must_say_which_image(
        self, tmp_path: Path, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """Choosing the first and hoping is how the wrong image reaches production."""
        with tarfile.open(archive) as tar:
            handle = tar.extractfile("index.json")
            assert handle is not None
            index = json.loads(handle.read())
        first = index["manifests"][0]
        second = {**first, "annotations": {"org.opencontainers.image.ref.name": "other"}}
        index["manifests"] = [first, second]
        ambiguous = rewrite_tar(
            archive, tmp_path / "two.tar", {"index.json": json.dumps(index).encode()}
        )

        with pytest.raises(InvalidRequest, match="more than one image"):
            add(images, env_id, ambiguous)

    def test_naming_the_image_picks_it(
        self, tmp_path: Path, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        with tarfile.open(archive) as tar:
            handle = tar.extractfile("index.json")
            assert handle is not None
            index = json.loads(handle.read())
        first = index["manifests"][0]
        index["manifests"] = [
            {**first, "annotations": {"org.opencontainers.image.ref.name": "wanted"}},
            {**first, "annotations": {"org.opencontainers.image.ref.name": "other"}},
        ]
        two = rewrite_tar(archive, tmp_path / "two.tar", {"index.json": json.dumps(index).encode()})

        result = images.add(
            env_id,
            MAIN,
            two,
            image="app",
            author="tester",
            platform=LINUX_AMD64,
            from_image="wanted",
        )
        assert result.layers == 2

    def test_an_archive_that_is_neither_format_is_refused(
        self, tmp_path: Path, images: ImageService, env_id: EnvId
    ) -> None:
        junk = tmp_path / "junk.tar"
        with tarfile.open(junk, "w") as tar:
            info = tarfile.TarInfo("hello.txt")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))
        with pytest.raises(InvalidRequest, match="neither"):
            add(images, env_id, junk)


class TestMultipleImages:
    def test_an_environment_holds_any_number_of_images(
        self, tmp_path: Path, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """Multi-Container: composed environments, met by storing rather than
        by pointing.
        """
        add(images, env_id, archive, image="app")
        add(images, env_id, build_oci_archive(tmp_path / "db.tar", [OTHER_LAYER]), image="db")

        index = images.list_images(env_id, MAIN)
        assert index.image_names == ("app", "db")

    def test_replacing_one_image_leaves_the_other_alone(
        self, tmp_path: Path, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        add(images, env_id, archive, image="app")
        add(images, env_id, build_oci_archive(tmp_path / "db.tar", [OTHER_LAYER]), image="db")
        before = images.list_images(env_id, MAIN).named("db")

        add(images, env_id, build_oci_archive(tmp_path / "v2.tar", [TOP_LAYER]), image="app")
        after = images.list_images(env_id, MAIN)

        assert after.named("db") == before
        assert after.named("app") != before


class TestDigestIndexIsOnlyAHint:
    def test_a_stale_row_never_resurrects_a_swept_layer(
        self, ledger: Ledger, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        """Reachability-based retention, at the one place a second index could break it.

        The alternate-digest index remembers "this SHA-256 is that object". If it
        were trusted on its own, a layer swept by collection would be *skipped*
        on re-ingest and the commit would reference content that is gone. So the
        hit is confirmed against the store's own existence predicate.
        """
        add(images, env_id, archive)
        layer = Digest.of(BASE_LAYER)
        row = ledger.digests.lookup(layer.algorithm, layer.encoded)
        assert row is not None

        # Sweep the layer the way collection would, tombstone included.
        ledger.store.tombstones.record([row.name], expires_at_us=2**62)
        ledger.store.delete([row.name])
        assert ledger.digests.lookup(layer.algorithm, layer.encoded) is not None, (
            "this test is only meaningful while the stale row is still there"
        )

        again = add(images, env_id, archive, image="again")
        assert again.layers_reused < again.layers, "a swept layer was reused from a stale row"
        assert stored_blob(ledger, env_id, layer) == BASE_LAYER

    def test_collection_forgets_rows_for_what_it_swept(
        self, ledger: Ledger, images: ImageService, env_id: EnvId, archive: Path
    ) -> None:
        add(images, env_id, archive)
        layer = Digest.of(BASE_LAYER)
        row = ledger.digests.lookup(layer.algorithm, layer.encoded)
        assert row is not None

        ledger.digests.forget([row.name])
        assert ledger.digests.lookup(layer.algorithm, layer.encoded) is None


# ─────────────────────────────────────────────────────────────────────────────
# The registry endpoint
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def app(ledger: Ledger) -> Any:
    return build_app(ledger=ledger, clock=ledger.clock)


@pytest.fixture
def state(app: Any) -> AppState:
    return app.state.ledger_state  # type: ignore[no-any-return]


@pytest.fixture
def client(app: Any) -> Iterator[TestClient]:
    with TestClient(app) as opened:
        yield opened


@pytest.fixture
def published(images: ImageService, env_id: EnvId, archive: Path) -> Any:
    return add(images, env_id, archive)


def token_for(
    state: AppState, *operations: Operation, prefix: str = f"{ORG}/*", principal: str = "puller"
) -> str:
    combined = Operation(0)
    for operation in operations:
        combined |= operation
    scope = Scope(operations=combined, selectors=(NamePrefixSelector(prefix),))
    return state.signer.mint(Principal(principal), scope, ttl_us=3600 * 1_000_000)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


REPO = f"/v2/{ENV}/app"


class TestRegistry:
    def test_the_version_check_announces_a_registry(self, client: TestClient) -> None:
        """The handshake every client makes first. Unauthenticated on purpose: a
        401 here only teaches the client to retry with a credential it will be
        asked for on the next request anyway.
        """
        response = client.get("/v2/")
        assert response.status_code == 200
        assert response.headers["docker-distribution-api-version"] == "registry/2.0"

    def test_a_pull_by_tag_serves_the_manifest_the_version_pinned(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        response = client.get(
            f"{REPO}/manifests/main", headers=bearer(token_for(state, Operation.READ))
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == MEDIA_IMAGE_MANIFEST
        assert response.headers["docker-content-digest"] == str(published.manifest_digest)

    def test_manifest_bytes_are_served_verbatim(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """A manifest's digest is the hash of its bytes, and every runtime
        re-hashes what it received — so re-serializing it, even identically-
        meaning, would make every pull fail verification.
        """
        del published
        response = client.get(
            f"{REPO}/manifests/main", headers=bearer(token_for(state, Operation.READ))
        )
        assert response.headers["docker-content-digest"] == f"sha256:{sha256_hex(response.content)}"

    def test_the_whole_pull_sequence_reaches_real_bytes(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """Manifest, then config, then every layer — what a runtime actually does."""
        del published
        headers = bearer(token_for(state, Operation.READ))
        manifest = client.get(f"{REPO}/manifests/main", headers=headers).json()

        config = client.get(f"{REPO}/blobs/{manifest['config']['digest']}", headers=headers)
        assert config.status_code == 200
        assert f"sha256:{sha256_hex(config.content)}" == manifest["config"]["digest"]

        for descriptor, expected in zip(manifest["layers"], (BASE_LAYER, TOP_LAYER), strict=True):
            blob = client.get(f"{REPO}/blobs/{descriptor['digest']}", headers=headers)
            assert blob.status_code == 200
            assert blob.content == expected
            assert int(blob.headers["content-length"]) == len(expected)

    def test_a_manifest_can_be_fetched_by_digest(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        headers = bearer(token_for(state, Operation.READ))
        response = client.get(f"{REPO}/manifests/{published.manifest_digest}", headers=headers)
        assert response.status_code == 200
        assert response.headers["docker-content-digest"] == str(published.manifest_digest)

    def test_head_carries_the_headers_and_no_body(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """How a client checks whether it already has an image without pulling it."""
        headers = bearer(token_for(state, Operation.READ))
        head = client.head(f"{REPO}/manifests/main", headers=headers)
        get = client.get(f"{REPO}/manifests/main", headers=headers)
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["docker-content-digest"] == str(published.manifest_digest)
        # The length must describe the manifest, not the empty body — it is what
        # the client uses to decide whether to fetch it at all.
        assert head.headers["content-length"] == get.headers["content-length"]

    def test_a_ranged_blob_read_returns_exactly_that_range(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """Through the registry: resuming a download does not read
        the bytes before the offset.
        """
        del published
        headers = bearer(token_for(state, Operation.READ))
        digest = f"sha256:{sha256_hex(BASE_LAYER)}"
        response = client.get(
            f"{REPO}/blobs/{digest}", headers={**headers, "Range": "bytes=1024-2047"}
        )
        assert response.status_code == 206
        assert response.content == BASE_LAYER[1024:2048]
        assert response.headers["content-range"] == f"bytes 1024-2047/{len(BASE_LAYER)}"

    def test_a_suffix_range_returns_the_end_of_the_blob(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """``bytes=-100`` is the LAST hundred bytes, not the first hundred.

        Read the wrong way it answers 206 with entirely the wrong content and the
        client has no way to tell — the worst failure this route can have, since
        every other error at least announces itself.
        """
        del published
        headers = bearer(token_for(state, Operation.READ))
        response = client.get(
            f"{REPO}/blobs/sha256:{sha256_hex(BASE_LAYER)}",
            headers={**headers, "Range": "bytes=-100"},
        )
        assert response.status_code == 206
        assert response.content == BASE_LAYER[-100:]
        assert response.headers["content-range"] == (
            f"bytes {len(BASE_LAYER) - 100}-{len(BASE_LAYER) - 1}/{len(BASE_LAYER)}"
        )

    def test_an_open_ended_range_runs_to_the_end(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        del published
        headers = bearer(token_for(state, Operation.READ))
        start = len(BASE_LAYER) - 64
        response = client.get(
            f"{REPO}/blobs/sha256:{sha256_hex(BASE_LAYER)}",
            headers={**headers, "Range": f"bytes={start}-"},
        )
        assert response.status_code == 206
        assert response.content == BASE_LAYER[start:]

    def test_a_range_beyond_the_blob_is_refused(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        del published
        headers = bearer(token_for(state, Operation.READ))
        response = client.get(
            f"{REPO}/blobs/sha256:{sha256_hex(BASE_LAYER)}",
            headers={**headers, "Range": f"bytes={len(BASE_LAYER)}-{len(BASE_LAYER) + 10}"},
        )
        assert response.status_code == 400

    def test_a_multi_range_request_is_refused_rather_than_half_answered(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """Answering only the first range looks like success and delivers the
        wrong bytes.
        """
        del published
        headers = bearer(token_for(state, Operation.READ))
        response = client.get(
            f"{REPO}/blobs/sha256:{sha256_hex(BASE_LAYER)}",
            headers={**headers, "Range": "bytes=0-10,20-30"},
        )
        assert response.status_code == 400
        assert response.json()["errors"][0]["code"] == "UNSUPPORTED"

    def test_tags_list_only_names_refs_that_hold_the_image(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId, published: Any
    ) -> None:
        """A tag that resolved to a version without the image would 404 on the
        very next request.
        """
        del published
        ledger.repo.create_ref(
            env_id,
            RefName("refs/heads/empty"),
            _commit_without_images(ledger),
            principal="tester",
        )
        response = client.get(f"{REPO}/tags/list", headers=bearer(token_for(state, Operation.READ)))
        assert response.json() == {"name": f"{ENV}/app", "tags": ["main"]}


class TestOldVersionsBringBackTheirImages:
    def test_moving_the_ref_back_serves_the_earlier_image(
        self,
        tmp_path: Path,
        client: TestClient,
        state: AppState,
        images: ImageService,
        ledger: Ledger,
        env_id: EnvId,
        archive: Path,
    ) -> None:
        """**Multi-Container, demonstrated.**

        The commit pins the manifest digest, so there is no tag that could have
        moved underneath a version. Rolling the ref back is one row update and no
        content moves at all — the old layers never went anywhere.
        """
        first = add(images, env_id, archive)
        second = add(
            images, env_id, build_oci_archive(tmp_path / "v2.tar", [BASE_LAYER, OTHER_LAYER])
        )
        assert first.manifest_digest != second.manifest_digest

        headers = bearer(token_for(state, Operation.READ))
        assert client.get(f"{REPO}/manifests/main", headers=headers).headers[
            "docker-content-digest"
        ] == str(second.manifest_digest)

        from src.service.commits import CommitService

        CommitService(ledger).revert(env_id, MAIN, first.commit, author="tester")

        assert client.get(f"{REPO}/manifests/main", headers=headers).headers[
            "docker-content-digest"
        ] == str(first.manifest_digest), "the old version did not bring back its own image"


class TestRegistryAuthorization:
    def test_a_pull_without_a_credential_asks_for_one(
        self, client: TestClient, published: Any
    ) -> None:
        del published
        response = client.get(f"{REPO}/manifests/main")
        assert response.status_code == 401
        assert "Basic" in response.headers["www-authenticate"]
        assert response.json()["errors"][0]["code"] == "UNAUTHORIZED"

    def test_basic_auth_carries_the_token_as_the_password(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """The only credential shape ``docker login`` can be told to send.

        It ends at the same verifier as a bearer token — the registry has no
        separate idea of who anyone is, which is exactly the "no
        registry credential exists".
        """
        del published
        import base64

        token = token_for(state, Operation.READ)
        credential = base64.b64encode(f"puller:{token}".encode()).decode()
        response = client.get(
            f"{REPO}/manifests/main", headers={"Authorization": f"Basic {credential}"}
        )
        assert response.status_code == 200

    def test_a_token_for_another_environment_is_denied(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        del published
        outsider = token_for(state, Operation.READ, prefix="someone-else/*")
        response = client.get(f"{REPO}/manifests/main", headers=bearer(outsider))
        assert response.status_code == 403
        assert response.json()["errors"][0]["code"] == "DENIED"

    def test_write_authority_is_not_required_to_pull(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        """The token granting ``env:read`` grants the pull."""
        del published
        response = client.get(
            f"{REPO}/manifests/main", headers=bearer(token_for(state, Operation.READ))
        )
        assert response.status_code == 200

    def test_a_layer_in_another_environment_is_not_served_by_knowing_its_hash(
        self, tmp_path: Path, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId
    ) -> None:
        """The bare-hash read, closed at the registry surface.

        A blob is found by resolving a *path* inside a version of the environment
        in the repository name. Looking the SHA-256 up in a global index instead
        would hand any caller any layer in the corpus, because a hash is not a
        credential — hashes leak through logs, diffs and manifests.
        """
        del env_id
        secret = layer_tar({"secret/weights.bin": b"S" * 9000})
        other = ledger.repo.create_env(EnvName("proximal/other")).env_id
        ImageService(ledger).add(
            other,
            MAIN,
            build_oci_archive(tmp_path / "secret.tar", [secret]),
            image="app",
            author="owner",
            platform=LINUX_AMD64,
        )

        # A perfectly valid token for the *demo* environment, and the exact hash.
        response = client.get(
            f"{REPO}/blobs/sha256:{sha256_hex(secret)}",
            headers=bearer(token_for(state, Operation.READ)),
        )
        assert response.status_code == 404
        assert response.json()["errors"][0]["code"] == "BLOB_UNKNOWN"

        # And it is genuinely reachable where it belongs, so the 404 above is
        # about authorization rather than about the blob not existing.
        allowed = client.get(
            f"/v2/proximal/other/app/blobs/sha256:{sha256_hex(secret)}",
            headers=bearer(token_for(state, Operation.READ)),
        )
        assert allowed.status_code == 200
        assert allowed.content == secret


class TestRegistryErrorContract:
    def test_an_unknown_tag_is_a_manifest_unknown(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        del published
        response = client.get(
            f"{REPO}/manifests/nope", headers=bearer(token_for(state, Operation.READ))
        )
        assert response.status_code == 404
        body = response.json()
        assert body["errors"][0]["code"] == "MANIFEST_UNKNOWN"
        # The distribution shape, not Ledger's: a runtime that gets the wrong one
        # reports "unknown error" with no detail at all.
        assert "errors" in body
        assert "code" not in body

    def test_an_unknown_environment_is_a_name_unknown(
        self, client: TestClient, state: AppState
    ) -> None:
        response = client.get(
            "/v2/proximal/absent/app/manifests/main",
            headers=bearer(token_for(state, Operation.READ)),
        )
        assert response.status_code == 404
        assert response.json()["errors"][0]["code"] == "NAME_UNKNOWN"

    def test_a_malformed_digest_is_reported_in_the_registry_shape(
        self, client: TestClient, state: AppState, published: Any
    ) -> None:
        del published
        response = client.get(
            f"{REPO}/blobs/sha256:not-a-digest", headers=bearer(token_for(state, Operation.READ))
        )
        assert response.status_code == 400
        assert response.json()["errors"][0]["code"] == "UNSUPPORTED"

    def test_the_main_api_keeps_its_own_error_shape(self, client: TestClient) -> None:
        """Guards the guard: a path-selected contract must not leak sideways."""
        response = client.get("/v1/envs/proximal/absent")
        assert "errors" not in response.json()


class TestRegistryRouteCoverage:
    def test_every_registry_route_declares_authorization(self) -> None:
        """A missing check is invisible until someone exploits it."""
        from src.api.oci import router

        public = {"/v2/", "/v2"}
        unguarded: list[str] = []
        for route in router.routes:
            path = getattr(route, "path", "")
            if path in public:
                continue
            dependant = getattr(route, "dependant", None)
            if dependant is None:  # pragma: no cover - every APIRoute has one
                continue
            if "dependency" not in _dependency_names(dependant):
                unguarded.append(f"{sorted(getattr(route, 'methods', []))} {path}")
        assert not unguarded, "registry routes with no authorization:\n  " + "\n  ".join(unguarded)

    def test_the_coverage_check_sees_the_real_routes(self) -> None:
        """Without this, a walk that found nothing would pass vacuously."""
        from src.api.oci import router

        paths = {getattr(r, "path", "") for r in router.routes}
        assert {p for p in paths if "manifests" in p or "blobs" in p}


def _dependency_names(dependant: Any, depth: int = 0) -> set[str]:
    if depth > 5:
        return set()
    names = {getattr(dependant.call, "__name__", "")}
    for sub in dependant.dependencies:
        names |= _dependency_names(sub, depth + 1)
    return names


def _commit_without_images(ledger: Ledger) -> Any:
    from src.fs.edit import empty_tree
    from src.ids import ChangeId

    tree = empty_tree(ledger.store, shape=ledger.shape_params)
    return ledger.store.put_object(
        Commit(
            tree=tree,
            parents=(),
            change_id=ChangeId("cd" * 16),
            author="tester",
            committer="tester",
            timestamp_us=ledger.clock.now_us(),
            message="no images",
        )
    ).name
