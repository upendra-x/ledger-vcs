"""The OCI vocabulary: digests, descriptors, and the index Ledger authors.

Three properties are load-bearing and none of them is obvious from the types:

* a digest is a *different* namespace from an object name, and confusing the two
  must be impossible rather than merely unlikely;
* the index Ledger writes is content that gets hashed, so it has to be a
  function of the set of images and not of the order they arrived in;
* selecting a platform out of a modern image index must skip build attestations,
  which sit alongside real manifests and are not runnable.
"""

from __future__ import annotations

import json

import pytest

from src.errors import InvalidRequest
from src.oci.digest import Digest, sha256_of
from src.oci.media import (
    MEDIA_LAYER_TAR,
    Compression,
    compression_of,
    uncompressed_layer_type,
)
from src.oci.model import (
    REF_NAME_ANNOTATION,
    Descriptor,
    ImageIndex,
    Platform,
    canonical_json,
    parse_index,
    parse_manifest,
    select_platform,
)


class TestDigest:
    def test_parses_and_renders(self) -> None:
        digest = Digest.parse("sha256:" + "ab" * 32)
        assert digest.algorithm == "sha256"
        assert str(digest) == "sha256:" + "ab" * 32

    def test_of_bytes_matches_hashlib(self) -> None:
        import hashlib

        assert Digest.of(b"hello").encoded == hashlib.sha256(b"hello").hexdigest()

    @pytest.mark.parametrize(
        "text",
        [
            "ab" * 32,  # no algorithm
            "sha512:" + "ab" * 64,  # an algorithm we deliberately do not serve
            "sha256:" + "AB" * 32,  # uppercase is not the canonical spelling
            "sha256:abc",  # wrong length
            "sha256:" + "zz" * 32,  # not hex
        ],
    )
    def test_rejects_anything_it_cannot_serve(self, text: str) -> None:
        """Rejected at construction, not at use.

        A digest that is merely *stored* wrong surfaces as a puzzling 404 from a
        container runtime hours later; rejected here it names its own problem.
        """
        with pytest.raises(InvalidRequest):
            Digest.parse(text)

    def test_streaming_digest_matches_whole_buffer(self) -> None:
        """The one-pass hash used while a layer is decompressed and chunked."""
        blocks = [b"one", b"two", b"three"]
        digest, size = sha256_of(blocks)
        assert digest == Digest.of(b"".join(blocks))
        assert size == len(b"".join(blocks))

    def test_is_ordered_so_a_blob_listing_is_canonical(self) -> None:
        digests = [Digest.of(bytes([n])) for n in range(8)]
        assert sorted(digests) == sorted(digests, key=str)


class TestCompression:
    @pytest.mark.parametrize(
        ("media_type", "expected"),
        [
            ("application/vnd.oci.image.layer.v1.tar+gzip", Compression.GZIP),
            ("application/vnd.docker.image.rootfs.diff.tar.gzip", Compression.GZIP),
            ("application/vnd.oci.image.layer.v1.tar+zstd", Compression.ZSTD),
            ("application/vnd.oci.image.layer.v1.tar", Compression.NONE),
            ("application/vnd.docker.image.rootfs.diff.tar", Compression.NONE),
        ],
    )
    def test_reads_compression_from_the_media_type(
        self, media_type: str, expected: Compression
    ) -> None:
        assert compression_of(media_type) is expected

    def test_foreign_layers_are_refused_with_a_reason(self) -> None:
        """Their bytes were never included, so there is nothing to version.

        Silently dropping one produces an image that pulls and then fails to
        start — a failure that surfaces on the rollout host rather than here.
        """
        with pytest.raises(InvalidRequest, match="foreign"):
            compression_of("application/vnd.docker.image.rootfs.foreign.diff.tar.gzip")

    def test_unknown_layer_types_are_refused(self) -> None:
        with pytest.raises(InvalidRequest, match="unrecognised"):
            compression_of("application/x-something-else")

    def test_everything_becomes_an_oci_tar(self) -> None:
        """Even a Docker-typed layer: what we store is a plain tar, and saying
        otherwise would claim a compression that is no longer there.
        """
        assert (
            uncompressed_layer_type("application/vnd.docker.image.rootfs.diff.tar.gzip")
            == MEDIA_LAYER_TAR
        )


class TestPlatform:
    def test_a_request_without_a_variant_accepts_one(self) -> None:
        """``linux/arm64`` is what a host calls itself; ``linux/arm64/v8`` is what
        an index calls the same thing.
        """
        assert Platform("arm64", "linux", "v8").matches(Platform("arm64", "linux"))

    def test_a_request_with_a_variant_is_exact(self) -> None:
        assert not Platform("arm64", "linux", "v8").matches(Platform("arm64", "linux", "v7"))

    def test_architecture_must_match(self) -> None:
        assert not Platform("amd64", "linux").matches(Platform("arm64", "linux"))


def _descriptor(name: str, digest_source: bytes, platform: Platform | None = None) -> Descriptor:
    return Descriptor(
        media_type="application/vnd.oci.image.manifest.v1+json",
        digest=Digest.of(digest_source),
        size=len(digest_source),
        annotations=((REF_NAME_ANNOTATION, name),),
        platform=platform,
    )


class TestImageIndex:
    def test_is_a_function_of_the_set_not_the_order(self) -> None:
        """The index is content that gets hashed.

        Two environments that ended up holding the same images must end up with
        the same index bytes, or the one node every image environment shares
        stops deduplicating.
        """
        a = _descriptor("app", b"a")
        b = _descriptor("db", b"b")
        forwards = ImageIndex().with_image("app", a).with_image("db", b)
        backwards = ImageIndex().with_image("db", b).with_image("app", a)
        assert forwards.to_bytes() == backwards.to_bytes()

    def test_replacing_an_image_keeps_one_entry(self) -> None:
        index = ImageIndex().with_image("app", _descriptor("app", b"v1"))
        index = index.with_image("app", _descriptor("app", b"v2"))
        assert index.image_names == ("app",)
        assert index.named("app") is not None
        assert index.named("app").digest == Digest.of(b"v2")  # type: ignore[union-attr]

    def test_round_trips_through_json(self) -> None:
        index = ImageIndex().with_image("app", _descriptor("app", b"a"))
        assert parse_index(index.to_bytes()) == index

    def test_names_the_image_it_was_added_under(self) -> None:
        """Regardless of what the source archive called it — an image's name
        inside an environment is the environment's business.
        """
        index = ImageIndex().with_image("app", _descriptor("something-else", b"a"))
        assert index.image_names == ("app",)


class TestSelectPlatform:
    def test_skips_build_attestations(self) -> None:
        """They ride along in every modern image index at ``unknown/unknown``.

        A "first manifest wins" reader picks one often enough to matter, and the
        result pulls successfully and then cannot be run.
        """
        attestation = _descriptor("att", b"att", Platform("unknown", "unknown"))
        real = _descriptor("app", b"app", Platform("arm64", "linux", "v8"))
        chosen = select_platform([attestation, real], Platform("arm64", "linux"))
        assert chosen == real

    def test_returns_none_when_nothing_matches(self) -> None:
        real = _descriptor("app", b"app", Platform("amd64", "linux"))
        assert select_platform([real], Platform("arm64", "linux")) is None


class TestSerialization:
    def test_canonical_json_is_compact_and_stable(self) -> None:
        document = {"b": 1, "a": 2}
        assert canonical_json(document) == b'{"b":1,"a":2}'

    def test_key_order_is_the_documents_own(self) -> None:
        """A manifest we are editing must keep the order it arrived in, so the
        only difference from the original is the field we actually changed.
        """
        assert canonical_json({"z": 1, "a": 2}) != canonical_json({"a": 2, "z": 1})

    def test_a_manifest_missing_its_parts_is_refused(self) -> None:
        with pytest.raises(InvalidRequest, match="config"):
            parse_manifest(json.dumps({"schemaVersion": 2}).encode())

    def test_a_descriptor_missing_a_field_names_the_field(self) -> None:
        with pytest.raises(InvalidRequest, match="size"):
            Descriptor.from_json({"mediaType": "x", "digest": "sha256:" + "ab" * 32})
