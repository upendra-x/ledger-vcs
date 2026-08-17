"""Synthetic container images, built in memory.

The integration suite must not need a container daemon: a test that only runs on
a machine with Docker is a test that stops running. Everything here produces the
same bytes ``docker save`` produces — a real OCI image layout, with gzipped
layers and a config whose ``diff_ids`` are the uncompressed digests — so the
ingest path under test is the real one.

Determinism matters as much as realism. Layer tars are built with fixed mtimes
and gzip is given ``mtime=0``, so the same logical image produces the same bytes
every run; a test asserting "the second ingest stored nothing" would otherwise
pass or fail on the clock.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "OCI_LAYER_GZIP",
    "build_docker_archive",
    "build_oci_archive",
    "layer_tar",
    "sha256_hex",
]

OCI_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def layer_tar(files: Mapping[str, bytes]) -> bytes:
    """An uncompressed layer: a tar of the given files, byte-stable."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            info.mtime = 0
            info.uid = info.gid = 0
            tar.addfile(info, io.BytesIO(files[name]))
    return buffer.getvalue()


def _config(diff_ids: Sequence[str], *, architecture: str, os_name: str) -> bytes:
    document = {
        "architecture": architecture,
        "os": os_name,
        "config": {"Cmd": ["/bin/sh"]},
        "rootfs": {"type": "layers", "diff_ids": [f"sha256:{d}" for d in diff_ids]},
    }
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode()


def build_oci_archive(
    path: Path,
    layers: Sequence[bytes],
    *,
    image: str = "demo:latest",
    architecture: str = "amd64",
    os_name: str = "linux",
    layer_media_type: str = OCI_LAYER_GZIP,
    nested_index: bool = False,
    extra_manifest: Mapping[str, Any] | None = None,
    compress: bool = True,
) -> Path:
    """Write an OCI image layout tarball. Returns ``path``.

    ``nested_index`` reproduces what modern ``docker save`` actually emits: the
    top-level index points at a *second* index which lists per-platform
    manifests, with a build attestation alongside them at ``unknown/unknown``.
    That shape is what a naive "take the first manifest" reader gets wrong.
    """
    blobs: dict[str, bytes] = {}

    def store(data: bytes) -> str:
        digest = sha256_hex(data)
        blobs[digest] = data
        return digest

    diff_ids = [sha256_hex(layer) for layer in layers]
    layer_descriptors = []
    for layer in layers:
        payload = gzip.compress(layer, mtime=0) if compress else layer
        layer_descriptors.append(
            {
                "mediaType": layer_media_type,
                "digest": f"sha256:{store(payload)}",
                "size": len(payload),
            }
        )

    config_bytes = _config(diff_ids, architecture=architecture, os_name=os_name)
    manifest: dict[str, Any] = {
        "schemaVersion": 2,
        "mediaType": OCI_MANIFEST,
        "config": {
            "mediaType": OCI_CONFIG,
            "digest": f"sha256:{store(config_bytes)}",
            "size": len(config_bytes),
        },
        "layers": layer_descriptors,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    manifest_digest = store(manifest_bytes)

    platform = {"architecture": architecture, "os": os_name}
    manifest_descriptor = {
        "mediaType": OCI_MANIFEST,
        "digest": f"sha256:{manifest_digest}",
        "size": len(manifest_bytes),
        "platform": platform,
    }

    if nested_index:
        attestation = json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": OCI_MANIFEST,
                "config": {"mediaType": OCI_CONFIG, "digest": f"sha256:{'0' * 64}", "size": 0},
                "layers": [],
            },
            separators=(",", ":"),
        ).encode()
        inner = json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": OCI_INDEX,
                "manifests": [
                    {
                        "mediaType": OCI_MANIFEST,
                        "digest": f"sha256:{store(attestation)}",
                        "size": len(attestation),
                        "platform": {"architecture": "unknown", "os": "unknown"},
                    },
                    manifest_descriptor,
                ],
            },
            separators=(",", ":"),
        ).encode()
        top = {
            "mediaType": OCI_INDEX,
            "digest": f"sha256:{store(inner)}",
            "size": len(inner),
            "annotations": {"org.opencontainers.image.ref.name": image},
        }
    else:
        top = {**manifest_descriptor, "annotations": {"org.opencontainers.image.ref.name": image}}

    index_bytes = json.dumps(
        {"schemaVersion": 2, "mediaType": OCI_INDEX, "manifests": [top]},
        separators=(",", ":"),
    ).encode()

    members = {
        "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        "index.json": index_bytes,
        **{f"blobs/sha256/{digest}": data for digest, data in blobs.items()},
    }
    _write_tar(path, members)
    return path


def build_docker_archive(
    path: Path, layers: Sequence[bytes], *, tag: str = "legacy:latest"
) -> Path:
    """Docker's *legacy* archive: uncompressed layers, addressed by file path.

    Still what a daemon without the containerd snapshotter writes, and the only
    format in which layers arrive already uncompressed — a code path that would
    otherwise never run.
    """
    diff_ids = [sha256_hex(layer) for layer in layers]
    config_bytes = _config(diff_ids, architecture="amd64", os_name="linux")
    config_name = f"{sha256_hex(config_bytes)}.json"

    members: dict[str, bytes] = {config_name: config_bytes}
    layer_paths: list[str] = []
    for index, layer in enumerate(layers):
        member = f"layer{index}/layer.tar"
        members[member] = layer
        layer_paths.append(member)

    members["manifest.json"] = json.dumps(
        [{"Config": config_name, "RepoTags": [tag], "Layers": layer_paths}]
    ).encode()
    _write_tar(path, members)
    return path


def _write_tar(path: Path, members: Mapping[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name])
            info.mtime = 0
            tar.addfile(info, io.BytesIO(members[name]))
