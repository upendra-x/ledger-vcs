"""Synthetic container images. Defined once, in ``demo.images``.

The demo needs a real OCI archive to show an image being versioned, and the
tests need the same one. A format writer duplicated between them would drift,
and the copy nobody runs would drift first — so there is one, and the tests
import it from where the demo keeps it.
"""

from __future__ import annotations

from demo.images import (
    OCI_LAYER_GZIP,
    build_docker_archive,
    build_oci_archive,
    layer_tar,
    sha256_hex,
)

__all__ = [
    "OCI_LAYER_GZIP",
    "build_docker_archive",
    "build_oci_archive",
    "layer_tar",
    "sha256_hex",
]
