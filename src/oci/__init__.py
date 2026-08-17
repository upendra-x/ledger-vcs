"""Container images, stored as ordinary content.

An image is not a reference to a registry tag that can be retagged
underneath a version — it is bytes the commit *contains*. Each layer is a blob,
chunked exactly like a dataset, and ``images/`` is a genuine OCI image layout so
a checkout is something ``skopeo`` and ``crane`` already understand.

Nothing in ``src.store`` or ``src.fs`` knows any of this. Everything here
is a mapping between two vocabularies — OCI's digests and media types on one
side, Ledger's trees and blobs on the other — and the mapping is the whole
module. That is why the Multi-Container requirement is met by *storing* rather
than by pointing.
"""

from __future__ import annotations
