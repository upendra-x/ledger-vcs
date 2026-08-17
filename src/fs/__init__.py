"""Reading content out of the store: files, directories, paths.

Every module here is pure *given an ObjectStore* — no globals, no ambient
configuration, and the only side effect anywhere is an idempotent ``put``. That
is what lets the read path be tested against an in-memory store with no server,
no filesystem and no clock.
"""

from __future__ import annotations
