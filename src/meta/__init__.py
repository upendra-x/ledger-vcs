"""The mutable column: which environments exist, and where their refs point.

Roughly 21M rows at the full scale, against 20 PiB of content — and
every concurrency property in the system is decided here. Two layers:

    repository.py   Ledger nouns. Ref CAS, op log, idempotency, undo, forks.
                    All semantics; one implementation for every backend.
    store.py        Items, conditions, single-partition transactions.
                    No domain knowledge whatsoever.
"""

from __future__ import annotations
