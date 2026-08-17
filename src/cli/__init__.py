"""The ``ledger`` command-line interface.

Deliberately shaped around jj's vocabulary rather than git's plumbing, because
the concepts should transfer to agents that already reason in commits,
changes, bookmarks, operations and undo.

The CLI holds no logic of its own: every subcommand is a thin call into a
service or, once the server exists, into ``src.client``. That is what keeps
the API and the CLI from drifting into two different products.
"""

from __future__ import annotations
