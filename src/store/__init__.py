"""The immutable column: content-addressed objects and the machinery around them.

The dependency direction inside this package is one-way. ``backend`` knows about
bytes and keys; ``cas`` knows about names and composes a backend, a catalog and
a tombstone store; nothing knows about environments, refs or HTTP.
"""

from __future__ import annotations
