"""Work that keeps the two stores consistent with each other.

Everything here reads *both* columns — the immutable objects and the mutable
names — and belongs to neither. Garbage collection is the example that forced the
package to exist: it asks the metadata store what is still reachable and the
object store what is still stored, and the answer is the difference.

It lived under ``src.store`` before, which made that package's own docstring
false ("nothing here knows about environments, refs or HTTP") and let the
collector reach up into the metadata repository unnoticed. A layering test now
checks the direction, so the mistake reports itself.
"""

from __future__ import annotations
