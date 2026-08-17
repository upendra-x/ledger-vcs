"""Everything whose output is an object *name*.

Every module in this package is pure: no I/O, no clock, no randomness, no
ambient configuration. Given the same inputs they produce the same bytes on
every machine, in every process, forever — which is what makes an object's
name the hash of its bytes.

That purity is what lets this package be exercised without a store, a server or
a fixture — and it is a rule to hold to when editing here, not one anything
checks for you.
"""

from __future__ import annotations
