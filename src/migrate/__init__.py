"""Getting an existing corpus in.

Delivery is phased as Read → Write → Build → Efficiency, which puts
*GitHub
import* in the first phase, alongside the object model itself. The claim it
supports is the one that makes adoption possible at all:

    *"Importing alone solves the problem, because reads are 7,000 of
    every 7,001 requests. The rate-limit pain disappears before Ledger owns a
    single byte of truth, so every later phase is taken by choice rather than
    under pressure."*

A version control system that can only hold environments created inside it is one
nobody can move to. This is the door.

It is also the measuring instrument for the first open question —
*"is 2 GiB of
unique content per environment right? Migration should measure it before the
year-10 numbers are relied on"* — which is why the importer reports what it
actually deduplicated rather than only what it converted.
"""

from __future__ import annotations
