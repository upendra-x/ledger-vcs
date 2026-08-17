"""One build pipeline, versioned centrally.

Today every environment carries its own GitHub Action that builds it and syncs
data into the platform — ten million copies of nearly the same workflow, each
free to drift, each edited one at a time when the process changes. Ledger
replaces all of them with one pipeline to which an environment contributes only
its manifest.

::

    UpdateRef succeeds
         │   ref-update event, ordered per environment
         ▼
    build queue ──▶ worker
                       │  (1) materialize the commit
                       │  (2) read the manifest — the only component in
                       │      Ledger that parses it
                       │  (3) build
                       │  (4) sync artifacts into the platform
                       │  (5) record note#<commit>#sync
                       ▼
                     done

Two properties do the real work.

**Events come from the operation log via the change stream**, so an event can
never describe a commit that was not published and there is no second write to
keep consistent. Delivery is at-least-once — the honest guarantee for a durable
queue — and the *effect* is exactly-once because a build is keyed by commit.

**A build is a pure function of a commit.** A commit captures its content rather
than pointing at storage that can change, so its inputs are fixed forever. Two
refs at the same commit build once; a fork that changed nothing inherits its
parent's build; re-triggering an unchanged commit returns the existing result.
"""

from __future__ import annotations
