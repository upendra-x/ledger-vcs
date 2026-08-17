# Ledger

A version control system for RL environments, implementing the design in
[TECH_DESIGN.pdf](TECH_DESIGN.pdf).

Ledger versions an *environment* — a directory tree of task code, datasets, ML
models, verifiers and container images — and serves it as a **service** rather
than a tool. Every operation is one request against one named version: commit,
branch, fork, read a file, pull an image, trigger a build. Nothing clones,
nothing pushes, and no working copy lives on the server, because no caller is a
human at a terminal.

## Run the demo

```
uv sync
make demo
```

That stages a clean corpus, starts the server, and opens
<http://127.0.0.1:8080/console>.

The page has twelve cards, one per requirement, read from
`src.verify.REQUIREMENTS` — the same list `ledger verify-requirements` checks
against the test suite, so the page cannot claim a requirement the verifier does
not know about. Every number a card shows is measured during the run, not
written into the page.

| Button | |
| --- | --- |
| **Run the essentials** | five cards, twenty steps — the five-to-seven minute telling |
| **Reset & run all** | all twelve, about ninety seconds of execution |
| **Reset** | back to a clean corpus, running nothing |

Prefer a terminal:

```
uv run python demo/e2e.py          # the whole system, with measured numbers
uv run ledger verify-requirements  # which requirements this build demonstrates
```

## Run the service

```
uv run ledgerd --data-dir ./data --dev --port 8080
```

`--dev` makes unauthenticated requests act as a local administrator. It is off by
default, it announces itself loudly at startup, and it is a convenience for one
machine — never a deployment mode.

| Surface | Where | For |
| --- | --- | --- |
| HTTP API | `/v1` | background automations |
| OCI registry | `/v2` | container runtimes — `docker pull` works unmodified |
| Web UI | `/` | a person looking at what is actually in there. Read-only |
| Metrics | `/v1/metrics` | the observability assumptions, in Prometheus text format |
| OpenAPI | `/docs` | the API, generated |

## Run the tests

```
make check                      # ruff + mypy --strict + the suite
uv run pytest -m "not slow"     # the inner loop
```

## How it is put together

A module may import only from packages above it:

```
ids · errors · clock · metrics · text   pure values, no I/O
format/                              FROZEN naming, codec, FastCDC, tree shape
store/                               content-addressed storage, catalog, keep-sets
fs/                                  blob reads, tree lookup, path, diff, closure
meta/                                refs, operation log, idempotency, sessions, outbox
maintenance/                         garbage collection — reads both stores
runtime/                             ingest and materialize
oci/ · build/                        images; the build and sync pipeline
auth/ · migrate/                     capability tokens; git import
service/                             commits, environments, images
api/                                 /v1, /v2, the UI — the only layer that knows HTTP
cli/                                 one module per verb group
```

Each module's docstring explains why it is the way it is. The ones worth reading
first:

| | |
| --- | --- |
| [`format/constants.py`](src/format/constants.py) | what is frozen, and why changing it renames every object |
| [`format/shape.py`](src/format/shape.py) | content-defined, **level-salted** splitting — and what fixed fanout would break |
| [`store/cas.py`](src/store/cas.py) | why ingest re-encodes rather than only re-hashing |
| [`meta/repository.py`](src/meta/repository.py) | generation CAS, the dense operation counter, idempotency |
| [`fs/closure.py`](src/fs/closure.py) | the walk that decides what a sweep may take |
| [`api/deps.py`](src/api/deps.py) | one authorization gate, and why a 403 and a 404 have to be indistinguishable |

`jj-backend/` implements `jj_lib::backend::Backend` against a running Ledger,
pinned to **jj-lib 0.44.0** — files, symlinks, trees and commits are Ledger's
objects, and the crate never computes an object name.
