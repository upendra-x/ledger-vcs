"""The page, and the two things it needs that are not ordinary API calls.

**A name distribution.** How the corpus spreads across the leading bits of its
own hashes, counted by kind. Read from the catalog's ``iter_shard`` — the same
prefix partitioning the collector walks — rather than from a directory scan, so
the cost follows the index rather than the store.

**Corpus totals as JSON.** ``/v1/metrics`` is Prometheus text, which is the
right answer for a collector and unreadable to a chart.

Everything else on the page is a plain ``fetch`` to ``/v1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from demo.portal.context import DEMO_ENV, MAIN, DemoDep
from src.errors import NotFound
from src.format.constants import FORMAT_FINGERPRINT, ObjectKind
from src.metrics import Snapshot, snapshot
from src.verify import REQUIREMENTS

__all__ = ["ASSETS", "router"]

router = APIRouter(prefix="/console", tags=["console"], include_in_schema=False)

ASSETS: Final = Path(__file__).resolve().parent / "assets"

# ─────────────────────────────────────────────────────────────────────────────
# The page
# ─────────────────────────────────────────────────────────────────────────────


@router.get("", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(content=(ASSETS / "index.html").read_text())


@router.get("/assets/{name}")
def asset(name: str) -> FileResponse:
    """Serve one of this package's own files.

    Resolved and then checked against the assets directory, so a name that
    climbed out of it — ``../../../etc/passwd`` — is a 404 rather than a read.
    The product has no static-file surface at all; adding one to the demo is not
    a reason to add a traversal with it.
    """
    path = (ASSETS / name).resolve()
    if not path.is_file() or ASSETS.resolve() not in path.parents:
        raise NotFound("no such asset", name=name)
    return FileResponse(path)


# ─────────────────────────────────────────────────────────────────────────────
# The requirement list
# ─────────────────────────────────────────────────────────────────────────────


class RequirementModel(BaseModel):
    name: str
    asks_for: str
    evidence: list[str]


class RequirementsResponse(BaseModel):
    requirements: list[RequirementModel]
    env: str
    ref: str


@router.get("/requirements")
def requirements() -> RequirementsResponse:
    """The twelve the page is organised by.

    Read from ``src.verify.REQUIREMENTS`` rather than restated here. That
    tuple is what ``ledger verify-requirements`` checks against the test suite,
    so the cards on the page and the traceability check cannot disagree about
    what the system claims to do — and adding a thirteenth requirement puts a
    thirteenth card on the page with no edit to the portal at all.
    """
    return RequirementsResponse(
        requirements=[
            RequirementModel(
                name=requirement.name,
                asks_for=requirement.asks_for,
                evidence=list(requirement.evidence),
            )
            for requirement in REQUIREMENTS
        ],
        env=DEMO_ENV,
        ref=MAIN,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The name distribution
# ─────────────────────────────────────────────────────────────────────────────


#: Buckets in the name distribution. Four bits of the digest — the first hex
#: character — which is the same prefix partitioning the collector shards on,
#: so the picture is of a real mechanism rather than one invented for a chart.
#: Sixteen reads well at demo scale; 256 (the on-disk fanout) is one-to-eleven
#: objects a bucket and looks like noise.
DISTRIBUTION_BITS: Final = 4
DISTRIBUTION_BUCKETS: Final = 1 << DISTRIBUTION_BITS


class NameBucket(BaseModel):
    #: The leading hex character these objects' names share.
    prefix: str
    #: Object counts by kind, keyed by the lowercase kind name.
    kinds: dict[str, int]
    total: int


class DistributionResponse(BaseModel):
    buckets: list[NameBucket]
    objects: int


@router.get("/distribution")
def distribution(demo: DemoDep) -> DistributionResponse:
    """How object names spread, and what the corpus is made of.

    An object's name is the hash of its bytes, so names are uniform by
    construction and every bucket holds about the same count. That is not a
    decoration: it is the property that lets the store shard by prefix at all,
    and this is the same ``iter_shard`` the collector walks.

    Counted from the catalog rather than from the object store: the catalog
    already holds a kind and a size per live object, clustered by digest, so
    each bucket is a bounded range scan of an index that is already in hash
    order. Walking the directories instead would be an ``rglob`` per request.
    """
    buckets: list[NameBucket] = []
    objects = 0
    for shard in range(DISTRIBUTION_BUCKETS):
        kinds: dict[str, int] = {}
        total = 0
        for entry in demo.ledger.store.catalog.iter_shard(DISTRIBUTION_BITS, shard):
            kinds[ObjectKind(entry.kind).name.lower()] = (
                kinds.get(ObjectKind(entry.kind).name.lower(), 0) + 1
            )
            total += 1
        buckets.append(NameBucket(prefix=f"{shard:x}", kinds=kinds, total=total))
        objects += total
    return DistributionResponse(buckets=buckets, objects=objects)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus totals
# ─────────────────────────────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    objects: int
    stored_bytes: int
    environments: int
    tombstones: int
    build_queue_depth: int
    #: The server's own count of objects fetched from the store, ever. Read
    #: either side of a request, the difference is how many objects that request
    #: cost — which is how the page measures a ranged read without the client
    #: having to believe anything.
    object_reads: int
    now_us: int
    format_fingerprint: str


@router.get("/stats")
def stats(demo: DemoDep) -> StatsResponse:
    """The corpus, in numbers a chart can use.

    Composed from ``src.metrics.snapshot`` rather than measured again here —
    the page and ``/v1/metrics`` must not be able to disagree about how large
    the corpus is.
    """
    taken = snapshot(demo.ledger)
    return StatsResponse(
        objects=int(_metric(taken, "ledger_corpus_objects")),
        stored_bytes=int(_metric(taken, "ledger_corpus_stored_bytes")),
        environments=int(_metric(taken, "ledger_environments")),
        tombstones=int(_metric(taken, "ledger_tombstones")),
        build_queue_depth=int(_metric(taken, "ledger_build_queue_depth")),
        object_reads=int(_metric(taken, "ledger_object_reads_total")),
        now_us=demo.clock.now_us(),
        format_fingerprint=FORMAT_FINGERPRINT,
    )


def _metric(taken: Snapshot, name: str) -> float:
    metric = taken.named(name)
    return metric.value if metric else 0.0
