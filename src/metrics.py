"""The numbers that say whether the assumptions still hold.

An unusual and correct point: *"Several numbers this design
rests on are properties of **how environments are built**, not of Ledger."*
Ledger cannot stop environments from drifting away from them — it is decided by
how images and datasets are produced — but it can see it coming.

So these are not service metrics. Latency and error rates say whether the system
is working; these say whether the *design* is still the right one. Each carries
the threshold for investigating, so a reader does not have to hold the
document open beside the dashboard.

The two that must never be silent are here as counters rather than gauges:
an object that failed verification, and a collection cycle that hit its circuit
breaker. Both mean a human has to look, and a gauge that returns to normal on the
next scrape would let them pass unnoticed.

Rendered in Prometheus text format because it is the format every collector
already reads, and inventing a second one is work that buys nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Final, final

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.instance import Ledger

__all__ = ["Counters", "Metric", "Snapshot", "render", "snapshot"]


@final
@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    value: float
    help_text: str
    kind: str = "gauge"
    #: The value to investigate at. Rendered as a comment beside the metric,
    #: because a threshold that lives only in a document is a threshold nobody
    #: applies.
    investigate_when: str = ""


@final
class Counters:
    """Events that must survive being observed.

    A verification failure that showed up as a gauge would read zero on the next
    scrape and nobody would ever know it happened. These only go up.
    """

    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._lock = Lock()
        self._values: dict[str, float] = {
            "ledger_object_verification_failures_total": 0.0,
            "ledger_gc_circuit_breaker_trips_total": 0.0,
            "ledger_objects_offered_total": 0.0,
            "ledger_objects_uploaded_total": 0.0,
            "ledger_object_reads_total": 0.0,
            "ledger_materializations_total": 0.0,
        }

    def increment(self, name: str, by: float = 1.0) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0.0) + by

    def value(self, name: str) -> float:
        with self._lock:
            return self._values.get(name, 0.0)

    def items(self) -> list[tuple[str, float]]:
        with self._lock:
            return sorted(self._values.items())


#: Process-wide, because these describe the corpus rather than a request.
COUNTERS: Final = Counters()

_COUNTER_HELP: Final = {
    "ledger_object_verification_failures_total": (
        "Objects whose bytes did not match the name they were stored under. "
        "One of the two signals that must never be silent."
    ),
    "ledger_gc_circuit_breaker_trips_total": (
        "Collection cycles aborted because they proposed to delete too much. "
        "The other signal that must never be silent."
    ),
    "ledger_objects_offered_total": "Objects clients offered during a write.",
    "ledger_objects_uploaded_total": "Objects clients actually had to upload.",
    "ledger_object_reads_total": "Objects fetched from the store.",
    "ledger_materializations_total": "Environments materialized.",
}


@final
@dataclass(frozen=True, slots=True)
class Snapshot:
    metrics: tuple[Metric, ...] = field(default_factory=tuple)

    def named(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)


def snapshot(ledger: Ledger, counters: Counters | None = None) -> Snapshot:
    """Read the corpus-shaped metrics, plus the counters."""
    source = counters or COUNTERS
    objects, stored_bytes = ledger.store.catalog.total()
    environments = len(ledger.repo.list_envs(limit=1_000_000))

    metrics: list[Metric] = [
        Metric(
            "ledger_corpus_objects",
            objects,
            "Objects the write catalog has ever recorded and not swept.",
        ),
        Metric(
            "ledger_corpus_stored_bytes",
            stored_bytes,
            "Bytes those objects occupy on the medium, after compression at rest.",
        ),
        Metric("ledger_environments", environments, "Environments that exist."),
        Metric(
            "ledger_unique_bytes_per_environment",
            stored_bytes / environments if environments else 0.0,
            "the capacity plan assumes ~2 GiB. Every storage number scales with it.",
            investigate_when="above ~3 GiB",
        ),
        Metric(
            "ledger_upload_ratio",
            _ratio(source, "ledger_objects_uploaded_total", "ledger_objects_offered_total"),
            "Objects uploaded ÷ objects offered. Deduplication, measured on the wire.",
            investigate_when="above ~20%",
        ),
        Metric(
            "ledger_reads_per_materialization",
            _ratio(source, "ledger_object_reads_total", "ledger_materializations_total"),
            "the capacity plan assumes ~25 object reads per materialization.",
            investigate_when="sustained above ~100",
        ),
        Metric(
            "ledger_tombstones",
            ledger.store.tombstones.count(),
            "Swept hashes still held back from deduplication.",
        ),
        Metric(
            "ledger_build_queue_depth",
            _queue_depth(ledger),
            "Builds waiting. The worker pool is sized on the commit rate.",
        ),
    ]
    metrics.extend(
        Metric(name, value, _COUNTER_HELP.get(name, ""), kind="counter")
        for name, value in source.items()
    )
    return Snapshot(metrics=tuple(metrics))


def render(snapshot_taken: Snapshot) -> str:
    """Prometheus text format."""
    return "".join(_lines(snapshot_taken))


def _lines(taken: Snapshot) -> Iterator[str]:
    for metric in taken.metrics:
        if metric.help_text:
            yield f"# HELP {metric.name} {metric.help_text}\n"
        yield f"# TYPE {metric.name} {metric.kind}\n"
        if metric.investigate_when:
            yield f"# investigate when {metric.investigate_when}\n"
        yield f"{metric.name} {_number(metric.value)}\n"


def _number(value: float) -> str:
    """Render a metric value without losing any of it.

    ``{:g}`` — the obvious choice, and the one this used to make — keeps six
    significant digits. That is invisible while a corpus is small and wrong the
    moment it is not: 3,001,478 bytes renders as ``3.00148e+06``, and 2 GiB
    renders 3,648 bytes short. Every storage number here is a byte count, and
    these decide whether the capacity plan's assumptions still hold, so a
    gauge that rounds is a gauge that answers the wrong question.

    Worse on the counters. Rounding is not monotonic, so a counter can render
    *lower* than its previous scrape while only ever having gone up. Prometheus
    reads a decrease as a counter reset and imputes the whole value as new —
    which turns a rounding artefact into a spike in ``rate()``, on exactly the
    two signals that must never be silent.

    Integers therefore print in full, and everything else in Python's shortest
    round-tripping form. Both are valid Prometheus values.
    """
    if value.is_integer():
        return str(int(value))
    return repr(value)


def _ratio(counters: Counters, numerator: str, denominator: str) -> float:
    bottom = counters.value(denominator)
    return counters.value(numerator) / bottom if bottom else 0.0


def _queue_depth(ledger: Ledger) -> int:
    from src.build.queue import BuildQueue

    return BuildQueue(ledger.meta, clock=ledger.clock).depth()
