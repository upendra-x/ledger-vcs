"""The signals that must never be silent.

Most of what is exposed here is ordinary reporting. Two things are not, and they
are the reason this file exists:

* an object that failed verification;
* a collection cycle that hit its circuit breaker.

Both mean a human has to look. Both are **counters**, because a gauge would read
zero on the next scrape and the event would pass unnoticed — which is the exact
failure mode "must never be silent" is warning about.

The rest of the metrics measure assumptions about *how environments are built*
rather than about Ledger. Ledger cannot hold those assumptions true; it can only
see them drifting, which is why each one carries the threshold to investigate at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from src.api.app import build_app
from src.clock import ManualClock
from src.errors import CorruptObject
from src.format.cdc import ChunkParams
from src.format.model import Chunk
from src.format.shape import ShapeParams
from src.ids import EnvName
from src.instance import Ledger
from src.metrics import COUNTERS, Counters, Metric, Snapshot, render, snapshot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(
        tmp_path / "ledger",
        clock=ManualClock(start_us=1_700_000_000_000_000),
        chunk_params=ChunkParams.for_average(4096),
        shape_params=ShapeParams(
            domain=b"ledger.tree.split.v1",
            period=32,
            min_entries=4,
            max_entries=64,
            max_node_bytes=16 * 1024,
        ),
        shard_count=4,
    ) as opened:
        yield opened


@pytest.fixture
def client(ledger: Ledger) -> Iterator[TestClient]:
    with TestClient(build_app(ledger=ledger, clock=ledger.clock)) as opened:
        yield opened


class TestTheSignalsThatMustNeverBeSilent:
    def test_a_verification_failure_is_counted(self, ledger: Ledger) -> None:
        """Damaged data is detected, and *recorded*.

        Raising is not enough on its own: an operator learns about corruption
        from a metric, not from a caller's stack trace.
        """
        before = COUNTERS.value("ledger_object_verification_failures_total")

        outcome = ledger.store.put_object(Chunk(b"content that will be damaged" * 20))
        path = ledger.root / "objects" / outcome.name.hex[:2] / outcome.name.hex[2:4]
        target = path / outcome.name.hex
        target.write_bytes(b"\x00" + b"something else entirely")

        with pytest.raises(CorruptObject):
            ledger.store.get(outcome.name)

        assert COUNTERS.value("ledger_object_verification_failures_total") == before + 1

    def test_it_is_a_counter_not_a_gauge(self, ledger: Ledger) -> None:
        """A gauge would read zero on the next scrape and lose the event."""
        counters = Counters()
        counters.increment("ledger_object_verification_failures_total")
        taken = snapshot(ledger, counters)
        metric = taken.named("ledger_object_verification_failures_total")

        assert metric is not None
        assert metric.kind == "counter"
        assert metric.value == 1
        # Reading it again does not reset it.
        assert _value(snapshot(ledger, counters), "ledger_object_verification_failures_total") == 1

    def test_the_circuit_breaker_is_counted(self, ledger: Ledger) -> None:
        """A breaker that trips and is only visible in a return value is a
        breaker nobody notices until the backlog does.
        """
        from dataclasses import replace

        from src.maintenance.gc import GarbageCollector, GcConfig

        for index in range(40):
            ledger.store.put_object(Chunk(f"object number {index}".encode() * 20))

        collector = GarbageCollector(
            ledger.store,
            ledger.keepsets,
            ledger.repo,
            clock=ledger.clock,
            digests=ledger.digests,
            config=replace(GcConfig(), min_corpus_objects=10, grace_us=0),
        )
        before = COUNTERS.value("ledger_gc_circuit_breaker_trips_total")
        plan = collector.plan()

        assert plan.aborted, "the fixture did not actually trip the breaker"
        assert COUNTERS.value("ledger_gc_circuit_breaker_trips_total") == before + 1


class TestTheDriftMetrics:
    def test_it_reports_what_the_corpus_costs(self, ledger: Ledger) -> None:
        ledger.repo.create_env(EnvName("proximal/demo"))
        ledger.store.put_object(Chunk(b"some content" * 100))

        taken = snapshot(ledger)
        assert _value(taken, "ledger_corpus_objects") >= 1
        assert _value(taken, "ledger_corpus_stored_bytes") > 0
        assert _value(taken, "ledger_environments") == 1

    def test_every_assumption_carries_its_threshold(self) -> None:
        """A threshold that lives only in a document is a threshold
        nobody applies at three in the morning.
        """
        expected = {
            "ledger_unique_bytes_per_environment",
            "ledger_upload_ratio",
            "ledger_reads_per_materialization",
        }
        taken = snapshot_for_thresholds()
        without = {
            metric.name
            for metric in taken
            if metric.name in expected and not metric.investigate_when
        }
        assert not without, f"assumptions with no threshold: {without}"

    def test_the_upload_ratio_is_deduplication_measured_on_the_wire(self, ledger: Ledger) -> None:
        counters = Counters()
        counters.increment("ledger_objects_offered_total", 100)
        counters.increment("ledger_objects_uploaded_total", 5)

        assert _value(snapshot(ledger, counters), "ledger_upload_ratio") == pytest.approx(0.05)

    def test_ratios_do_not_divide_by_zero(self, ledger: Ledger) -> None:
        """A fresh deployment has no denominators, and a metrics endpoint that
        raises is a metrics endpoint that gets removed.
        """
        taken = snapshot(ledger, Counters())
        assert _value(taken, "ledger_upload_ratio") == 0
        assert _value(taken, "ledger_reads_per_materialization") == 0


class TestTheEndpoint:
    def test_it_renders_prometheus_text(self, client: TestClient) -> None:
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "# TYPE ledger_corpus_objects gauge" in response.text
        assert "# TYPE ledger_object_verification_failures_total counter" in response.text

    def test_it_names_no_environment_principal_or_object(self, ledger: Ledger) -> None:
        """It is unauthenticated, so it must report only aggregates.

        A metric carrying an environment name would turn an operational endpoint
        into an enumeration oracle over the corpus.
        """
        ledger.repo.create_env(EnvName("proximal/secret-project"))
        outcome = ledger.store.put_object(Chunk(b"secret content" * 50))

        rendered = render(snapshot(ledger))
        assert "secret-project" not in rendered
        assert outcome.name.hex not in rendered

    def test_it_works_on_an_empty_deployment(self, client: TestClient) -> None:
        assert client.get("/v1/metrics").status_code == 200

    def test_a_byte_count_is_rendered_exactly(self) -> None:
        """Six significant digits is not enough for a byte count.

        ``{:g}`` renders 2 GiB as ``2.14748e+09`` — 3,648 bytes short — and every
        storage metric here is a byte count watched to decide whether
        the capacity plan's sizing still holds. The rounding starts at a million, which is the
        first scrape anybody would trust.
        """
        rendered = render(Snapshot(metrics=(Metric("ledger_bytes", 2**31, "bytes"),)))
        assert "ledger_bytes 2147483648\n" in rendered

    def test_a_counter_never_renders_lower_than_it_climbed(self) -> None:
        """The failure that rounding causes downstream, not just in the text.

        Prometheus reads a decrease in a counter as a reset and imputes the whole
        value as new. So a *rounded* counter can produce a spike in ``rate()``
        without anything having happened — on the two signals that must never
        be silent, which is where a false spike is worst.
        """
        climbing = [1_000_001, 1_000_002, 1_000_003, 1_000_004]
        rendered = [
            render(Snapshot(metrics=(Metric("ledger_x_total", v, "", kind="counter"),)))
            for v in climbing
        ]
        seen = [float(text.strip().split()[-1]) for text in rendered]
        assert seen == sorted(seen), f"a monotonic counter rendered out of order: {seen}"
        assert len(set(seen)) == len(climbing), "distinct counter values collapsed onto one"

    def test_a_ratio_keeps_enough_of_itself_to_be_read(self) -> None:
        """The other half: fixing precision must not turn ratios into integers."""
        rendered = render(Snapshot(metrics=(Metric("ledger_ratio", 0.125, "ratio"),)))
        assert "ledger_ratio 0.125\n" in rendered


def _value(taken: Snapshot, name: str) -> float:
    metric = taken.named(name)
    assert metric is not None, f"no metric called {name}"
    return metric.value


def snapshot_for_thresholds() -> tuple[Metric, ...]:
    """The metric definitions, without needing a populated corpus."""
    from pathlib import Path as _Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as scratch, Ledger(_Path(scratch)) as ledger:
        return snapshot(ledger).metrics
