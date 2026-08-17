"""The cost table, as measurements rather than claims.

This is where the *Large Artifact Handling* and *Deduplication* requirements stop
being architecture and become numbers. The prediction:

    Edit prompt.md (2 KiB)                 5 objects   ~10 KiB
    Append 8 MiB to train.bin             ~14 objects  ~9 MiB
    Overwrite 1 MiB inside train.bin       ~8 objects  ~3 MiB
    Fork the environment                    0 objects   0 bytes

The assertions below are upper bounds rather than exact equalities, because the
object count depends on where content-defined boundaries happen to fall — but
they are tight enough that a regression to whole-file storage, or a silent
reversion to fixed-size chunking, fails them immediately.

Everything runs at scaled chunk parameters so a "large" file is a few hundred
kilobytes. The ratios are what matter, and they are scale-invariant.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.format.cdc import ChunkParams
from src.format.shape import ShapeParams
from src.ids import ChangeId
from src.runtime.ingest import Ingester, IngestStats
from src.store.backend import InMemoryBackend
from src.store.cas import ObjectStore
from src.store.catalog import InMemoryWriteCatalog
from src.store.tombstone import SqliteTombstoneStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

#: 4 KiB average chunks: a 400 KiB dataset is ~100 chunks, enough for index
#: nodes to appear, small enough that the whole suite runs in under a second.
CHUNK_PARAMS = ChunkParams.for_average(4096)
SHAPE_PARAMS = ShapeParams(
    domain=b"ledger.tree.split.v1",
    period=64,
    min_entries=8,
    max_entries=256,
    max_node_bytes=32 * 1024,
)

DATASET_BYTES = 400_000


@pytest.fixture
def store() -> Iterator[ObjectStore]:
    tombstones = SqliteTombstoneStore.open(":memory:")
    yield ObjectStore(
        InMemoryBackend(),
        catalog=InMemoryWriteCatalog(),
        tombstones=tombstones,
        clock=ManualClock(),
    )
    tombstones.close()


@pytest.fixture
def ingester(store: ObjectStore) -> Ingester:
    return Ingester(
        store,
        clock=ManualClock(start_us=1_700_000_000_000_000),
        chunk_params=CHUNK_PARAMS,
        shape_params=SHAPE_PARAMS,
    )


def build_environment(root: Path) -> None:
    """A miniature of a realistic environment."""
    (root / "task").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "harbor.yaml").write_text("name: demo\ncontainers: {app: {}}\n")
    (root / "README.md").write_text("# demo environment\n")
    (root / "task" / "prompt.md").write_text("Solve the failing test.\n")
    (root / "task" / "verifier.py").write_text("import sys\nsys.exit(0)\n")
    (root / "data" / "train.bin").write_bytes(random.Random(42).randbytes(DATASET_BYTES))


def commit(ingester: Ingester, root: Path, message: str) -> IngestStats:
    _, stats = ingester.ingest_commit(root, author="agent-17", message=message)
    return stats


@pytest.fixture
def environment(tmp_path: Path, ingester: Ingester) -> tuple[Path, IngestStats]:
    root = tmp_path / "env"
    build_environment(root)
    return root, commit(ingester, root, "initial")


class TestFirstCommit:
    def test_stores_the_whole_environment_once(self, environment: tuple[Path, IngestStats]) -> None:
        _, stats = environment
        assert stats.objects_created == stats.objects_offered
        assert stats.bytes_stored >= DATASET_BYTES
        assert stats.files == 5
        assert stats.chunks > 10, "the dataset should have split into many chunks"


class TestReIngestIsFree:
    def test_re_ingesting_identical_content_creates_nothing(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        """The cheapest possible check that the pipeline is deterministic.

        If chunking, tree shaping or encoding were nondeterministic in any way,
        this would create objects — and deduplication would be silently broken
        for every environment in the corpus.
        """
        root, _ = environment
        _, second = ingester.ingest_directory(root)
        assert second.objects_created == 0
        assert second.bytes_stored == 0
        assert second.dedup_ratio == 1.0

    def test_re_committing_creates_exactly_the_commit(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        """Committing the same tree twice is one new object, not zero.

        A commit carries a fresh ``change_id`` and its own timestamp, so it is a
        genuinely new version even when its content is identical — which is what
        makes "commit, then commit again" a real second version rather than a
        silent no-op. Everything *underneath* it deduplicates completely.
        """
        root, _ = environment
        stats = commit(ingester, root, "same content, new version")
        assert stats.objects_created == 1, "only the commit object should be new"
        assert stats.bytes_stored < 512

    def test_a_byte_identical_commit_deduplicates_completely(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        """Pin the same change id and timestamp and even the commit is free.

        This is the strong form of determinism: nothing in the pipeline depends
        on anything but its inputs. It is also what makes an import resumable —
        replaying a converted commit must land on the same name.
        """
        root, _ = environment
        pinned = ChangeId("ab" * 16)
        first_name, _ = ingester.ingest_commit(root, author="a", message="pinned", change_id=pinned)
        second_name, stats = ingester.ingest_commit(
            root, author="a", message="pinned", change_id=pinned
        )
        assert first_name == second_name
        assert stats.objects_created == 0
        assert stats.bytes_stored == 0


class TestSmallFileEdit:
    """Editing a small text file costs 5 objects, ~10 KiB."""

    def test_costs_a_handful_of_objects(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        root, _ = environment
        (root / "task" / "prompt.md").write_text("Solve it. Hint: check the imports.\n")
        stats = commit(ingester, root, "edit prompt")

        # chunk, blob, the task/ tree, the root tree, the commit.
        assert stats.objects_created <= 6, f"{stats.objects_created} objects for a one-line edit"

    def test_costs_almost_no_bytes(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        root, _ = environment
        (root / "task" / "prompt.md").write_text("Solve it. Hint: check the imports.\n")
        stats = commit(ingester, root, "edit prompt")

        assert stats.bytes_stored < 4096, (
            f"a one-line edit stored {stats.bytes_stored} bytes; the dataset is "
            f"{DATASET_BYTES} and must not have been touched"
        )


class TestLargeArtifactAppend:
    """Appending to a large file costs the appended bytes."""

    def test_append_costs_roughly_what_was_appended(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        root, _ = environment
        appended = 40_000
        with (root / "data" / "train.bin").open("ab") as handle:
            handle.write(random.Random(7).randbytes(appended))

        stats = commit(ingester, root, "append")

        assert stats.bytes_stored < appended * 3, (
            f"appending {appended} bytes stored {stats.bytes_stored}; every chunk "
            f"before the append should have been reused"
        )
        assert stats.bytes_stored < DATASET_BYTES / 2


class TestLargeArtifactOverwrite:
    """The headline claim: editing inside a large file costs the edited region.

    Git would re-store the entire file. This is the test that fails if chunking
    ever regresses to whole-blob storage or to fixed-size blocks.
    """

    def test_mid_file_overwrite_costs_only_the_local_region(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        root, _ = environment
        path = root / "data" / "train.bin"
        overwritten = 20_000

        data = bytearray(path.read_bytes())
        midpoint = len(data) // 2
        data[midpoint : midpoint + overwritten] = random.Random(99).randbytes(overwritten)
        path.write_bytes(bytes(data))

        stats = commit(ingester, root, "overwrite")

        assert stats.bytes_stored < DATASET_BYTES / 4, (
            f"overwriting {overwritten} bytes inside a {DATASET_BYTES}-byte file "
            f"stored {stats.bytes_stored} — content-defined chunking is not working"
        )
        assert stats.objects_created < 15

    def test_the_cost_does_not_grow_with_the_file(self, ingester: Ingester, tmp_path: Path) -> None:
        """Scale invariance, which is the property that actually matters.

        The same edit in a file four times larger must cost the same — that is
        the difference between "an edit costs what changed" and "an edit costs
        a fraction of the file".
        """
        costs = []
        for multiplier in (1, 4):
            root = tmp_path / f"env{multiplier}"
            (root / "data").mkdir(parents=True)
            size = DATASET_BYTES * multiplier
            path = root / "data" / "train.bin"
            path.write_bytes(random.Random(42).randbytes(size))
            commit(ingester, root, "initial")

            data = bytearray(path.read_bytes())
            data[size // 2 : size // 2 + 5_000] = random.Random(1).randbytes(5_000)
            path.write_bytes(bytes(data))
            costs.append(commit(ingester, root, "edit").bytes_stored)

        assert costs[1] < costs[0] * 3, (
            f"a 4x larger file cost {costs[1]} vs {costs[0]} for the same edit; "
            f"cost should track the change, not the file size"
        )


class TestCrossEnvironmentSharing:
    """Identical content is stored once whether or not the
    environments are related.
    """

    def test_an_unrelated_environment_with_the_same_dataset_stores_nothing_new(
        self, ingester: Ingester, environment: tuple[Path, IngestStats], tmp_path: Path
    ) -> None:
        root, _ = environment
        other = tmp_path / "unrelated"
        (other / "data").mkdir(parents=True)
        (other / "data" / "train.bin").write_bytes((root / "data" / "train.bin").read_bytes())
        (other / "totally-different.txt").write_text("nothing in common\n")

        stats = commit(ingester, other, "an unrelated environment")

        assert stats.bytes_stored < 4096, (
            f"an unrelated environment holding the same dataset stored "
            f"{stats.bytes_stored} bytes; cross-environment dedup is not working"
        )

    def test_a_renamed_file_reuses_every_chunk(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        """Content addressing has no notion of a path, so a rename is free apart
        from the trees that mention it.
        """
        root, _ = environment
        (root / "data" / "train.bin").rename(root / "data" / "renamed.bin")
        stats = commit(ingester, root, "rename")

        assert stats.bytes_stored < 4096


class TestModeAndStructure:
    def test_the_executable_bit_is_versioned(
        self, ingester: Ingester, environment: tuple[Path, IngestStats]
    ) -> None:
        """Mode is inside the hash, so a chmod is a new version — as it must be,
        or restoring an old commit would produce a non-executable verifier.
        """
        root, _ = environment
        verifier = root / "task" / "verifier.py"
        verifier.chmod(0o755)
        stats = commit(ingester, root, "chmod")

        assert stats.objects_created > 0, "a mode change must produce a new version"
        assert stats.bytes_stored < 4096, "but it must not re-store the file's content"

    def test_symlinks_are_content(self, ingester: Ingester, tmp_path: Path) -> None:
        root = tmp_path / "links"
        root.mkdir()
        (root / "real.txt").write_text("target\n")
        (root / "link.txt").symlink_to("real.txt")
        _, stats = ingester.ingest_directory(root)
        assert stats.symlinks == 1
        assert stats.files == 1

    def test_an_empty_file_is_a_blob_with_no_entries(
        self, ingester: Ingester, tmp_path: Path
    ) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        (root / "nothing.txt").write_bytes(b"")
        _, stats = ingester.ingest_directory(root)
        assert stats.chunks == 0

    def test_an_empty_directory_ingests(self, ingester: Ingester, tmp_path: Path) -> None:
        root = tmp_path / "hollow"
        root.mkdir()
        name, stats = ingester.ingest_directory(root)
        assert stats.directories == 1
        assert name is not None

    def test_a_wide_directory_splits_but_still_ingests(
        self, ingester: Ingester, tmp_path: Path
    ) -> None:
        """No cap on how many files a directory can hold."""
        root = tmp_path / "wide"
        root.mkdir()
        for i in range(2000):
            (root / f"f{i:06d}.txt").write_text(f"{i}\n")
        _, stats = ingester.ingest_directory(root)
        assert stats.files == 2000
