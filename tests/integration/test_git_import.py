"""Converting a git repository into Ledger.

GitHub import comes first, and a claim rests on it: the rate-limit pain
disappears before Ledger owns a single byte of truth. A system that can only
hold environments created inside it is one nobody can move to, so this is the
door.

The test that matters most is ``test_the_imported_tree_matches_git_exactly``:
converting history is only worth anything if the result *is* the history. The
rest defend the two decisions the importer makes on purpose — re-chunking rather
than copying git's representation, and refusing loudly what it cannot convert
faithfully.
"""

from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.errors import InvalidRequest
from src.format.cdc import ChunkParams
from src.format.model import Commit
from src.format.shape import ShapeParams
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.migrate.from_git import GIT_SHA1, GitImporter, Unsupported

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.ids import EnvId

MAIN = RefName("refs/heads/main")
ENV = "proximal/imported"

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Agent Seventeen",
            "GIT_AUTHOR_EMAIL": "agent@example.invalid",
            "GIT_COMMITTER_NAME": "Agent Seventeen",
            "GIT_COMMITTER_EMAIL": "agent@example.invalid",
            "GIT_AUTHOR_DATE": "2024-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2024-01-01T00:00:00Z",
        },
    )
    return result.stdout


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
def env_id(ledger: Ledger) -> EnvId:
    return ledger.repo.create_env(EnvName(ENV)).env_id


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small repository with the shapes that matter: modes, symlinks, history."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "--quiet", "--initial-branch=main")

    (root / "README.md").write_text("# a repository\n")
    (root / "task").mkdir()
    (root / "task" / "prompt.md").write_text("solve it\n")
    (root / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (root / "run.sh").chmod(0o755)
    (root / "latest").symlink_to("README.md")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "first version")

    (root / "task" / "prompt.md").write_text("solve it carefully\n")
    (root / "data.bin").write_bytes(bytes(range(256)) * 300)
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "second version")

    (root / "task" / "verifier.py").write_text("def verify(x):\n    return True\n")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "third version")
    return root


@pytest.fixture
def importer(ledger: Ledger) -> GitImporter:
    return GitImporter(ledger)


@requires_git
class TestImporting:
    def test_the_history_arrives(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        report = importer.import_repository(repo, env_id, MAIN)

        assert report.commits == 3
        assert report.head is not None
        assert ledger.repo.get_ref(env_id, MAIN).target == report.head

        from src.service.commits import CommitService

        messages = [
            entry.commit.message.strip() for entry in CommitService(ledger).log(env_id, MAIN)
        ]
        assert messages == ["third version", "second version", "first version"]

    def test_the_imported_tree_matches_git_exactly(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path, tmp_path: Path
    ) -> None:
        """**The claim this whole module rests on.**

        Converting history is worth nothing unless the result *is* the history.
        Materializing the imported head must reproduce a git checkout byte for
        byte — including the executable bit and symlinks, both of which are
        inside Ledger's hash and would otherwise be silently dropped.
        """
        report = importer.import_repository(repo, env_id, MAIN)
        assert report.head is not None

        checkout = tmp_path / "from-ledger"
        tree = ledger.store.get_as(report.head, Commit).tree
        ledger.materializer.materialize_tree(tree, checkout)

        differences = _compare(repo, checkout)
        assert not differences, "the imported tree differs from git:\n  " + "\n  ".join(differences)

    def test_modes_and_symlinks_survive(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path, tmp_path: Path
    ) -> None:
        report = importer.import_repository(repo, env_id, MAIN)
        assert report.head is not None
        checkout = tmp_path / "checked-out"
        ledger.materializer.materialize_tree(
            ledger.store.get_as(report.head, Commit).tree, checkout
        )

        assert os.access(checkout / "run.sh", os.X_OK), "the executable bit was lost"
        assert (checkout / "latest").is_symlink()
        assert os.readlink(checkout / "latest") == "README.md"

    def test_the_original_sha_is_recorded(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        """Every imported commit says where it came from, so a migration can be
        audited against the repository it replaced.
        """
        report = importer.import_repository(repo, env_id, MAIN)
        assert report.head is not None

        commit = ledger.store.get_as(report.head, Commit)
        recorded = dict(commit.metadata)["git.sha"]
        assert recorded == git(repo, "rev-parse", "HEAD").strip()

    def test_the_author_survives(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        report = importer.import_repository(repo, env_id, MAIN)
        assert report.head is not None
        commit = ledger.store.get_as(report.head, Commit)
        assert commit.author == "Agent Seventeen <agent@example.invalid>"


@requires_git
class TestItConvertsRatherThanCopies:
    def test_a_file_unchanged_across_commits_is_stored_once(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        """The reason to convert rather than copy git's representation.

        ``README.md`` appears in all three commits. Its content is seen three
        times and stored once — which is what ``git_bytes`` versus
        ``bytes_stored`` is measuring.
        """
        report = importer.import_repository(repo, env_id, MAIN)

        assert report.git_bytes > report.stats.bytes_stored
        assert report.dedup_ratio > 0, "importing history deduplicated nothing"

    def test_a_repeated_blob_is_never_read_twice(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        """A file surviving a hundred commits appears in a hundred trees.

        Reading it from git each time would make the import cost the history's
        logical size rather than its content's — the exact thing this system
        exists not to do. The reuse count is that saving, made visible.
        """
        report = importer.import_repository(repo, env_id, MAIN)
        assert report.reused > 0

    def test_re_importing_stores_nothing(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        """Resumable, not merely restartable.

        The ``git_sha → object name`` index means a re-run converts nothing it
        already converted — which is what lets a large migration be interrupted.
        """
        first = importer.import_repository(repo, env_id, MAIN)
        second = importer.import_repository(repo, env_id, MAIN)

        assert second.head == first.head
        assert second.stats.objects_created == 0, "a re-import stored something"

    def test_the_git_sha_is_findable_afterwards(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        """The same alternate-digest index the container registry uses. A git
        SHA-1 is another ecosystem's name for content Ledger holds.
        """
        importer.import_repository(repo, env_id, MAIN)
        sha = git(repo, "rev-parse", "HEAD").strip()

        found = ledger.digests.lookup(GIT_SHA1, sha)
        assert found is not None
        assert ledger.repo.get_ref(env_id, MAIN).target == found.name


@requires_git
class TestGitOrderingIsNotByteOrdering:
    def test_a_tree_is_re_sorted_into_ledgers_order(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, tmp_path: Path
    ) -> None:
        """The subtle one, and it would corrupt names rather than fail loudly.

        git orders tree entries as though every directory ended in ``/``, so
        ``foo`` (a directory) sorts *after* ``foo.txt`` in git and *before* it in
        plain byte order. Ledger's codec requires byte order — a tree built in
        git's order would either be rejected or, far worse, get a different name
        for identical content and quietly stop deduplicating.
        """
        repo = tmp_path / "ordering"
        repo.mkdir()
        git(repo, "init", "--quiet", "--initial-branch=main")
        (repo / "foo").mkdir()
        (repo / "foo" / "inner.txt").write_text("inner\n")
        (repo / "foo.txt").write_text("outer\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "ordering")

        # git really does order these the other way round.
        listed = [line.split("\t")[-1] for line in git(repo, "ls-tree", "HEAD").splitlines()]
        assert listed == ["foo.txt", "foo"], f"the fixture no longer reproduces it: {listed}"

        report = importer.import_repository(repo, env_id, MAIN)
        assert report.head is not None

        from src.fs.tree import iter_entries

        tree = ledger.store.get_as(report.head, Commit).tree
        names = [entry.name for entry in iter_entries(ledger.store, tree)]
        assert names == sorted(names), "the tree was written in git's order"
        assert names == [b"foo", b"foo.txt"]


@requires_git
class TestWhatItRefuses:
    """Content it cannot convert faithfully stops the import, by default.

    Importing an LFS pointer or a submodule as-is produces an environment that
    *looks* complete and is not — the worst outcome available here, because it
    fails later, elsewhere, to somebody else.
    """

    @staticmethod
    def _with_lfs_pointer(repo: Path) -> None:
        (repo / "model.bin").write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
            "size 12345\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "add a large model")

    def test_an_lfs_pointer_stops_the_import(
        self, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        self._with_lfs_pointer(repo)
        with pytest.raises(InvalidRequest, match=r"model\.bin"):
            importer.import_repository(repo, env_id, MAIN)

    def test_skipping_is_possible_but_must_be_asked_for(
        self, ledger: Ledger, env_id: EnvId, repo: Path
    ) -> None:
        self._with_lfs_pointer(repo)
        report = GitImporter(ledger, skip_unsupported=True).import_repository(repo, env_id, MAIN)

        assert not report.complete
        assert [p.path for p in report.unsupported] == ["model.bin"]
        assert report.unsupported[0].reason is Unsupported.LFS_POINTER
        # And the rest of the history did arrive.
        assert report.commits == 4

    def test_a_submodule_stops_the_import(
        self, importer: GitImporter, env_id: EnvId, repo: Path, tmp_path: Path
    ) -> None:
        """A submodule is a commit id in another repository — a pointer to
        content held elsewhere, like a foreign container layer.
        """
        other = tmp_path / "other"
        other.mkdir()
        git(other, "init", "--quiet", "--initial-branch=main")
        (other / "file.txt").write_text("elsewhere\n")
        git(other, "add", "-A")
        git(other, "commit", "--quiet", "-m", "in another repo")

        git(
            repo,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "--quiet",
            str(other),
            "vendor",
        )
        git(repo, "commit", "--quiet", "-m", "add a submodule")

        with pytest.raises(InvalidRequest, match="vendor"):
            importer.import_repository(repo, env_id, MAIN)


@requires_git
class TestScope:
    def test_only_the_newest_commits_can_be_taken(
        self, ledger: Ledger, importer: GitImporter, env_id: EnvId, repo: Path
    ) -> None:
        """A ten-year history is not always what somebody wants to move."""
        report = importer.import_repository(repo, env_id, MAIN, limit=2)

        assert report.commits == 2
        from src.service.commits import CommitService

        messages = [e.commit.message.strip() for e in CommitService(ledger).log(env_id, MAIN)]
        assert messages == ["third version", "second version"]

    def test_an_unknown_ref_says_so(self, importer: GitImporter, env_id: EnvId, repo: Path) -> None:
        with pytest.raises(InvalidRequest):
            importer.import_repository(repo, env_id, MAIN, source_ref="refs/heads/nonexistent")


def _compare(left: Path, right: Path) -> list[str]:
    """Every difference between two trees, ignoring git's own directory."""
    differences: list[str] = []
    for base, directories, files in os.walk(left):
        directories[:] = [d for d in directories if d != ".git"]
        for name in files:
            source = Path(base) / name
            relative = source.relative_to(left)
            target = right / relative
            if not target.exists() and not target.is_symlink():
                differences.append(f"missing: {relative}")
            elif source.is_symlink() or target.is_symlink():
                if os.readlink(source) != os.readlink(target):
                    differences.append(f"symlink differs: {relative}")
            elif not filecmp.cmp(source, target, shallow=False):
                differences.append(f"content differs: {relative}")
            elif os.access(source, os.X_OK) != os.access(target, os.X_OK):
                differences.append(f"mode differs: {relative}")
    return differences
