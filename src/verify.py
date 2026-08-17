"""Which requirements this build actually satisfies, checked at run time.

A traceability table in a document is a claim. This one is a *program*: each
requirement names the test that demonstrates it, and the check confirms that test
exists, is collected by the suite, and is not skipped. So a requirement stops
being satisfied the moment its evidence is deleted, renamed or marked ``skip`` —
which is exactly when a table in a document would silently start lying.

It deliberately does **not** re-run the tests. Re-running them here would make
the check a second, slower test suite that could disagree with the first; the
suite is the authority, and this reports on it.

*Format Agnostic* is met by the *absence* of code rather than by behaviour, so
it is demonstrated by a structural test — one that asserts no write route exists
— which is why it appears here alongside the rest.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

__all__ = ["REQUIREMENTS", "Requirement", "RequirementStatus", "verify"]

ROOT = Path(__file__).resolve().parent.parent


@final
@dataclass(frozen=True, slots=True)
class Requirement:
    """One requirement, and the evidence that it holds."""

    name: str
    #: The requirement's own words, so the check is against what was asked for
    #: rather than against somebody's summary of it.
    asks_for: str
    #: ``tests/path.py::TestClass::test_name`` for each piece of evidence.
    evidence: tuple[str, ...]


REQUIREMENTS: Final[tuple[Requirement, ...]] = (
    Requirement(
        name="Version Environments",
        asks_for="commit, list history, diff two versions, restore an earlier one",
        evidence=(
            "tests/integration/test_diff_fork.py::TestDiff::test_reports_added_removed_and_modified",
        ),
    ),
    Requirement(
        name="Branching / Forking",
        asks_for="branch for parallel experiments; fork into a variant",
        evidence=("tests/integration/test_diff_fork.py::TestFork::test_a_fork_copies_zero_bytes",),
    ),
    Requirement(
        name="Large Artifact Handling",
        asks_for="editing part of a large file must not re-store the whole file",
        evidence=(
            "tests/integration/test_ingest_costs.py::TestLargeArtifactOverwrite"
            "::test_mid_file_overwrite_costs_only_the_local_region",
            "tests/integration/test_ingest_costs.py::TestLargeArtifactOverwrite"
            "::test_the_cost_does_not_grow_with_the_file",
        ),
    ),
    Requirement(
        name="Multi-Container Support",
        asks_for="an old version must bring back exactly the containers it was committed with",
        evidence=(
            "tests/integration/test_oci.py::TestOldVersionsBringBackTheirImages"
            "::test_moving_the_ref_back_serves_the_earlier_image",
            "tests/integration/test_oci.py::TestIngest::test_layers_are_stored_uncompressed",
        ),
    ),
    Requirement(
        name="Read at scale",
        asks_for="read one file without cloning; no size limit, no cap on directory width",
        evidence=(
            "tests/integration/test_round_trip.py::TestReadPath"
            "::test_a_ranged_read_does_not_fetch_the_whole_file",
            "tests/integration/test_round_trip.py::TestListing::test_pages_a_wide_directory_with_a_cursor",
        ),
    ),
    Requirement(
        name="Scoped authorization",
        asks_for="subsets of operations over subsets of environments",
        evidence=(
            "tests/integration/test_api.py::TestAuthorization",
            "tests/integration/test_oci.py::TestRegistryAuthorization"
            "::test_a_layer_in_another_environment_is_not_served_by_knowing_its_hash",
        ),
    ),
    Requirement(
        name="Build & Sync",
        asks_for="trigger a build and sync into the platform, replacing the per-environment Action",
        evidence=(
            "tests/integration/test_build.py::TestBuild::test_commit_then_build_then_sync_note",
            "tests/integration/test_build.py::TestBuildIsAPureFunctionOfACommit"
            "::test_a_fork_inherits_its_parents_build_without_rebuilding",
        ),
    ),
    Requirement(
        name="Format Agnostic",
        asks_for="the storage layer must not know what the packaging format is",
        evidence=(
            "tests/integration/test_ui.py::TestItCannotWrite::test_every_ui_route_is_a_read",
        ),
    ),
    Requirement(
        name="Concurrent Automations",
        asks_for="work on one environment never waits on another; collisions give a clear error",
        evidence=(
            "tests/integration/test_metadata.py::TestPartitionIsolation",
            "tests/integration/test_metadata.py::TestRefCompareAndSwap"
            "::test_a_conflict_carries_enough_to_rebase",
            "tests/integration/test_build.py::TestOrderingAndExclusivity"
            "::test_two_environments_never_block_each_other",
        ),
    ),
    Requirement(
        name="Deduplication",
        asks_for="identical content stored once, whether or not the environments are related",
        evidence=(
            "tests/integration/test_ingest_costs.py::TestCrossEnvironmentSharing",
            "tests/integration/test_oci.py::TestIngest::test_a_rebuild_shares_its_base_layer",
            "tests/integration/test_git_import.py::TestItConvertsRatherThanCopies"
            "::test_a_file_unchanged_across_commits_is_stored_once",
            "tests/unit/test_compression.py::TestTheNamesAreUntouched"
            "::test_the_codec_does_not_change_any_name",
        ),
    ),
    Requirement(
        name="Stability",
        asks_for="a retry must not duplicate; damaged data detected, not served",
        evidence=(
            "tests/integration/test_metadata.py::TestIdempotency",
            "tests/unit/test_cas.py::TestDeliveryVerification",
            "tests/integration/test_gc.py::TestResurrection",
        ),
    ),
    Requirement(
        name="Native for Agents",
        asks_for="adopt jj's model rather than inventing vocabulary",
        evidence=(
            "tests/integration/test_jj.py::TestCommits"
            "::test_a_commit_round_trips_with_both_signatures",
            "tests/integration/test_jj.py::TestCommits::test_info_reports_the_ids_jj_needs",
            "tests/integration/test_metadata.py::TestOperationLog",
            "tests/integration/test_metadata.py::TestUndo",
        ),
    ),
)


@final
@dataclass(frozen=True, slots=True)
class RequirementStatus:
    requirement: Requirement
    present: tuple[str, ...]
    missing: tuple[str, ...]
    skipped: tuple[str, ...]

    @property
    def satisfied(self) -> bool:
        return not self.missing and not self.skipped and bool(self.present)


def verify(root: Path | None = None) -> list[RequirementStatus]:
    """Check every requirement's evidence exists and is not skipped."""
    base = root or ROOT
    index = _collect(base)
    statuses: list[RequirementStatus] = []
    for requirement in REQUIREMENTS:
        present, missing, skipped = [], [], []
        for node in requirement.evidence:
            state = index.get(node)
            if state is None:
                missing.append(node)
            elif state == "skipped":
                skipped.append(node)
            else:
                present.append(node)
        statuses.append(
            RequirementStatus(
                requirement=requirement,
                present=tuple(present),
                missing=tuple(missing),
                skipped=tuple(skipped),
            )
        )
    return statuses


def _collect(root: Path) -> dict[str, str]:
    """Every test node the suite defines, and whether it is skipped *here*.

    The structure is read from the source rather than by running pytest: running
    the suite to report on the suite would be a second, slower suite that could
    disagree with the first.

    The one thing source cannot answer is whether a **conditional** guard fires,
    so those are resolved separately — see ``_alias_states``.
    """
    index: dict[str, str] = {}
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken test file fails elsewhere
            continue
        aliases = _skipping_aliases(tree, path, root)
        module_skipped = _has_module_skip(tree, aliases)
        for node_id, skipped in _nodes(tree, relative, aliases):
            index[node_id] = "skipped" if (skipped or module_skipped) else "present"
    return index


def _skipping_aliases(tree: ast.Module, path: Path, root: Path) -> frozenset[str]:
    """Module-level guard names that are skipping **on this machine**.

    The suite spells its environment guards as ``requires_docker =
    pytest.mark.skipif(...)`` and applies them by name. That was invisible to a
    check looking for the substring ``skip`` in the decorator, because
    ``@requires_docker`` unparses to ``requires_docker`` and nothing else.

    It mattered more than a missed decorator usually would. Two requirements rest
    on evidence guarded this way — *Multi-Container Support* on a real ``docker
    pull``, *Native for Agents* on the Rust backend's round trip — so on a
    machine with neither Docker nor a Rust toolchain this reported 12/12 while
    two of those twelve had quietly not run. Turning *we do not know* into *we
    are fine* is the single failure this module exists to prevent.

    Treating every alias as a skip would trade that error for its mirror image:
    a machine that *does* have Docker would be told a requirement it genuinely
    demonstrates is unproven. So the condition is resolved rather than assumed.
    ``pytest.mark.skipif(condition, ...)`` evaluates ``condition`` when the
    module is imported, so by the time the mark object exists the answer is
    already a boolean sitting in ``mark.args[0]`` — importing the module and
    reading it is exact, and costs one ``shutil.which`` per guard.
    """
    candidates = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign) and "skip" in ast.unparse(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if not candidates:
        return frozenset()

    module = _import_test_module(path, root)
    if module is None:
        # Could not resolve it, so do not claim it passed. Conservative in the
        # only direction that is safe for a check whose job is honesty.
        return frozenset(candidates)

    skipping: set[str] = set()
    for name in candidates:
        mark = getattr(module, name, None)
        args = getattr(mark, "args", ())
        # `skipif(condition, reason=...)` → args[0]; a bare `skip` has no args
        # and always fires.
        if not args or bool(args[0]):
            skipping.add(name)
    return frozenset(skipping)


def _import_test_module(path: Path, root: Path) -> ModuleType | None:
    """Import a test module for its module-level constants, and nothing more.

    Importing runs module-level code — which is exactly the probes these guards
    are built from — and *not* fixtures, which pytest calls per test. Anything
    that fails to import is reported as unresolved rather than as fine.

    ``tests/`` and the project root go on the path for the duration, because that
    is what ``tests/conftest.py`` does for pytest and a module importing
    ``support.images`` cannot be read without it. Restored afterwards: a
    reporting command must not leave the interpreter's path rearranged.
    """
    spec = importlib.util.spec_from_file_location(f"_ledger_verify_{path.stem}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
        return None
    module = importlib.util.module_from_spec(spec)
    original = list(sys.path)
    sys.path[:0] = [str(root), str(root / "tests")]
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - a test module that cannot import
        return None
    finally:
        sys.path[:] = original
    return module


def _nodes(tree: ast.Module, relative: str, aliases: frozenset[str]) -> Iterator[tuple[str, bool]]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_skipped = _has_skip(node.decorator_list, aliases)
            yield f"{relative}::{node.name}", class_skipped
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name.startswith("test"):
                    yield (
                        f"{relative}::{node.name}::{member.name}",
                        class_skipped or _has_skip(member.decorator_list, aliases),
                    )
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            yield f"{relative}::{node.name}", _has_skip(node.decorator_list, aliases)


def _has_skip(decorators: Sequence[ast.expr], aliases: frozenset[str]) -> bool:
    return any(_mentions_skip(ast.unparse(decorator), aliases) for decorator in decorators)


def _mentions_skip(rendered: str, aliases: frozenset[str]) -> bool:
    """Whether a rendered decorator or mark is a skip, directly or by alias."""
    if "skip" in rendered:
        return True
    # `@requires_docker` renders bare; `@requires_docker(reason=...)` does not.
    return any(rendered == alias or rendered.startswith(f"{alias}(") for alias in aliases)


def _has_module_skip(tree: ast.Module, aliases: frozenset[str]) -> bool:
    """``pytestmark = pytest.mark.skip`` at module level, or an alias for one.

    ``slow`` is *not* a skip: those tests run, they are merely excluded from the
    inner loop. Counting them as missing would understate what this build does.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            rendered = ast.unparse(node.value)
            if _mentions_skip(rendered, aliases):
                return True
            # A list, as in `pytestmark = [pytest.mark.slow, requires_docker]`.
            if isinstance(node.value, ast.List):
                return any(
                    _mentions_skip(ast.unparse(element), aliases) for element in node.value.elts
                )
    return False
