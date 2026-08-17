"""Object names must not depend on anything about the process that computed them.

Same bytes → same name, everywhere, forever. *Everywhere*
includes a different interpreter run with a different hash seed, a different
iteration order, and a different locale — and the failure mode if it does not
hold is the worst kind: a client and a server compute different names for
identical content, `HasObjects` reports everything missing, and deduplication
silently stops working while every test in the suite stays green.

Golden vectors pin the names themselves, so a change to the encoding is caught
even if determinism within one process is preserved.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from src.format.codec import name_of
from src.format.constants import MODE_EXEC, MODE_REGULAR, EntryKind
from src.format.model import Blob, BlobEntry, Chunk, Commit, Tree, TreeEntry
from src.ids import ChangeId, ObjectName

_A = ObjectName(b"\xaa" * 32)
_B = ObjectName(b"\xbb" * 32)

# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN VECTORS
#
# One entry per structural case the encoding has to get right. These are the
# specification a second implementation is written against — the Rust jj backend
# has to reproduce them exactly or it will write objects the Python side cannot
# deduplicate against.
#
# A failure here means the encoding changed and every object already stored has
# been renamed. Do not update these to make the build green; see the note in
# ledger/format/constants.py.
# ─────────────────────────────────────────────────────────────────────────────
GOLDEN: dict[str, tuple[object, str]] = {
    "chunk-single-byte": (
        Chunk(b"\x00"),
        "b3:44c659d145ea5700686e4b14564a7ab07c3a2e665b0e602827c8a1af3e134ee3",
    ),
    "chunk-text": (
        Chunk(b"hello world"),
        "b3:54e6629900c62aa2499fa7a51f40a69b9cb36420dece60c1e6a21824aba980cf",
    ),
    "chunk-all-byte-values": (
        Chunk(bytes(range(256))),
        "b3:14417625898094e6e20cb60c6c2f54893434c686995ae4b9824e3ea5860ac6be",
    ),
    "blob-empty-file": (
        Blob(level=0, entries=()),
        "b3:353c369527e5d3a0dc7949ab42cd02bb0982f3c78803b37c2d2faa4ac7684c76",
    ),
    "blob-two-chunks": (
        Blob(level=0, entries=(BlobEntry(_A, 11), BlobEntry(_B, 22))),
        "b3:35a6aea3de8836d6f5cae2189af88727a6ed3ec6eb6b864a2983acfe27606268",
    ),
    "blob-index-node": (
        Blob(level=1, entries=(BlobEntry(_A, 1000), BlobEntry(_B, 2000))),
        "b3:e913f2cedc538c5c4c06e4723a511d7e4a284a68d8f80b040f74ab58a1c1c285",
    ),
    "tree-empty": (
        Tree(level=0, entries=()),
        "b3:846b03a10b5d8176f316a9f445acfea47c7f480c64ccf1eb15843cf067cc1b64",
    ),
    "tree-mixed-kinds-and-modes": (
        Tree(
            level=0,
            entries=(
                TreeEntry(b"README.md", EntryKind.BLOB, _A, MODE_REGULAR, 11),
                TreeEntry(b"data", EntryKind.TREE, _B, 0, 0),
                TreeEntry(b"run.sh", EntryKind.BLOB, _A, MODE_EXEC, 42),
            ),
        ),
        "b3:abcf18a2ed98c15d1582ed10e51df9f3a01e894ba7d2795e7cb9d23e2f9e6fa5",
    ),
    "tree-interior-node": (
        Tree(
            level=1,
            entries=(
                TreeEntry(b"mmm", EntryKind.TREE, _A, 0, 0),
                TreeEntry(b"zzz", EntryKind.TREE, _B, 0, 0),
            ),
        ),
        "b3:a0f427ee110ad62d3a0d08a873ef3f95b14706551bdf480500aa6f71a2ff0d07",
    ),
    "commit-root": (
        Commit(
            tree=_A,
            parents=(),
            change_id=ChangeId("ab" * 16),
            author="agent-17",
            committer="agent-17",
            timestamp_us=1_700_000_000_000_000,
            message="add the verifier",
            metadata=(),
        ),
        "b3:9251beec725e601d72632a4b97601714a6e9e7cb219262f28e2cd7e4c79e0e4a",
    ),
    "commit-merge-with-metadata": (
        Commit(
            tree=_A,
            parents=(_A, _B),
            change_id=ChangeId("cd" * 16),
            author="agent-17",
            committer="ledger",
            timestamp_us=1_700_000_000_000_000,
            message="merge",
            metadata=(("git_sha", "deadbeef"), ("rollout", "r-8821")),
        ),
        "b3:05fee43b8b80363c3f64ca841464b5ee4223e4816a9d3b293c633fa451a82e2d",
    ),
}


@pytest.mark.parametrize(("case", "expected"), [(k, v[1]) for k, v in GOLDEN.items()])
def test_golden_object_names(case: str, expected: str) -> None:
    obj, _ = GOLDEN[case]
    assert str(name_of(obj)) == expected, (  # type: ignore[arg-type]
        f"the encoding of {case} changed — every object of this shape already "
        f"stored has been renamed"
    )


def test_golden_vectors_are_all_distinct() -> None:
    """Guards the guard: a copy-paste that duplicated a vector would make one of
    the structural cases untested.
    """
    names = [v[1] for v in GOLDEN.values()]
    assert len(set(names)) == len(names)


def test_names_are_stable_across_interpreter_runs() -> None:
    """The property that actually matters: a second process agrees.

    Run in subprocesses with different ``PYTHONHASHSEED`` values. If any part of
    the encoding ever depended on set or dict iteration order — a plausible way
    to write a "sort the metadata" step — this is what would catch it.
    """
    script = textwrap.dedent(
        """
        from src.format.codec import name_of
        from src.format.constants import EntryKind, MODE_REGULAR
        from src.format.model import Blob, BlobEntry, Chunk, Commit, Tree, TreeEntry
        from src.ids import ChangeId, ObjectName

        a = ObjectName(b"\\xaa" * 32)
        b = ObjectName(b"\\xbb" * 32)

        objects = [
            Chunk(b"hello world"),
            Blob(level=0, entries=(BlobEntry(a, 11), BlobEntry(b, 22))),
            Tree(level=0, entries=(
                TreeEntry(b"README.md", EntryKind.BLOB, a, MODE_REGULAR, 11),
                TreeEntry(b"data", EntryKind.TREE, b, 0, 0),
            )),
            Commit(
                tree=a, parents=(b,), change_id=ChangeId("ab" * 16),
                author="agent-17", committer="agent-17",
                timestamp_us=1700000000000000, message="add the verifier",
                metadata=(("git_sha", "deadbeef"), ("rollout", "r-8821")),
            ),
        ]
        print(" ".join(str(name_of(o)) for o in objects))
        """
    )
    results = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"PYTHONHASHSEED": str(seed)},
        ).stdout.strip()
        for seed in (0, 1, 7, 4242)
    }
    assert len(results) == 1, f"names differ across hash seeds: {results}"


def test_names_are_stable_within_a_process() -> None:
    chunk = Chunk(b"hello world")
    assert name_of(chunk) == name_of(Chunk(b"hello world"))


def test_metadata_order_does_not_depend_on_insertion_order() -> None:
    """Commit metadata must be sorted, and the sort must be by key alone.

    Building the same commit from differently-ordered pairs must either produce
    the same name or be rejected — never two different valid names.
    """
    from src.errors import NotCanonical

    ordered = Commit(
        tree=_A,
        parents=(),
        change_id=ChangeId("ab" * 16),
        author="a",
        committer="a",
        timestamp_us=0,
        message="",
        metadata=(("alpha", "1"), ("beta", "2")),
    )
    from dataclasses import replace

    try:
        shuffled_name = name_of(replace(ordered, metadata=(("beta", "2"), ("alpha", "1"))))
    except NotCanonical:
        return  # rejected outright, which is the stronger guarantee
    raise AssertionError(f"unsorted metadata was accepted and named {shuffled_name}")


def test_executable_bit_changes_the_name() -> None:
    """Mode is inside the hash, so a chmod is a new version — as it must be, or
    materializing an old commit would produce a non-executable verifier.
    """
    regular = Tree(level=0, entries=(TreeEntry(b"run.sh", EntryKind.BLOB, _A, MODE_REGULAR, 10),))
    executable = Tree(level=0, entries=(TreeEntry(b"run.sh", EntryKind.BLOB, _A, MODE_EXEC, 10),))
    assert name_of(regular) != name_of(executable)


def test_identical_content_in_unrelated_shapes_shares_leaf_names() -> None:
    """Cross-environment deduplication in miniature.

    Nothing in an object identifies which environment it belongs to, so the same
    file appearing in two unrelated trees is one blob, named once.
    """
    blob = Blob(level=0, entries=(BlobEntry(_A, 100),))
    tree_one = Tree(
        level=0,
        entries=(TreeEntry(b"shared.bin", EntryKind.BLOB, name_of(blob), MODE_REGULAR, 100),),
    )
    tree_two = Tree(
        level=0,
        entries=(
            TreeEntry(b"other.txt", EntryKind.BLOB, _B, MODE_REGULAR, 5),
            TreeEntry(b"shared.bin", EntryKind.BLOB, name_of(blob), MODE_REGULAR, 100),
        ),
    )
    shared = {e.target for e in tree_one.entries} & {e.target for e in tree_two.entries}
    assert shared == {name_of(blob)}
    assert name_of(tree_one) != name_of(tree_two)
