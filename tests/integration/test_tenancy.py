"""What a token must *not* be able to do.

Every test here asserts a refusal. That is deliberate: an authorization bug is
invisible from the permitted side — every legitimate request keeps working — so
the only tests that can catch one are the ones that try the illegitimate thing
and demand a "no".

Three holes are covered, and all three were real. Each existed because a check
was made in one place and skipped in another that reached the same data by a
different route, which is the shape almost every authorization bug has.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import build_app
from src.auth.model import NamePrefixSelector, Operation, Principal, Scope
from src.clock import ManualClock
from src.format.cdc import ChunkParams
from src.format.shape import ShapeParams
from src.instance import Ledger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.api.deps import AppState

OURS = "proximal/demo"
THEIRS = "rival/secret"


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
def app(ledger: Ledger) -> Any:
    return build_app(ledger=ledger, clock=ledger.clock)


@pytest.fixture
def state(app: Any) -> AppState:
    return app.state.ledger_state  # type: ignore[no-any-return]


@pytest.fixture
def client(app: Any) -> Iterator[TestClient]:
    with TestClient(app) as opened:
        yield opened


def token(state: AppState, prefix: str, *operations: Operation, principal: str = "agent") -> str:
    """A token holding ``operations`` over one namespace and nothing else."""
    combined = Operation(0)
    for operation in operations:
        combined |= operation
    return state.signer.mint(
        Principal(principal),
        Scope(operations=combined, selectors=(NamePrefixSelector(prefix),)),
        ttl_us=3600 * 1_000_000,
    )


def bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


def basic(value: str) -> dict[str, str]:
    """The browser's credential: the token as an HTTP Basic password."""
    encoded = base64.b64encode(f"ledger:{value}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


@pytest.fixture
def both_environments(client: TestClient, state: AppState) -> None:
    for name in (OURS, THEIRS):
        org = name.split("/", 1)[0]
        creator = token(state, f"{org}/*", Operation.CREATE)
        response = client.post("/v1/envs", json={"name": name}, headers=bearer(creator))
        assert response.status_code == 201, response.text


class TestTheIndexIsNotAnOracle:
    """The browser's front page used to list the whole corpus.

    Every *other* page resolved its name through the single gate and refused
    honestly, so the per-environment checks all looked correct. The index reached
    the same names by a different route — a global scan — and printed them to
    anybody holding any valid token at all.
    """

    def test_it_lists_only_what_the_token_can_read(
        self, client: TestClient, state: AppState, both_environments: None
    ) -> None:
        del both_environments
        ours = token(state, "proximal/*", Operation.READ)
        body = client.get("/", headers=basic(ours)).text
        assert OURS in body
        assert THEIRS not in body, "the index leaked an environment this token cannot read"

    def test_a_token_for_neither_sees_nothing(
        self, client: TestClient, state: AppState, both_environments: None
    ) -> None:
        del both_environments
        outsider = token(state, "nobody/*", Operation.READ)
        body = client.get("/", headers=basic(outsider)).text
        assert OURS not in body
        assert THEIRS not in body

    def test_the_page_still_works_for_what_is_permitted(
        self, client: TestClient, state: AppState, both_environments: None
    ) -> None:
        """The refusal must not be achieved by breaking the feature."""
        del both_environments
        ours = token(state, "proximal/*", Operation.READ)
        response = client.get("/", headers=basic(ours))
        assert response.status_code == 200
        assert "1 environment(s)" in response.text


class TestCreateRespectsTheNamespace:
    """``env:create`` has no environment to select against, so it was checked as
    a bare operation bit — which let a token scoped to one namespace claim a name
    in another. Creation claims the name *globally*, so that is not just an
    overstep: it permanently denies the real owner their own namespace.
    """

    def test_a_token_cannot_create_outside_its_prefix(
        self, client: TestClient, state: AppState
    ) -> None:
        creator = token(state, "proximal/*", Operation.CREATE)
        response = client.post(
            "/v1/envs", json={"name": "someone-else/thing"}, headers=bearer(creator)
        )
        assert response.status_code == 403, response.text

    def test_it_can_still_create_inside_its_prefix(
        self, client: TestClient, state: AppState
    ) -> None:
        creator = token(state, "proximal/*", Operation.CREATE)
        response = client.post("/v1/envs", json={"name": OURS}, headers=bearer(creator))
        assert response.status_code == 201, response.text


class TestJjReadsAreBoundToTheirEnvironment:
    """The jj seam reads objects by bare id.

    Under global deduplication an object name is a corpus-wide address, so
    ``env:read`` on one environment plus a known hash was enough to read any
    object in the corpus. The ordinary API binds a commit to an environment by
    walking ancestry; a bare file id has nothing to walk from, so the binding has
    to come from the environment's keep-set instead.
    """

    def test_another_environments_object_is_not_served_by_knowing_its_id(
        self, client: TestClient, state: AppState, both_environments: None
    ) -> None:
        del both_environments
        writer = token(state, "rival/*", Operation.READ, Operation.WRITE, principal="rival-bot")
        written = client.post(
            f"/v1/envs/{THEIRS}/jj/files",
            content=b"the other team's private dataset",
            headers={**bearer(writer), "Content-Type": "application/octet-stream"},
        )
        assert written.status_code == 200, written.text
        file_id = written.json()["id"]

        # The rival can read back what it just wrote.
        assert (
            client.get(f"/v1/envs/{THEIRS}/jj/files/{file_id}", headers=bearer(writer)).status_code
            == 200
        )

        # We hold read *and* write on our own namespace, and we know the hash.
        ours = token(state, "proximal/*", Operation.READ, Operation.WRITE)
        stolen = client.get(f"/v1/envs/{OURS}/jj/files/{file_id}", headers=bearer(ours))
        assert stolen.status_code == 404, (
            f"an object from {THEIRS} was served through {OURS} by its hash: {stolen.text}"
        )

    def test_a_refusal_is_spelled_like_an_absence(
        self, client: TestClient, state: AppState, both_environments: None
    ) -> None:
        """404 and not 403, and identical to a hash that names nothing.

        Two spellings of "no" are an oracle: a caller could otherwise sort hashes
        into *exists elsewhere* and *does not exist* by reading status codes.
        """
        del both_environments
        ours = token(state, "proximal/*", Operation.READ, Operation.WRITE)
        writer = token(state, "rival/*", Operation.READ, Operation.WRITE, principal="rival-bot")
        real = client.post(
            f"/v1/envs/{THEIRS}/jj/files",
            content=b"private",
            headers={**bearer(writer), "Content-Type": "application/octet-stream"},
        ).json()["id"]
        absent = "00" * 32

        elsewhere = client.get(f"/v1/envs/{OURS}/jj/files/{real}", headers=bearer(ours))
        nowhere = client.get(f"/v1/envs/{OURS}/jj/files/{absent}", headers=bearer(ours))
        assert elsewhere.status_code == nowhere.status_code == 404
        assert elsewhere.json()["code"] == nowhere.json()["code"]
        assert elsewhere.json()["message"] == nowhere.json()["message"]

    def test_a_writer_can_read_back_what_it_just_wrote(
        self, client: TestClient, state: AppState, both_environments: None
    ) -> None:
        """The gate must not break the round trip it protects.

        jj writes objects one request at a time and reads them back while
        assembling a commit, long before any ref moves — so "reachable from a
        ref" alone would refuse a client its own in-flight content.
        """
        del both_environments
        ours = token(state, "proximal/*", Operation.READ, Operation.WRITE)
        payload = b"a file jj is about to put in a tree"
        file_id = client.post(
            f"/v1/envs/{OURS}/jj/files",
            content=payload,
            headers={**bearer(ours), "Content-Type": "application/octet-stream"},
        ).json()["id"]

        read_back = client.get(f"/v1/envs/{OURS}/jj/files/{file_id}", headers=bearer(ours))
        assert read_back.status_code == 200, read_back.text
        assert read_back.content == payload


class TestTheReachabilityMemoIsBounded:
    """The commit-to-environment check memoizes every commit an ancestry walk
    passes. On a long-running API plane that set grows with *traffic* rather than
    with anything that has a natural ceiling, so it has to evict.
    """

    def test_it_evicts_rather_than_growing_without_limit(self) -> None:
        from src.auth.policy import MAX_MEMOIZED_ANSWERS, ReachabilityChecker

        checker = ReachabilityChecker.__new__(ReachabilityChecker)
        checker._yes = OrderedDict()

        for index in range(MAX_MEMOIZED_ANSWERS + 500):
            checker._remember("env", f"commit-{index}")

        assert len(checker._yes) == MAX_MEMOIZED_ANSWERS

    def test_it_evicts_the_least_recently_used(self) -> None:
        """A commit asked about constantly must not be evicted by a burst of
        one-off lookups — that is the access pattern this exists for.
        """
        from src.auth.policy import MAX_MEMOIZED_ANSWERS, ReachabilityChecker

        checker = ReachabilityChecker.__new__(ReachabilityChecker)
        checker._yes = OrderedDict()

        checker._remember("env", "hot")
        for index in range(MAX_MEMOIZED_ANSWERS - 1):
            checker._remember("env", f"cold-{index}")
            checker._remembered("env", "hot")

        checker._remember("env", "one-more")
        assert checker._remembered("env", "hot")
        assert not checker._remembered("env", "cold-0")
