"""The jj seam, from Python.

Ten routes, 385 lines, and until now its only verification was ``tests/roundtrip.rs``
— behind a ``slow`` mark, behind a Rust toolchain, behind a running server. That
is a lot of gates in front of the surface a whole client surface depends on, and
it meant a change here was untested on any machine without cargo.

These exercise the same mappings the Rust suite does, in-process. The Rust tests
still earn their place: they prove *jj itself* accepts what this returns, which
no amount of Python can. This proves the endpoint behaves before it gets there.
"""

from __future__ import annotations

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

ENV = "proximal/demo"
JJ = f"/v1/envs/{ENV}/jj"

#: jj hands its backend one signature per role, each with a name, an email and a
#: timezone offset. A Ledger commit has one author string and one timestamp, so
#: the rest rides in metadata — this is the shape that has to survive that.
SIGNATURE = {
    "name": "Agent Seventeen",
    "email": "a17@example.invalid",
    "timestamp_millis": 1_700_000_000_000,
    "tz_offset": 60,
}


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


@pytest.fixture
def auth(state: AppState) -> dict[str, str]:
    scope = Scope(
        operations=Operation.READ | Operation.WRITE | Operation.CREATE,
        selectors=(NamePrefixSelector("proximal/*"),),
    )
    token = state.signer.mint(Principal("agent-17"), scope, ttl_us=3600 * 1_000_000)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def env(client: TestClient, auth: dict[str, str]) -> str:
    response = client.post("/v1/envs", json={"name": ENV}, headers=auth)
    assert response.status_code == 201, response.text
    return ENV


def write_file(client: TestClient, auth: dict[str, str], payload: bytes) -> str:
    response = client.post(
        f"{JJ}/files",
        content=payload,
        headers={**auth, "Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


class TestFilesAndSymlinks:
    def test_a_file_round_trips(self, client: TestClient, auth: dict[str, str], env: str) -> None:
        del env
        payload = b"the quick brown fox\n" * 100
        file_id = write_file(client, auth, payload)
        read = client.get(f"{JJ}/files/{file_id}", headers=auth)
        assert read.status_code == 200, read.text
        assert read.content == payload

    def test_identical_files_get_one_id(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """Deduplication seen from jj's side: the same bytes at two paths are one
        object, which is what makes a fork of a large environment free.
        """
        del env
        first = write_file(client, auth, b"identical")
        second = write_file(client, auth, b"identical")
        assert first == second

    def test_an_empty_file_round_trips(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        del env
        file_id = write_file(client, auth, b"")
        assert client.get(f"{JJ}/files/{file_id}", headers=auth).content == b""

    def test_a_symlink_round_trips(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """A symlink's target is content, stored as an ordinary blob — which is
        what keeps the object model at four types.
        """
        del env
        written = client.post(f"{JJ}/symlinks", json={"target": "../data/train.bin"}, headers=auth)
        assert written.status_code == 200, written.text
        read = client.get(f"{JJ}/symlinks/{written.json()['id']}", headers=auth)
        assert read.json()["target"] == "../data/train.bin"

    def test_a_symlink_without_a_target_is_refused(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        del env
        assert client.post(f"{JJ}/symlinks", json={}, headers=auth).status_code == 400


class TestTrees:
    def test_the_empty_tree_has_a_stable_id(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """jj asks for it once and compares against it, so it must not move."""
        del env
        first = client.get(f"{JJ}/empty-tree", headers=auth)
        assert first.status_code == 200, first.text
        assert first.json()["id"] == client.get(f"{JJ}/empty-tree", headers=auth).json()["id"]

    def test_a_tree_round_trips_with_its_modes(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """The executable bit is inside the hash, so an old commit that brought
        back a non-executable verifier would not be the same version.
        """
        del env
        script = write_file(client, auth, b"#!/bin/sh\necho hi\n")
        readme = write_file(client, auth, b"# demo\n")
        written = client.post(
            f"{JJ}/trees",
            json={
                "entries": [
                    {"name": "run.sh", "kind": "file", "id": script, "executable": True},
                    {"name": "README.md", "kind": "file", "id": readme, "executable": False},
                ]
            },
            headers=auth,
        )
        assert written.status_code == 200, written.text

        read = client.get(f"{JJ}/trees/{written.json()['id']}", headers=auth)
        assert read.status_code == 200, read.text
        by_name = {e["name"]: e for e in read.json()["entries"]}
        assert by_name["run.sh"]["executable"] is True
        assert by_name["README.md"]["executable"] is False
        assert by_name["run.sh"]["kind"] == "file"

    def test_a_duplicate_entry_name_is_refused(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """Two entries with one name would give the tree a second valid encoding,
        which is exactly what the canonical format exists to prevent.
        """
        del env
        blob = write_file(client, auth, b"x")
        response = client.post(
            f"{JJ}/trees",
            json={
                "entries": [
                    {"name": "a", "kind": "file", "id": blob},
                    {"name": "a", "kind": "file", "id": blob},
                ]
            },
            headers=auth,
        )
        assert response.status_code == 400, response.text

    def test_a_submodule_is_refused_rather_than_approximated(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        del env
        response = client.post(
            f"{JJ}/trees",
            json={"entries": [{"name": "vendor", "kind": "gitsubmodule", "id": "00" * 32}]},
            headers=auth,
        )
        assert response.status_code == 400, response.text


class TestCommits:
    def _commit(self, client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
        tree = client.post(f"{JJ}/trees", json={"entries": []}, headers=auth).json()["id"]
        response = client.post(
            f"{JJ}/commits",
            json={
                "root_tree": [tree],
                "parents": [],
                "author": SIGNATURE,
                "committer": SIGNATURE,
                "description": "a change jj made",
            },
            headers=auth,
        )
        assert response.status_code == 200, response.text
        return dict(response.json())

    def test_a_commit_round_trips_with_both_signatures(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """The lossy mapping, checked in the direction that matters.

        jj carries two full signatures with emails and timezone offsets; a Ledger
        commit carries one author string and one timestamp. The rest rides in
        ``metadata``, so a jj commit has to come back *exactly* as it went in.
        """
        del env
        written = self._commit(client, auth)
        read = client.get(f"{JJ}/commits/{written['id']}", headers=auth)
        assert read.status_code == 200, read.text
        body = read.json()
        assert body["author"] == SIGNATURE
        assert body["committer"] == SIGNATURE
        assert body["description"] == "a change jj made"

    def test_a_conflicted_tree_is_refused_with_its_reason(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """jj models a conflict as a tree with more than one term. Ledger has no
        such object, so this is refused rather than silently resolved.
        """
        del env
        tree = client.post(f"{JJ}/trees", json={"entries": []}, headers=auth).json()["id"]
        response = client.post(
            f"{JJ}/commits",
            json={
                "root_tree": [tree, tree],
                "parents": [],
                "author": SIGNATURE,
                "committer": SIGNATURE,
                "description": "conflicted",
            },
            headers=auth,
        )
        assert response.status_code == 400
        assert "conflict" in response.text.lower()

    def test_a_commit_with_no_tree_is_refused(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        del env
        response = client.post(
            f"{JJ}/commits",
            json={"root_tree": [], "author": SIGNATURE, "committer": SIGNATURE},
            headers=auth,
        )
        assert response.status_code == 400

    def test_info_reports_the_ids_jj_needs(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        """jj fixes its id widths at startup and compares against them forever."""
        del env
        response = client.get(f"{JJ}/info", headers=auth)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["commit_id_length"] == 32
        assert body["change_id_length"] == 16


class TestAuthorization:
    def test_reading_requires_a_token(self, client: TestClient, env: str) -> None:
        del env
        assert client.get(f"{JJ}/empty-tree").status_code == 401

    def test_writing_requires_more_than_read(
        self, client: TestClient, state: AppState, env: str
    ) -> None:
        del env
        scope = Scope(operations=Operation.READ, selectors=(NamePrefixSelector("proximal/*"),))
        token = state.signer.mint(Principal("reader"), scope, ttl_us=3600 * 1_000_000)
        response = client.post(
            f"{JJ}/files",
            content=b"x",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
        )
        assert response.status_code == 403, response.text

    def test_an_unknown_id_is_not_found(
        self, client: TestClient, auth: dict[str, str], env: str
    ) -> None:
        del env
        assert client.get(f"{JJ}/files/{'00' * 32}", headers=auth).status_code == 404
