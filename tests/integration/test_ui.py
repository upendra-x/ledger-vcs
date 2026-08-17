"""The read-only browser.

Two things here are worth more than the rest of the file put together.

**It cannot write.** A UI that could would be a second write path, and the whole
argument for one atomic point would have to be made twice. That is
asserted structurally, by walking the router, rather than trusted to review.

**It escapes everything.** Almost all of what these pages display is content
somebody uploaded — environment names, commit messages, file names — so a single
unescaped interpolation is a stored cross-site scripting hole reachable by
committing a file. ``TestEscaping`` commits a payload and checks it comes back
inert.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import build_app
from src.auth.model import NamePrefixSelector, Operation, Principal, Scope
from src.clock import ManualClock
from src.format.cdc import ChunkParams
from src.format.shape import ShapeParams
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.service.commits import CommitService

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.api.deps import AppState
    from src.ids import EnvId, ObjectName

ENV = "proximal/demo"
MAIN = RefName("refs/heads/main")


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
def source(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    (root / "task").mkdir(parents=True)
    (root / "README.md").write_text("# a demo environment\n")
    (root / "task" / "prompt.md").write_text("solve it\n")
    (root / "blob.bin").write_bytes(bytes(range(256)) * 4)
    return root


@pytest.fixture
def env_id(ledger: Ledger, source: Path) -> EnvId:
    env = ledger.repo.create_env(EnvName(ENV)).env_id
    CommitService(ledger).commit(env, MAIN, source, author="agent-17", message="first version")
    return env


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


def token_for(state: AppState, *, prefix: str = "proximal/*") -> str:
    scope = Scope(operations=Operation.READ, selectors=(NamePrefixSelector(prefix),))
    minted: str = state.signer.mint(Principal("reader"), scope, ttl_us=3600 * 1_000_000)
    return minted


def bearer(state: AppState, **kwargs: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token_for(state, **kwargs)}"}


def head_commit(ledger: Ledger, env_id: EnvId) -> ObjectName:
    return ledger.repo.get_ref(env_id, MAIN).target


class TestPages:
    def test_the_index_lists_environments(
        self, client: TestClient, state: AppState, env_id: EnvId
    ) -> None:
        del env_id
        response = client.get("/", headers=bearer(state))
        assert response.status_code == 200
        assert ENV in response.text
        assert f'href="/ui/{ENV}"' in response.text

    def test_the_environment_page_shows_refs_and_history(
        self, client: TestClient, state: AppState, env_id: EnvId
    ) -> None:
        del env_id
        response = client.get(f"/ui/{ENV}", headers=bearer(state))
        assert response.status_code == 200
        assert "refs/heads/main" in response.text
        assert "first version" in response.text

    def test_a_commit_browses_its_tree(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId
    ) -> None:
        commit = head_commit(ledger, env_id)
        response = client.get(f"/ui/{ENV}/commits/{commit}", headers=bearer(state))
        assert response.status_code == 200
        assert "README.md" in response.text
        assert "task/" in response.text

    def test_a_subdirectory_is_reachable(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId
    ) -> None:
        commit = head_commit(ledger, env_id)
        response = client.get(f"/ui/{ENV}/commits/{commit}/tree/task", headers=bearer(state))
        assert response.status_code == 200
        assert "prompt.md" in response.text

    def test_a_text_file_previews(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId
    ) -> None:
        commit = head_commit(ledger, env_id)
        response = client.get(
            f"/ui/{ENV}/commits/{commit}/file/task/prompt.md", headers=bearer(state)
        )
        assert response.status_code == 200
        assert "solve it" in response.text

    def test_a_binary_file_says_so_rather_than_rendering_mojibake(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId
    ) -> None:
        commit = head_commit(ledger, env_id)
        response = client.get(f"/ui/{ENV}/commits/{commit}/file/blob.bin", headers=bearer(state))
        assert response.status_code == 200
        assert "binary" in response.text

    def test_the_diff_against_the_parent_is_shown(
        self,
        client: TestClient,
        state: AppState,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
    ) -> None:
        (source / "task" / "prompt.md").write_text("solve it differently\n")
        second = CommitService(ledger).commit(
            env_id, MAIN, source, author="agent-17", message="second"
        )
        response = client.get(f"/ui/{ENV}/commits/{second.commit}", headers=bearer(state))
        assert response.status_code == 200
        assert "task/prompt.md" in response.text
        assert "Changed in this version" in response.text


class TestEscaping:
    """Content somebody uploaded must never become markup.

    Reachable by anyone who can commit, which is the point: this is not a
    theoretical injection, it is a file name.
    """

    PAYLOAD = "<script>alert(1)</script>"
    #: A path component cannot contain "/", so the filename case needs a
    #: payload without one. This is a real one.
    NAME_PAYLOAD = "<img src=x onerror=alert(1)>.txt"

    def test_a_file_named_like_a_script_tag_is_inert(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId, source: Path
    ) -> None:
        (source / self.NAME_PAYLOAD).write_text("harmless\n")
        result = CommitService(ledger).commit(
            env_id, MAIN, source, author="agent-17", message="added a payload"
        )
        response = client.get(f"/ui/{ENV}/commits/{result.commit}", headers=bearer(state))

        assert response.status_code == 200
        assert self.NAME_PAYLOAD not in response.text
        assert "&lt;img src=x onerror=alert(1)&gt;" in response.text

    def test_a_commit_message_is_inert(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId, source: Path
    ) -> None:
        del env_id
        result = CommitService(ledger).commit(
            ledger.repo.resolve_env_name(EnvName(ENV)),
            MAIN,
            source,
            author="agent-17",
            message=self.PAYLOAD,
        )
        del result
        response = client.get(f"/ui/{ENV}", headers=bearer(state))

        assert response.status_code == 200
        assert self.PAYLOAD not in response.text
        assert "&lt;script&gt;" in response.text

    def test_file_content_is_inert(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId, source: Path
    ) -> None:
        (source / "payload.html").write_text(self.PAYLOAD)
        result = CommitService(ledger).commit(
            env_id, MAIN, source, author="agent-17", message="html file"
        )
        response = client.get(
            f"/ui/{ENV}/commits/{result.commit}/file/payload.html", headers=bearer(state)
        )

        assert response.status_code == 200
        assert self.PAYLOAD not in response.text
        assert "&lt;script&gt;" in response.text


class TestAuthorization:
    def test_a_browser_is_asked_for_a_credential(self, client: TestClient, env_id: EnvId) -> None:
        """A 401 with ``WWW-Authenticate`` is what makes a browser prompt at all."""
        del env_id
        response = client.get(f"/ui/{ENV}")
        assert response.status_code == 401

    def test_basic_auth_carries_the_token_as_the_password(
        self, client: TestClient, state: AppState, env_id: EnvId
    ) -> None:
        """The only credential a browser can be asked for without inventing a
        login page and a session — both of which would be new server state.
        """
        del env_id
        credential = base64.b64encode(f"reader:{token_for(state)}".encode()).decode()
        response = client.get(f"/ui/{ENV}", headers={"Authorization": f"Basic {credential}"})
        assert response.status_code == 200

    def test_a_token_for_another_environment_is_refused(
        self, client: TestClient, state: AppState, env_id: EnvId
    ) -> None:
        del env_id
        response = client.get(f"/ui/{ENV}", headers=bearer(state, prefix="someone-else/*"))
        assert response.status_code == 403

    def test_a_commit_from_another_environment_is_refused(
        self, client: TestClient, state: AppState, ledger: Ledger, env_id: EnvId, source: Path
    ) -> None:
        """The commit-to-environment binding, at the UI surface too.

        Otherwise a hash observed anywhere plus a token for any environment reads
        that environment's content through this one.
        """
        other = ledger.repo.create_env(EnvName("proximal/other")).env_id
        elsewhere = CommitService(ledger).commit(
            other, MAIN, source, author="someone", message="not yours"
        )
        del env_id

        response = client.get(f"/ui/{ENV}/commits/{elsewhere.commit}", headers=bearer(state))
        assert response.status_code == 403


class TestItCannotWrite:
    def test_every_ui_route_is_a_read(self) -> None:
        """**Structural, not reviewed.**

        A UI that could write would be a second write path, and every argument
        about the single atomic point would have to be made twice. Adding a POST
        here fails this test rather than a code review.
        """
        from src.api.ui.routes import router

        offenders = [
            f"{sorted(methods)} {getattr(route, 'path', '')}"
            for route in router.routes
            if not (methods := set(getattr(route, "methods", set()))) <= {"GET", "HEAD"}
        ]
        assert not offenders, "the read-only browser has a write route:\n  " + "\n  ".join(
            offenders
        )

    def test_the_check_sees_the_real_routes(self) -> None:
        from src.api.ui.routes import router

        paths = {getattr(r, "path", "") for r in router.routes}
        assert len(paths) >= 4, f"only found {paths}"

    def test_every_environment_page_declares_authorization(self) -> None:
        from src.api.ui.routes import router

        unguarded = [
            getattr(route, "path", "")
            for route in router.routes
            if "{org}" in getattr(route, "path", "")
            and "dependency" not in _dependency_names(getattr(route, "dependant", None))
        ]
        assert not unguarded, f"UI routes with no authorization: {unguarded}"


def _dependency_names(dependant: Any, depth: int = 0) -> set[str]:
    if dependant is None or depth > 5:
        return set()
    names = {getattr(dependant.call, "__name__", "")}
    for sub in dependant.dependencies:
        names |= _dependency_names(sub, depth + 1)
    return names
