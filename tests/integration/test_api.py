"""The HTTP surface: the error contract, and the authorization matrix.

Two things are asserted here that nothing else can assert:

* **the error contract carries state** — a 409 returns enough to rebase without
  a second round trip, 422 means exactly one thing, 429 carries ``Retry-After``,
  and a 403 is decided before any resolution work happens;
* **every route declares its authorization** — a missing check is invisible
  until someone exploits it, so the route table is walked and asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import build_app
from src.api.deps import AppState
from src.auth.model import NamePrefixSelector, Operation, Principal, Scope
from src.clock import ManualClock
from src.format.cdc import ChunkParams
from src.format.codec import encode, name_of
from src.format.constants import MODE_REGULAR, EntryKind
from src.format.model import Blob, BlobEntry, Chunk, Commit, Tree, TreeEntry
from src.format.shape import ShapeParams
from src.ids import ChangeId, ObjectName
from src.instance import Ledger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

ENV = "proximal/demo"


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(start_us=1_700_000_000_000_000)


@pytest.fixture
def ledger(tmp_path: Path, clock: ManualClock) -> Iterator[Ledger]:
    with Ledger(
        tmp_path / "ledger",
        clock=clock,
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


def token_for(state: AppState, *operations: Operation, principal: str = "agent-17") -> str:
    scope = Scope(operations=_union(operations), selectors=(NamePrefixSelector("proximal/*"),))
    return state.signer.mint(Principal(principal), scope, ttl_us=3600 * 1_000_000)


def _union(operations: tuple[Operation, ...]) -> Operation:
    result = Operation(0)
    for operation in operations:
        result |= operation
    return result


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin(state: AppState) -> dict[str, str]:
    return auth_header(
        token_for(
            state,
            Operation.READ,
            Operation.WRITE,
            Operation.CREATE,
            Operation.FORK,
            Operation.ANNOTATE,
            Operation.ADMIN,
        )
    )


@pytest.fixture
def env(client: TestClient, admin: dict[str, str]) -> str:
    response = client.post("/v1/envs", json={"name": ENV}, headers=admin)
    assert response.status_code == 201, response.text
    return ENV


def publish_commit(
    ledger: Ledger, message: str = "v1", parent: ObjectName | None = None
) -> ObjectName:
    """Write a tiny environment directly, so API tests need no upload dance."""
    chunk = Chunk(f"content for {message}".encode())
    ledger.store.put_object(chunk)
    blob = Blob(level=0, entries=(BlobEntry(name_of(chunk), chunk.size),))
    ledger.store.put_object(blob)
    tree = Tree(
        level=0,
        entries=(TreeEntry(b"README.md", EntryKind.BLOB, name_of(blob), MODE_REGULAR, chunk.size),),
    )
    ledger.store.put_object(tree)
    commit = Commit(
        tree=name_of(tree),
        parents=(parent,) if parent else (),
        change_id=ChangeId("ab" * 16),
        author="agent-17",
        committer="agent-17",
        timestamp_us=ledger.clock.now_us(),
        message=message,
    )
    return ledger.store.put_object(commit).name


class TestHealth:
    def test_reports_the_format_fingerprint(self, client: TestClient) -> None:
        """Two deployments share a corpus iff they share this value, so surfacing
        it turns "why is nothing deduplicating" into one comparison.
        """
        from src.format.constants import FORMAT_FINGERPRINT

        body = client.get("/v1/healthz").json()
        assert body["status"] == "ok"
        assert body["format_fingerprint"] == FORMAT_FINGERPRINT


class TestAuthentication:
    def test_a_missing_token_is_401(self, client: TestClient) -> None:
        assert client.get(f"/v1/envs/{ENV}").status_code == 401

    def test_a_malformed_header_is_401(self, client: TestClient) -> None:
        response = client.get(f"/v1/envs/{ENV}", headers={"Authorization": "Basic xyz"})
        assert response.status_code == 401

    def test_a_tampered_token_is_401(self, client: TestClient, state: AppState) -> None:
        token = token_for(state, Operation.READ)
        head, body, signature = token.split(".")
        forged = f"{head}.{body}.{signature[:-4]}AAAA"
        assert client.get(f"/v1/envs/{ENV}", headers=auth_header(forged)).status_code == 401

    def test_an_expired_token_is_401(
        self, client: TestClient, state: AppState, clock: ManualClock, env: str
    ) -> None:
        token = state.signer.mint(
            Principal("agent-17"),
            Scope(operations=Operation.READ, selectors=(NamePrefixSelector("proximal/*"),)),
            ttl_us=60 * 1_000_000,
        )
        assert client.get(f"/v1/envs/{env}", headers=auth_header(token)).status_code == 200
        clock.advance_seconds(61)
        assert client.get(f"/v1/envs/{env}", headers=auth_header(token)).status_code == 401

    def test_a_token_from_an_unknown_key_is_401(
        self, client: TestClient, clock: ManualClock
    ) -> None:
        from src.auth.tokens import TokenSigner

        rogue = TokenSigner.generate(key_id="rogue", clock=clock)
        token = rogue.mint(
            Principal("attacker"),
            Scope(operations=Operation.READ, selectors=(NamePrefixSelector("*"),)),
            ttl_us=3600 * 1_000_000,
        )
        assert client.get(f"/v1/envs/{ENV}", headers=auth_header(token)).status_code == 401


class TestAuthorization:
    """Design *Scoped Authorization*, as a negative matrix."""

    def test_read_only_cannot_write(
        self, client: TestClient, state: AppState, env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger)
        reader = auth_header(token_for(state, Operation.READ))

        response = client.put(
            f"/v1/envs/{env}/refs/refs/heads/main",
            json={"target": str(commit)},
            headers=reader,
        )
        assert response.status_code == 403
        assert response.json()["code"] == "forbidden"

    def test_write_does_not_imply_read(self, client: TestClient, state: AppState, env: str) -> None:
        """Effective permission is the union of explicitly granted
        operations. Implication would silently widen every grant.
        """
        writer = auth_header(token_for(state, Operation.WRITE))
        assert client.get(f"/v1/envs/{env}").status_code != 200
        assert client.get(f"/v1/envs/{env}", headers=writer).status_code == 403

    def test_a_token_for_another_org_cannot_read_this_one(
        self, client: TestClient, state: AppState, env: str
    ) -> None:
        scope = Scope(operations=Operation.READ, selectors=(NamePrefixSelector("other/*"),))
        token = state.signer.mint(Principal("outsider"), scope, ttl_us=3600 * 1_000_000)
        assert client.get(f"/v1/envs/{env}", headers=auth_header(token)).status_code == 403

    def test_a_stranger_cannot_tell_a_refusal_from_an_absence(
        self, client: TestClient, state: AppState, env: str
    ) -> None:
        """**The enumeration oracle.** A 403 must say nothing
        about whether its target exists — and a 403 says nothing only if the
        answer for a name that *does not* exist is identical.

        Otherwise the pair is a lookup service: try a name, read the status code,
        learn whether that environment is real. Nothing is authenticated away by
        this — the caller holds a perfectly valid token — so the corpus's
        environment names, which are business intelligence about what a
        competitor is training on, are enumerable by anyone with any account.

        Asserted as an equality between the two responses rather than against a
        literal, because what matters is not which code is returned but that one
        cannot be told from the other.
        """
        scope = Scope(operations=Operation.READ, selectors=(NamePrefixSelector("other/*"),))
        token = auth_header(state.signer.mint(Principal("snoop"), scope, ttl_us=3600 * 1_000_000))

        exists = client.get(f"/v1/envs/{env}", headers=token)
        absent = client.get("/v1/envs/proximal/does-not-exist", headers=token)

        assert exists.status_code == absent.status_code
        assert exists.json()["code"] == absent.json()["code"]
        assert exists.json()["message"] == absent.json()["message"]

    def test_a_pinned_token_learns_nothing_about_the_namespace_it_kept(
        self, client: TestClient, state: AppState, admin: dict[str, str], env: str
    ) -> None:
        """The sharpest form of the oracle, and the one a selector check misses.

        A per-rollout token is *narrowed* from a broad one, so it keeps the
        ``proximal/*`` selector it was minted from and merely gains a pin. Read
        the selectors alone and such a token still tells presence from absence
        across the whole namespace — while being permitted to read exactly one
        environment in it. These are handed to agent code treated as
        untrusted with respect to the corpus, which makes this the caller that
        most needs to learn nothing.
        """
        client.post("/v1/envs", json={"name": "proximal/pinned-to-me"}, headers=admin)
        mine = client.get("/v1/envs/proximal/pinned-to-me", headers=admin).json()["env_id"]

        pinned = auth_header(
            state.signer.mint(
                Principal("job:rollout/r-9001"),
                Scope(
                    operations=Operation.READ,
                    selectors=(NamePrefixSelector("proximal/*"),),
                    env_id=mine,
                ),
                ttl_us=3600 * 1_000_000,
            )
        )

        assert client.get("/v1/envs/proximal/pinned-to-me", headers=pinned).status_code == 200
        exists = client.get(f"/v1/envs/{env}", headers=pinned)
        absent = client.get("/v1/envs/proximal/does-not-exist", headers=pinned)
        assert exists.status_code == absent.status_code
        assert exists.json()["message"] == absent.json()["message"]

    def test_a_caller_holding_the_namespace_still_learns_a_name_is_free(
        self, client: TestClient, admin: dict[str, str]
    ) -> None:
        """The other half, and why the fix is not simply "always 403".

        Someone granted ``proximal/*`` already knows what is and is not in that
        namespace — they can list it. Hiding absence from them would tell them
        nothing they could not find out, and would turn every typo into a
        misleading permission error.
        """
        response = client.get("/v1/envs/proximal/does-not-exist", headers=admin)
        assert response.status_code == 404

    def test_an_env_pinned_token_cannot_reach_another_environment(
        self, client: TestClient, state: AppState, admin: dict[str, str], env: str
    ) -> None:
        """A per-rollout token pins one environment, because the agent code
        running inside it is untrusted with respect to the corpus.
        """
        client.post("/v1/envs", json={"name": "proximal/other"}, headers=admin)
        other_id = client.get("/v1/envs/proximal/other", headers=admin).json()["env_id"]

        pinned = state.signer.mint(
            Principal("job:rollout/r-8821"),
            Scope(
                operations=Operation.READ,
                selectors=(NamePrefixSelector("proximal/*"),),
                env_id=other_id,
            ),
            ttl_us=3600 * 1_000_000,
        )
        assert client.get(f"/v1/envs/{env}", headers=auth_header(pinned)).status_code == 403
        assert client.get("/v1/envs/proximal/other", headers=auth_header(pinned)).status_code == 200

    def test_annotate_cannot_read_content(
        self, client: TestClient, state: AppState, env: str, ledger: Ledger
    ) -> None:
        """The builder needs only ``env:annotate``, never write
        access to any environment it builds.
        """
        commit = publish_commit(ledger)
        builder = auth_header(token_for(state, Operation.ANNOTATE, principal="builder"))

        assert (
            client.put(
                f"/v1/envs/{env}/commits/{commit}/notes/build",
                json={"body": {"status": "ok"}},
                headers=builder,
            ).status_code
            == 200
        )
        assert client.get(f"/v1/envs/{env}/refs", headers=builder).status_code == 403

    def test_a_minted_token_cannot_widen_its_parent(
        self, client: TestClient, state: AppState, env: str
    ) -> None:
        """Attenuation only ever narrows. Widening is not expressible, which is
        what makes handing a token to a rollout safe.
        """
        reader = auth_header(token_for(state, Operation.READ))
        response = client.post(
            "/v1/tokens",
            json={"principal": "job:rollout/r-1", "operations": ["env:write"]},
            headers=reader,
        )
        assert response.status_code == 403

    def test_a_minted_token_may_narrow(self, client: TestClient, state: AppState, env: str) -> None:
        admin_token = token_for(state, Operation.READ, Operation.WRITE)
        response = client.post(
            "/v1/tokens",
            json={"principal": "job:rollout/r-1", "operations": ["env:read"], "ttl_seconds": 60},
            headers=auth_header(admin_token),
        )
        assert response.status_code == 200
        assert response.json()["operations"] == ["env:read"]

        narrowed = auth_header(response.json()["token"])
        assert client.get(f"/v1/envs/{env}", headers=narrowed).status_code == 200
        assert client.post(f"/v1/envs/{env}/sessions", headers=narrowed).status_code == 403


class TestCommitBinding:
    """The specification gap: a commit must belong to the environment.

    Without this, "no bare-hash read" is defeated one level up — a commit hash
    observed from environment B, presented with a valid token for A, would read
    B's content through A's authorization.
    """

    def test_a_commit_from_another_environment_is_refused(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        client.post("/v1/envs", json={"name": "proximal/secret"}, headers=admin)
        secret_commit = publish_commit(ledger, message="secret content")
        client.put(
            "/v1/envs/proximal/secret/refs/refs/heads/main",
            json={"target": str(secret_commit)},
            headers=admin,
        )

        # A perfectly valid token for `env`, and a real commit — from elsewhere.
        response = client.get(f"/v1/envs/{env}/commits/{secret_commit}", headers=admin)
        assert response.status_code == 403
        assert "not part of this environment" in response.json()["message"]

    def test_a_commit_in_this_environment_is_allowed(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger)
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(commit)}, headers=admin
        )
        assert client.get(f"/v1/envs/{env}/commits/{commit}", headers=admin).status_code == 200

    def test_an_ancestor_is_still_part_of_the_environment(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        """Reading an *old* version must keep working — that is the whole point
        of history — so reachability walks ancestors, not just ref targets.
        """
        first = publish_commit(ledger, message="v1")
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(first)}, headers=admin
        )
        second = publish_commit(ledger, message="v2", parent=first)
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main",
            json={"target": str(second), "expected_generation": 1},
            headers=admin,
        )
        assert client.get(f"/v1/envs/{env}/commits/{first}", headers=admin).status_code == 200


class TestErrorContract:
    def test_a_conflict_carries_enough_to_rebase(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        first = publish_commit(ledger, message="v1")
        second = publish_commit(ledger, message="v2")
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(first)}, headers=admin
        )

        response = client.put(
            f"/v1/envs/{env}/refs/refs/heads/main",
            json={"target": str(second), "expected_generation": 99},
            headers=admin,
        )
        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "conflict"
        # Enough to rebase without a second round trip — which means
        # the ref's current target and generation, not "your condition failed".
        assert body["details"]["current_generation"] == 1
        assert body["details"]["current_target"] == str(first)
        assert body["details"]["expected_generation"] == 99

    def test_a_reused_key_with_a_different_payload_is_422(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        """422 means exactly one thing. FastAPI's default 422 for body validation
        is remapped to 400 so this stays unambiguous.
        """
        first = publish_commit(ledger, message="v1")
        second = publish_commit(ledger, message="v2")
        headers = {**admin, "Idempotency-Key": "k-1"}

        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(first)}, headers=headers
        )
        response = client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(second)}, headers=headers
        )
        assert response.status_code == 422
        assert response.json()["code"] == "idempotency_mismatch"

    def test_a_replayed_key_returns_the_original_outcome(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger)
        headers = {**admin, "Idempotency-Key": "k-2"}
        body = {"target": str(commit)}

        first = client.put(f"/v1/envs/{env}/refs/refs/heads/main", json=body, headers=headers)
        second = client.put(f"/v1/envs/{env}/refs/refs/heads/main", json=body, headers=headers)

        assert first.status_code == second.status_code == 200
        assert first.json()["generation"] == second.json()["generation"] == 1

    def test_body_validation_is_400_not_422(
        self, client: TestClient, admin: dict[str, str], env: str
    ) -> None:
        response = client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": "not-a-name"}, headers=admin
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

    def test_a_missing_environment_is_404(self, client: TestClient, admin: dict[str, str]) -> None:
        assert client.get("/v1/envs/proximal/nonexistent", headers=admin).status_code == 404


class TestReadPath:
    def test_resolve_then_read_a_file(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger, message="hello")
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(commit)}, headers=admin
        )

        resolved = client.get(f"/v1/envs/{env}/refs/refs/heads/main/resolve", headers=admin).json()
        assert resolved["commit"] == str(commit)

        body = client.get(f"/v1/envs/{env}/commits/{commit}/file/README.md", headers=admin)
        assert body.status_code == 200
        assert body.content == b"content for hello"

    def test_a_ranged_read(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger, message="hello")
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(commit)}, headers=admin
        )
        body = client.get(
            f"/v1/envs/{env}/commits/{commit}/file/README.md?offset=4&length=3", headers=admin
        )
        assert body.content == b"ent"

    def test_listing_a_directory(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger)
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(commit)}, headers=admin
        )
        entries = client.get(f"/v1/envs/{env}/commits/{commit}/tree/", headers=admin).json()
        assert [e["name"] for e in entries["entries"]] == ["README.md"]
        assert entries["cursor"] is None


class TestContentTickets:
    """Possessing a hash must not be enough to fetch bytes."""

    def test_a_ticket_url_serves_the_bytes(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger, message="hello")
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(commit)}, headers=admin
        )
        described = client.get(
            f"/v1/envs/{env}/commits/{commit}/file/README.md?inline=false", headers=admin
        ).json()

        served = client.get(described["content_url"])
        assert served.status_code == 200
        assert served.content == b"content for hello"

    def test_a_bare_hash_without_a_ticket_is_refused(
        self, client: TestClient, ledger: Ledger
    ) -> None:
        """The whole point: hashes leak through logs and diffs, so a hash must
        not itself be a credential.
        """
        commit = publish_commit(ledger)
        response = client.get(f"/v1/content/{commit}?ticket=forged&principal=attacker")
        assert response.status_code == 401

    def test_a_ticket_for_one_object_does_not_open_another(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        commit = publish_commit(ledger, message="hello")
        other = publish_commit(ledger, message="different")
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(commit)}, headers=admin
        )
        described = client.get(
            f"/v1/envs/{env}/commits/{commit}/file/README.md?inline=false", headers=admin
        ).json()
        ticket = described["content_url"].split("ticket=")[1].split("&")[0]

        stolen = client.get(f"/v1/content/{other}?ticket={ticket}&principal=agent-17")
        assert stolen.status_code == 401

    def test_a_ticket_expires(
        self,
        client: TestClient,
        admin: dict[str, str],
        env: str,
        ledger: Ledger,
        clock: ManualClock,
    ) -> None:
        commit = publish_commit(ledger, message="hello")
        client.put(
            f"/v1/envs/{env}/refs/refs/heads/main", json={"target": str(commit)}, headers=admin
        )
        url = client.get(
            f"/v1/envs/{env}/commits/{commit}/file/README.md?inline=false", headers=admin
        ).json()["content_url"]

        assert client.get(url).status_code == 200
        clock.advance_seconds(16 * 60)
        assert client.get(url).status_code == 401


class TestObjectUpload:
    def test_the_server_rejects_bytes_that_do_not_match_their_name(
        self, client: TestClient, admin: dict[str, str], env: str
    ) -> None:
        """Under global deduplication this is a tenancy boundary, not a
        formality: a writer able to store arbitrary bytes under a chosen name
        would poison an environment it cannot reach.
        """
        honest = Chunk(b"the real content")
        response = client.put(
            f"/v1/envs/{env}/objects/{name_of(honest)}",
            content=encode(Chunk(b"malicious substitute")),
            headers=admin,
        )
        assert response.status_code == 500
        assert response.json()["code"] == "corrupt_object"

    def test_missing_reports_what_must_be_uploaded(
        self, client: TestClient, admin: dict[str, str], env: str, ledger: Ledger
    ) -> None:
        present = Chunk(b"already here")
        ledger.store.put_object(present)
        absent = Chunk(b"never seen")

        response = client.post(
            f"/v1/envs/{env}/objects/missing",
            json={"names": [str(name_of(present)), str(name_of(absent))]},
            headers=admin,
        )
        assert response.json()["missing"] == [str(name_of(absent))]

    def test_upload_then_read_back(
        self, client: TestClient, admin: dict[str, str], env: str
    ) -> None:
        chunk = Chunk(b"uploaded over http")
        created = client.put(
            f"/v1/envs/{env}/objects/{name_of(chunk)}", content=encode(chunk), headers=admin
        )
        assert created.status_code == 201

        again = client.put(
            f"/v1/envs/{env}/objects/{name_of(chunk)}", content=encode(chunk), headers=admin
        )
        assert again.status_code == 200, "a re-upload deduplicates rather than creating"


class TestRouteCoverage:
    def test_every_route_declares_authorization(self) -> None:
        """A missing authorization check is invisible until someone exploits it.

        Walk the route table and assert that every path under ``/v1`` either
        requires a capability or is on the explicit public list.

        The router is inspected directly rather than through ``app.routes``:
        this FastAPI version *mounts* an included router rather than copying its
        routes up, so the app's own list holds a single opaque entry.
        """
        from src.api.routes import router

        # Deliberately unauthenticated. `/metrics` reports corpus-wide aggregates
        # and names no environment, principal or object; `/content` is
        # authorized by ticket instead of by token.
        public = {"/v1/healthz", "/v1/metrics", "/v1/content/{name}", "/v1/tokens"}
        unguarded: list[str] = []

        for route in router.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/v1") or path in public:
                continue
            dependant = getattr(route, "dependant", None)
            if dependant is None:
                continue
            names = _dependency_names(dependant)
            if not ({"dependency", "current_capability"} & names):
                unguarded.append(f"{sorted(getattr(route, 'methods', []))} {path}")

        assert not unguarded, "routes with no authorization:\n  " + "\n  ".join(unguarded)

    def test_the_coverage_check_sees_the_real_routes(self) -> None:
        """Guards the guard: if the walk found nothing, the assertion above is
        vacuous — which is exactly what happened when this FastAPI version
        started mounting included routers instead of flattening them.
        """
        from src.api.routes import router

        paths = {getattr(r, "path", "") for r in router.routes}
        assert len([p for p in paths if p.startswith("/v1/envs")]) >= 8


def _dependency_names(dependant: Any, depth: int = 0) -> set[str]:
    if depth > 5:
        return set()
    names = {getattr(dependant.call, "__name__", "")}
    for sub in dependant.dependencies:
        names |= _dependency_names(sub, depth + 1)
    return names
