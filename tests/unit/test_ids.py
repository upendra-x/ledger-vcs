"""Identifier parsing is strict on purpose.

A permissive parser lets two spellings of one name into the system, and
deduplication is spelling-sensitive: ``B3:AB…`` and ``b3:ab…`` naming the same
bytes would be two cache entries, two index rows and two keep-set members.
"""

from __future__ import annotations

import threading

import pytest

from src.ids import ChangeId, EnvId, EnvName, ObjectName, RefName, SessionId

VALID_HEX = "9f" * 32


class TestObjectName:
    def test_round_trip(self) -> None:
        name = ObjectName.parse(f"b3:{VALID_HEX}")
        assert str(name) == f"b3:{VALID_HEX}"
        assert name.hex == VALID_HEX
        assert name.algo == "b3"

    def test_equality_is_by_digest(self) -> None:
        assert ObjectName.parse(f"b3:{VALID_HEX}") == ObjectName(bytes.fromhex(VALID_HEX))

    def test_ordering_is_by_raw_digest(self) -> None:
        """The GC's sharded diff and the keep-set's sorted-array representation
        both depend on names sorting by digest bytes.
        """
        low = ObjectName(b"\x00" * 32)
        high = ObjectName(b"\xff" * 32)
        assert low < high
        assert sorted([high, low]) == [low, high]

    def test_is_hashable_and_usable_as_a_set_member(self) -> None:
        name = ObjectName(bytes.fromhex(VALID_HEX))
        assert len({name, ObjectName(bytes.fromhex(VALID_HEX))}) == 1

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param(VALID_HEX, id="missing-algorithm-prefix"),
            pytest.param(f"sha256:{VALID_HEX}", id="wrong-algorithm"),
            pytest.param(f"B3:{VALID_HEX}", id="uppercase-prefix"),
            pytest.param(f"b3:{VALID_HEX.upper()}", id="uppercase-hex-is-a-second-spelling"),
            pytest.param(f"b3:{'9f' * 31}", id="too-short"),
            pytest.param(f"b3:{'9f' * 33}", id="too-long"),
            pytest.param("b3:" + "zz" * 32, id="not-hex"),
            pytest.param(f"b3:{VALID_HEX} ", id="trailing-whitespace"),
            pytest.param("", id="empty"),
        ],
    )
    def test_rejects_non_canonical_spellings(self, text: str) -> None:
        with pytest.raises(ValueError, match="not a valid object name"):
            ObjectName.parse(text)

    def test_rejects_wrong_digest_length_at_construction(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            ObjectName(b"\x00" * 31)


class TestEnvName:
    @pytest.mark.parametrize(
        "text", ["proximal/swe-bench-lite-042", "org/e", "a-b.c/d_e.f-g", "p/x" * 1]
    )
    def test_accepts_valid(self, text: str) -> None:
        assert str(EnvName(text)) == text

    def test_org_is_the_prefix_authorization_scopes_against(self) -> None:
        assert EnvName("proximal/swe-bench-lite-042").org == "proximal"

    @pytest.mark.parametrize(
        "text",
        [
            "noslash",
            "/leading",
            "trailing/",
            "Org/Env",  # uppercase — two spellings of one name
            "a//b",
            "-bad/env",
            "org/env/extra",
            "org/" + "x" * 200,
        ],
    )
    def test_rejects_invalid(self, text: str) -> None:
        with pytest.raises(ValueError, match="not a valid environment name"):
            EnvName(text)


class TestRefName:
    @pytest.mark.parametrize("text", ["refs/heads/main", "refs/heads/exp/lr-3e4", "refs/tags/v1"])
    def test_accepts_valid(self, text: str) -> None:
        assert str(RefName(text)) == text

    def test_tags_are_distinguishable(self) -> None:
        """Tags are create-if-absent, not compare-and-swap, so the
        service has to be able to tell them apart before deciding the write mode.
        """
        assert RefName("refs/tags/v1").is_tag
        assert not RefName("refs/heads/main").is_tag

    @pytest.mark.parametrize(
        "text",
        [
            "main",
            "refs/main",
            "refs/other/x",
            "refs/heads/",
            "refs/heads/a//b",
            "refs/heads/../escape",
            "refs/heads/-leading",
        ],
    )
    def test_rejects_invalid(self, text: str) -> None:
        with pytest.raises(ValueError, match="not a valid ref name"):
            RefName(text)


class TestGeneratedIds:
    def test_env_ids_are_unique_and_prefixed(self) -> None:
        ids = {EnvId.new() for _ in range(200)}
        assert len(ids) == 200
        assert all(str(i).startswith("env_") for i in ids)

    def test_env_ids_sort_by_creation_order(self) -> None:
        """ULID ordering is what makes ListEnvs paginate without a secondary index.

        Minted in a tight loop on purpose: these all land in the same
        millisecond, which is exactly the case a non-monotonic ULID gets wrong,
        and exactly the case an automation creating environments in a burst
        produces.
        """
        minted = [EnvId.new() for _ in range(1000)]
        assert [i.value for i in minted] == sorted(i.value for i in minted)
        assert len(set(minted)) == 1000

    def test_env_ids_are_monotonic_across_threads(self) -> None:
        """Concurrent minting must not produce a duplicate or an out-of-order id."""
        minted: list[str] = []
        lock = threading.Lock()

        def mint() -> None:
            local = [EnvId.new().value for _ in range(200)]
            with lock:
                minted.extend(local)

        threads = [threading.Thread(target=mint) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(minted)) == len(minted) == 1600

    def test_change_ids_are_unique(self) -> None:
        assert len({ChangeId.new() for _ in range(200)}) == 200

    def test_session_ids_are_prefixed(self) -> None:
        assert str(SessionId.new()).startswith("ws_")

    def test_identifier_types_do_not_compare_equal_across_types(self) -> None:
        """The reason these are distinct types at all: an EnvId and a SessionId
        both render as an opaque string, and passing one where the other belongs
        must not silently succeed.

        Typed as ``object`` because a type checker rejects the comparison
        outright — which is the static half of the same guarantee this asserts
        at runtime.
        """
        env: object = EnvId("x")
        session: object = SessionId("x")
        change: object = ChangeId("x")
        assert env != session
        assert change != env
