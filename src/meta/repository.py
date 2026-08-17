"""All mutable-state semantics, expressed once over the store's primitives.

Everything that makes the mutable side *correct* is here rather than in the
store: the generation compare-and-swap, the dense operation counter, the
idempotency triple-outcome, undo, and the fork/rename sequences. The store
knows about conditioned items; this knows about refs.

That split is the whole point. A domain-level store interface would force every
backend to re-implement these, so the invariants would be proven once per
backend. Written here, they are proven once and every backend inherits them.

**The commit transaction.** Moving a ref is one single-partition transaction
containing, in order:

    1. compare-and-swap the ref on its *generation*
    2. bump the environment's dense operation counter
    3. append the operation-log entry
    4. append the change record
    5. record the idempotency outcome
    6. emit the change-stream event

All six, or none. The ordering matters: the ref condition is evaluated first, so
a genuine conflict fails before anything else is touched and cannot be confused
with counter contention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, final

from blake3 import blake3

from src.errors import Conflict, IdempotencyMismatch, InvalidRequest, NotFound
from src.ids import ChangeId, EnvId, EnvName, ObjectName, RefName, SessionId
from src.meta import keys
from src.meta.keys import GlobalSpace, ItemKind
from src.meta.models import (
    ChangeRecord,
    Environment,
    EnvState,
    IdempotencyRecord,
    Note,
    OpKind,
    OpLogEntry,
    Ref,
    RefLifecycle,
    WriteSession,
)
from src.meta.store import (
    Absent,
    Delete,
    Emit,
    GenerationIs,
    Item,
    Key,
    Put,
    VersionIs,
    WriteOp,
    canonical_json,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.clock import Clock
    from src.meta.store import MetadataStore

__all__ = ["DEFAULT_REF", "MetadataRepository", "RefUpdate"]

#: How long an idempotency record survives. Past it the record is
#: gone and the request is treated as fresh — which is safe because the
#: compare-and-swap remains the backstop against double-application.
IDEMPOTENCY_TTL_US: Final = 24 * 3600 * 1_000_000

#: How long a write session's lease runs before its objects become collectable.
WRITE_LEASE_TTL_US: Final = 3600 * 1_000_000

#: Bounded retries when the operation counter is contended. Contention is
#: per-environment and the capacity plan projects roughly one commit per environment per
#: fourteen hours, so reaching this bound means something is wrong.
_COUNTER_RETRIES: Final = 8

#: Object names per session page. A thousand 32-byte names is a ~70 KB item,
#: comfortably inside every partitioned store's item-size limit.
_SESSION_PAGE_SIZE: Final = 1000

#: The ref an environment resolves to when nothing else is said.
DEFAULT_REF: Final = RefName("refs/heads/main")


@final
@dataclass(frozen=True, slots=True)
class RefUpdate:
    """The outcome of a ref mutation."""

    ref: Ref
    op_sequence: int
    #: True when this was a replay of a recorded idempotent outcome rather than
    #: a fresh application. Callers surface it so a retrying client can tell
    #: "my write landed" from "I lost a race".
    replayed: bool = False


@final
class MetadataRepository:
    """Ledger nouns, over the store's conditioned items."""

    __slots__ = ("_clock", "_store")

    def __init__(self, store: MetadataStore, *, clock: Clock) -> None:
        self._store = store
        self._clock = clock

    # ── environments ─────────────────────────────────────────────────────────

    def create_env(
        self,
        name: EnvName,
        *,
        owner: str = "",
        default_ref: RefName = DEFAULT_REF,
        labels: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> Environment:
        """Claim a name, then write the environment record.

        Two writes in two keyspaces, deliberately not one transaction: name
        uniqueness spans all environments, so the registry cannot live in any
        one environment's partition. The claim happens first, so a
        crash between the steps leaves a claimed-but-unused name that
        ``sweep_orphan_name_claims`` reclaims — the safe direction, since the
        alternative would be an environment nobody can address.

        **Idempotency lives on the claim rather than in a partition.** Every
        other mutation records its key under ``idem#`` in the environment it
        touches; this one has no environment until it succeeds. The name claim is
        the only thing that exists at the moment the decision is made, so the key
        rides there — and a retry is answered by the same read that detects a
        collision, with no extra round trip.

        Without it a retried create is indistinguishable from losing a race: both
        are "name already taken", and an automation replaying a request it never
        saw the response to would report a failure for work that had succeeded.
        """
        env = Environment(
            env_id=EnvId.new(),
            name=name,
            default_ref=default_ref,
            owner=owner,
            labels=dict(labels or {}),
            created_at_us=self._clock.now_us(),
        )
        claim: dict[str, Any] = {"env_id": str(env.env_id)}
        if idempotency_key is not None:
            claim["idempotency_key"] = idempotency_key

        if not self._store.claim_global(GlobalSpace.NAMES, str(name), claim):
            replayed = self._replay_create_env(name, idempotency_key)
            if replayed is not None:
                return replayed
            raise Conflict("environment name is already taken", name=str(name))

        partition = str(env.env_id)
        self._store.transact_write(
            [
                Put(Key(partition, keys.env_sk()), ItemKind.ENV, env.to_body(), condition=Absent()),
                Put(Key(partition, keys.seq_sk()), ItemKind.SEQ, {"next": 1}, condition=Absent()),
            ]
        )
        return env

    def _replay_create_env(self, name: EnvName, key: str | None) -> Environment | None:
        """The environment this key already created, if it is the one holding the name.

        Only a claim carrying the *same* key replays. A different key, or none,
        means somebody else owns the name and the caller has to be told so.
        """
        if key is None:
            return None
        claim = self._store.read_global(GlobalSpace.NAMES, str(name))
        if claim is None or claim.body.get("idempotency_key") != key:
            return None
        return self.get_env(EnvId(str(claim.body["env_id"])))

    def get_env(self, env: EnvId) -> Environment:
        item = self._store.get(Key(str(env), keys.env_sk()))
        if item is None:
            raise NotFound("environment does not exist", env_id=str(env))
        return Environment.from_body(item.body)

    def resolve_env_name(self, name: EnvName) -> EnvId:
        """One indexed lookup. Automations hold ids; humans and CLIs use names."""
        item = self._store.read_global(GlobalSpace.NAMES, str(name))
        if item is None:
            raise NotFound("environment name is not registered", name=str(name))
        return EnvId(item.body["env_id"])

    def list_envs(self, *, prefix: str = "", limit: int = 100) -> list[EnvName]:
        return [
            EnvName(item.key.sk)
            for item in self._store.scan_global(GlobalSpace.NAMES, prefix=prefix, limit=limit)
        ]

    def rename_env(self, env: EnvId, new_name: EnvName) -> Environment:
        """Claim the new name, repoint the record, release the old one.

        **A rename is atomic for the new name, and swapping two names is not.**
        That bound is worth stating rather than hiding: making swaps atomic would
        mean putting all ten million names in one partition — a system-wide
        contention point bought for a rare convenience. Automations
        hold ids, so only name-based lookups see the window.
        """
        record = self.get_env(env)
        if record.name == new_name:
            return record

        if not self._store.claim_global(GlobalSpace.NAMES, str(new_name), {"env_id": str(env)}):
            raise Conflict("environment name is already taken", name=str(new_name))

        updated = replace(record, name=new_name)
        self._store.transact_write(
            [Put(Key(str(env), keys.env_sk()), ItemKind.ENV, updated.to_body())]
        )
        self._store.delete_global(GlobalSpace.NAMES, str(record.name))
        return updated

    def set_env_state(self, env: EnvId, state: EnvState) -> Environment:
        return self._replace_env(env, state=state)

    def set_env_labels(self, env: EnvId, labels: Mapping[str, str]) -> Environment:
        """Replace an environment's labels.

        Whole-map replacement rather than a merge, because labels are a *grant
        selector*: a partial update whose semantics depend on what
        was already there makes "which environments does this token reach" a
        question nobody can answer from the request alone.
        """
        return self._replace_env(env, labels=dict(labels))

    def _replace_env(self, env: EnvId, **changes: Any) -> Environment:
        updated = replace(self.get_env(env), **changes)
        self._store.transact_write(
            [Put(Key(str(env), keys.env_sk()), ItemKind.ENV, updated.to_body())]
        )
        return updated

    def sweep_orphan_name_claims(self, *, limit: int = 100) -> int:
        """Reclaim names claimed by a rename or create that never completed.

        The check is exact rather than heuristic: a claim is orphaned only if the
        environment it names does not point back at it. A claim whose target
        agrees is live, whatever its age.
        """
        reclaimed = 0
        for item in self._store.scan_global(GlobalSpace.NAMES, limit=limit):
            env_id = EnvId(item.body["env_id"])
            record = self._store.get(Key(str(env_id), keys.env_sk()))
            if record is None or record.body.get("name") != item.key.sk:
                self._store.delete_global(GlobalSpace.NAMES, item.key.sk)
                reclaimed += 1
        return reclaimed

    # ── refs ─────────────────────────────────────────────────────────────────

    def get_ref(self, env: EnvId, name: RefName) -> Ref:
        item = self._store.get(Key(str(env), keys.ref_sk(name)))
        if item is None:
            raise NotFound("ref does not exist", env_id=str(env), ref=str(name))
        return Ref.from_body(item.body)

    def try_get_ref(self, env: EnvId, name: RefName) -> Ref | None:
        item = self._store.get(Key(str(env), keys.ref_sk(name)))
        return Ref.from_body(item.body) if item else None

    def list_refs(self, env: EnvId, *, limit: int = 1000) -> list[Ref]:
        return [
            Ref.from_body(item.body)
            for item in self._store.query(str(env), keys.ref_prefix(), limit=limit)
        ]

    def create_ref(
        self,
        env: EnvId,
        name: RefName,
        target: ObjectName,
        *,
        principal: str,
        lifecycle: RefLifecycle = RefLifecycle.PERMANENT,
        ttl_us: int | None = None,
        change_id: ChangeId | None = None,
        idempotency_key: str | None = None,
        idempotency_request: Mapping[str, Any] | None = None,
    ) -> RefUpdate:
        """Create a ref that does not exist. Create-if-absent, not a CAS."""
        return self._mutate_ref(
            env,
            name,
            target=target,
            condition=Absent(),
            generation=1,
            kind=OpKind.CREATE_REF,
            principal=principal,
            lifecycle=lifecycle,
            ttl_us=ttl_us,
            change_id=change_id,
            idempotency_key=idempotency_key,
            idempotency_request=idempotency_request,
            before=None,
        )

    def update_ref(
        self,
        env: EnvId,
        name: RefName,
        *,
        expected_generation: int,
        target: ObjectName,
        principal: str,
        change_id: ChangeId | None = None,
        idempotency_key: str | None = None,
        idempotency_request: Mapping[str, Any] | None = None,
        op_kind: OpKind = OpKind.UPDATE_REF,
    ) -> RefUpdate:
        """Move a ref, comparing on its generation.

        Raises ``Conflict`` carrying the ref's *current* target and generation,
        which is everything the caller needs to rebase without a second round
        trip — the "a 409 is not a bare failure".
        """
        current = self.try_get_ref(env, name)
        if current is None:
            raise NotFound("ref does not exist", env_id=str(env), ref=str(name))
        if current.name.is_tag:
            raise Conflict(
                "tags are written once and then frozen", ref=str(name), target=str(current.target)
            )

        return self._mutate_ref(
            env,
            name,
            target=target,
            condition=GenerationIs(expected_generation),
            generation=current.generation + 1,
            kind=op_kind,
            principal=principal,
            lifecycle=current.lifecycle,
            ttl_us=None,
            keep_expiry=current.expires_at_us,
            change_id=change_id,
            idempotency_key=idempotency_key,
            idempotency_request=idempotency_request,
            before=current.target,
        )

    def delete_ref(
        self,
        env: EnvId,
        name: RefName,
        *,
        expected_generation: int,
        principal: str,
        idempotency_key: str | None = None,
    ) -> int:
        """Delete a ref. Instant and synchronous — no automation waits on
        collection. Reclaiming the storage happens later, and
        reclaims exactly what nothing else still needs.
        """
        current = self.get_ref(env, name)
        partition = str(env)

        def build(sequence: int, counter_version: int) -> list[WriteOp]:
            return [
                Delete(
                    Key(partition, keys.ref_sk(name)),
                    condition=GenerationIs(expected_generation),
                ),
                *self._counter_writes(partition, sequence, counter_version),
                self._op_write(
                    partition,
                    OpLogEntry(
                        sequence=sequence,
                        kind=OpKind.DELETE_REF,
                        ref=name,
                        before=current.target,
                        after=None,
                        principal=principal,
                        at_us=self._clock.now_us(),
                    ),
                ),
                Emit(
                    partition,
                    "ref.deleted",
                    {
                        "env_id": partition,
                        "ref": str(name),
                        "before": str(current.target),
                        "op_sequence": sequence,
                    },
                ),
            ]

        request = {
            "op": str(OpKind.DELETE_REF),
            "ref": str(name),
            "expected_generation": expected_generation,
        }
        sequence, _ = self._commit_with_counter(
            partition, build, idempotency_key, request, {"deleted": str(name)}
        )
        return sequence

    # ── the shared ref mutation path ─────────────────────────────────────────

    def _mutate_ref(
        self,
        env: EnvId,
        name: RefName,
        *,
        target: ObjectName,
        condition: Absent | GenerationIs,
        generation: int,
        kind: OpKind,
        principal: str,
        lifecycle: RefLifecycle,
        ttl_us: int | None,
        change_id: ChangeId | None,
        idempotency_key: str | None,
        before: ObjectName | None,
        idempotency_request: Mapping[str, Any] | None = None,
        keep_expiry: int | None = None,
    ) -> RefUpdate:
        partition = str(env)
        now = self._clock.now_us()
        expires_at = keep_expiry if ttl_us is None else now + ttl_us
        if lifecycle is RefLifecycle.EPHEMERAL and expires_at is None:
            raise InvalidRequest("an ephemeral ref requires a time to live")

        ref = Ref(
            name=name,
            target=target,
            generation=generation,
            lifecycle=lifecycle,
            expires_at_us=expires_at,
            updated_by=principal,
            updated_at_us=now,
        )

        def build(sequence: int, counter_version: int) -> list[WriteOp]:
            writes: list[WriteOp] = [
                Put(
                    Key(partition, keys.ref_sk(name)),
                    ItemKind.REF,
                    ref.to_body(),
                    condition=condition,
                    generation=generation,
                    expires_at_us=expires_at,
                ),
                *self._counter_writes(partition, sequence, counter_version),
                self._op_write(
                    partition,
                    OpLogEntry(
                        sequence=sequence,
                        kind=kind,
                        ref=name,
                        before=before,
                        after=target,
                        principal=principal,
                        at_us=now,
                    ),
                ),
            ]
            if change_id is not None:
                writes.append(self._change_write(partition, change_id, target))
            writes.append(
                Emit(
                    partition,
                    "ref.updated",
                    {
                        "env_id": partition,
                        "ref": str(name),
                        "before": str(before) if before else None,
                        "after": str(target),
                        "generation": generation,
                        # The operation this event *is*. Events
                        # come from the operation log, and carrying its sequence
                        # is what lets a consumer key work on the operation
                        # rather than inventing a second notion of order that
                        # could disagree with the first.
                        "op_sequence": sequence,
                    },
                )
            )
            return writes

        # A caller may supply the request the fingerprint should cover, because
        # only the caller knows what was actually asked for. A commit, for
        # instance, is "make this ref name a commit of this tree" — whether that
        # becomes a create or a compare-and-swap is the service's business, and
        # a retry must not be rejected merely because the first attempt created
        # the ref and the second would update it.
        request = idempotency_request or {
            "op": str(kind),
            "ref": str(name),
            "target": str(target),
            "expected_generation": (
                condition.generation if isinstance(condition, GenerationIs) else None
            ),
        }
        outcome = {"target": str(target), "generation": generation}
        try:
            sequence, replayed = self._commit_with_counter(
                partition, build, idempotency_key, request, outcome
            )
        except Conflict as conflict:
            raise self._describe_conflict(env, name, conflict) from conflict
        if replayed:
            return RefUpdate(ref=self.get_ref(env, name), op_sequence=sequence, replayed=True)
        return RefUpdate(ref=ref, op_sequence=sequence)

    def _describe_conflict(self, env: EnvId, name: RefName, conflict: Conflict) -> Conflict:
        """Turn a failed condition into something a caller can act on.

        The response is ``409 { target, generation }`` — *"here
        is what it actually is now"* — and the reason is that a caller must be able to
        rebase **without a second round trip**. A bare "your condition failed"
        forces exactly that round trip, and under contention the re-read races
        the next writer.

        It also stops the store's own vocabulary escaping. The partition and sort
        keys are how *this* backend happens to address a ref; a caller that
        started matching on them would break when the backend changed.
        """
        current = self.try_get_ref(env, name)
        details: dict[str, Any] = {
            "ref": str(name),
            "expected_generation": conflict.details.get("expected"),
        }
        if current is not None:
            details["current_target"] = str(current.target)
            details["current_generation"] = current.generation
        else:
            # The ref was deleted rather than moved. Saying so is the difference
            # between "retry from generation 43" and "there is nothing to retry".
            details["current_target"] = None
            details["current_generation"] = None
        return Conflict(
            "this ref moved since you read it; rebase onto the current target",
            **details,
        )

    def _commit_with_counter(
        self,
        partition: str,
        build: Any,
        idempotency_key: str | None,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> tuple[int, bool]:
        """Run a mutation, allocating an operation sequence and honouring the key.

        The idempotency record is written **inside the committing transaction**,
        never before it. A record written first would claim a key for a mutation
        that then failed, and every subsequent retry would replay a success that
        never happened.

        The fingerprint covers the **request**, not the outcome. That distinction
        is not pedantic: an outcome contains server-computed state — the new
        generation — which differs between the original call and its replay, so
        fingerprinting it would report every genuine retry as a client bug.
        """
        if idempotency_key is not None:
            recorded = self._replay(partition, idempotency_key, request)
            if recorded is not None:
                return recorded, True

        last_conflict: Conflict | None = None
        for _ in range(_COUNTER_RETRIES):
            sequence, counter_version = self._next_sequence(partition)
            writes = list(build(sequence, counter_version))
            if idempotency_key is not None:
                writes.append(
                    self._idempotency_write(partition, idempotency_key, request, outcome, sequence)
                )
            try:
                self._store.transact_write(writes)
            except Conflict as exc:
                if exc.details.get("sk") == keys.seq_sk():
                    last_conflict = exc
                    continue  # counter contention, not a real conflict

                # A conflict on a *keyed* request may mean a concurrent retry of
                # the same request already applied it — in which case the ref
                # compare-and-swap fails first, before the idempotency record is
                # ever reached. So re-check the key before reporting a conflict:
                # if the winner recorded our exact request, their outcome is
                # ours. Without this, N concurrent retries produce one success
                # and N-1 spurious 409s for work that demonstrably happened.
                if idempotency_key is not None:
                    recorded = self._replay(partition, idempotency_key, request)
                    if recorded is not None:
                        return recorded, True
                raise
            return sequence, False

        raise Conflict(
            "operation counter is too contended to allocate a sequence",
            partition=partition,
        ) from last_conflict

    def _next_sequence(self, partition: str) -> tuple[int, int]:
        item = self._store.get(Key(partition, keys.seq_sk()))
        if item is None:
            raise NotFound("environment does not exist", env_id=partition)
        return int(item.body["next"]), item.version

    def _counter_writes(self, partition: str, sequence: int, counter_version: int) -> list[WriteOp]:
        """Advance the dense per-environment counter, conditionally.

        Read outside the transaction and advanced inside it under a version
        condition — the shape a partitioned store can actually execute, since
        none of them let a transaction read its own writes.
        """
        return [
            Put(
                Key(partition, keys.seq_sk()),
                ItemKind.SEQ,
                {"next": sequence + 1},
                condition=VersionIs(counter_version),
            )
        ]

    def _op_write(self, partition: str, entry: OpLogEntry) -> WriteOp:
        return Put(
            Key(partition, keys.op_sk(entry.sequence)),
            ItemKind.OP,
            entry.to_body(),
            condition=Absent(),
        )

    def _change_write(self, partition: str, change: ChangeId, commit: ObjectName) -> WriteOp:
        """Append to a change's commit list, newest first.

        An automation refers to "the change that adds the verifier"
        across an amendment, instead of chasing a hash that moves underneath it.
        """
        existing = self._store.get(Key(partition, keys.change_sk(change)))
        commits = [commit, *ChangeRecord.from_body(existing.body).commits] if existing else [commit]
        record = ChangeRecord(change_id=change, commits=commits)
        return Put(Key(partition, keys.change_sk(change)), ItemKind.CHANGE, record.to_body())

    # ── idempotency ──────────────────────────────────────────────────────────

    def recorded_outcome(
        self, env: EnvId, key: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """The outcome already recorded for this key, if any.

        Lets a service detect a replay *before* doing work whose only effect
        would be unreferenced garbage — writing a commit object that no ref will
        ever name. Raises ``IdempotencyMismatch`` if the key was used for a
        different request.
        """
        item = self._store.get(Key(str(env), keys.idem_sk(key)))
        if item is None:
            return None
        record = IdempotencyRecord.from_body(item.body)
        if record.fingerprint != _fingerprint(request):
            raise IdempotencyMismatch(
                "idempotency key was reused with a different request", key=key
            )
        return record.outcome

    def _replay(self, partition: str, key: str, request: Mapping[str, Any]) -> int | None:
        item = self._store.get(Key(partition, keys.idem_sk(key)))
        if item is None:
            return None
        record = IdempotencyRecord.from_body(item.body)
        if record.fingerprint != _fingerprint(request):
            raise IdempotencyMismatch(
                "idempotency key was reused with a different request",
                key=key,
            )
        return int(record.outcome["op_sequence"])

    def _idempotency_write(
        self,
        partition: str,
        key: str,
        request: Mapping[str, Any],
        outcome: Mapping[str, Any],
        sequence: int,
    ) -> WriteOp:
        record = IdempotencyRecord(
            key=key,
            fingerprint=_fingerprint(request),
            outcome={**outcome, "op_sequence": sequence},
            expires_at_us=self._clock.now_us() + IDEMPOTENCY_TTL_US,
            created_at_us=self._clock.now_us(),
        )
        return Put(
            Key(partition, keys.idem_sk(key)),
            ItemKind.IDEMPOTENCY,
            record.to_body(),
            condition=Absent(),
            expires_at_us=record.expires_at_us,
        )

    # ── the operation log ────────────────────────────────────────────────────

    def list_ops(
        self, env: EnvId, *, limit: int = 100, descending: bool = True
    ) -> list[OpLogEntry]:
        items = self._store.query(str(env), f"{ItemKind.OP}#", limit=limit, descending=descending)
        return [OpLogEntry.from_body(item.body) for item in items]

    def _get_op(self, env: EnvId, sequence: int) -> OpLogEntry:
        item = self._store.get(Key(str(env), keys.op_sk(sequence)))
        if item is None:
            raise NotFound("operation does not exist", env_id=str(env), sequence=sequence)
        return OpLogEntry.from_body(item.body)

    def undo(
        self, env: EnvId, sequence: int, *, principal: str, idempotency_key: str | None = None
    ) -> RefUpdate:
        """Restore a ref to its state before an operation.

        **Undo is an ordinary ref update and is treated as one.** It expects the
        generation the ref has *now*, not the one it had then — so a writer who
        moved it in the meantime produces the usual 409 and the caller decides
        whether that change survives. An undo that ignored the generation would
        be exactly the lost update compare-and-swap exists to prevent, dressed up as a
        recovery feature.
        """
        entry = self._get_op(env, sequence)
        if entry.ref is None:
            raise InvalidRequest("this operation did not move a ref", sequence=sequence)

        current = self.try_get_ref(env, entry.ref)

        if entry.before is None:
            # Undoing a creation means deleting it again.
            if current is None:
                raise NotFound("ref is already absent", ref=str(entry.ref))
            self.delete_ref(
                env,
                entry.ref,
                expected_generation=current.generation,
                principal=principal,
                idempotency_key=idempotency_key,
            )
            return RefUpdate(ref=current, op_sequence=sequence)

        if current is None:
            # Undoing a deletion recreates it, failing if the name was taken again.
            return self.create_ref(
                env,
                entry.ref,
                entry.before,
                principal=principal,
                idempotency_key=idempotency_key,
            )

        return self.update_ref(
            env,
            entry.ref,
            expected_generation=current.generation,
            target=entry.before,
            principal=principal,
            idempotency_key=idempotency_key,
            op_kind=OpKind.UNDO,
        )

    # ── changes ──────────────────────────────────────────────────────────────

    def resolve_change(self, env: EnvId, change: ChangeId) -> ChangeRecord:
        """A change, by its full id or by any prefix that names exactly one.

        Prefixes are accepted because a full change id is thirty-two hex
        characters and nothing ever shows one: history is displayed abbreviated,
        the way every version control system displays it. An exact-match-only
        lookup therefore rejects the only form a person or an agent has ever
        seen — the id is *there on the screen* and it does not work — which is a
        worse failure than not offering the command.

        Ambiguity is an error rather than a first match. Silently picking one of
        two changes would let an automation amend something it never named, and
        the whole point of a change id is to be the identity that survives
        rewriting.

        Costs one prefix scan inside a single partition, which is
        the same shape as any other read here.
        """
        exact = self._store.get(Key(str(env), keys.change_sk(change)))
        if exact is not None:
            return ChangeRecord.from_body(exact.body)

        # Two are enough to know it is ambiguous, and stop the scan from being
        # unbounded when someone passes an empty or one-character prefix.
        matches = self._store.query(str(env), keys.change_sk(change), limit=2)
        if not matches:
            raise NotFound("change does not exist", env_id=str(env), change_id=str(change))
        if len(matches) > 1:
            raise InvalidRequest(
                "that change id prefix matches more than one change",
                change_id=str(change),
                matches=[item.key.sk.removeprefix(f"{ItemKind.CHANGE}#") for item in matches],
            )
        return ChangeRecord.from_body(matches[0].body)

    # ── notes ────────────────────────────────────────────────────────────────

    def put_note(
        self,
        env: EnvId,
        commit: ObjectName,
        namespace: str,
        body: Mapping[str, Any],
        *,
        principal: str = "",
    ) -> Note:
        note = Note(
            commit=commit,
            namespace=namespace,
            body=dict(body),
            updated_by=principal,
            updated_at_us=self._clock.now_us(),
        )
        self._store.transact_write(
            [
                Put(
                    Key(str(env), keys.note_sk(commit, namespace)),
                    ItemKind.NOTE,
                    note.to_body(),
                )
            ]
        )
        return note

    def get_note(self, env: EnvId, commit: ObjectName, namespace: str) -> Note:
        item = self._store.get(Key(str(env), keys.note_sk(commit, namespace)))
        if item is None:
            raise NotFound("note does not exist", commit=str(commit), namespace=namespace)
        return Note.from_body(item.body)

    def list_notes(self, env: EnvId, commit: ObjectName | None = None) -> list[Note]:
        items = self._store.query(str(env), keys.note_prefix(commit), limit=1000)
        return [Note.from_body(item.body) for item in items]

    # ── write sessions ───────────────────────────────────────────────────────

    def begin_write(
        self, env: EnvId, *, principal: str, ttl_us: int = WRITE_LEASE_TTL_US
    ) -> WriteSession:
        session = WriteSession(
            session_id=SessionId.new(),
            principal=principal,
            expires_at_us=self._clock.now_us() + ttl_us,
            created_at_us=self._clock.now_us(),
        )
        self._store.transact_write(
            [
                Put(
                    Key(str(env), keys.session_sk(session.session_id)),
                    ItemKind.SESSION,
                    session.to_body(),
                    condition=Absent(),
                    expires_at_us=session.expires_at_us,
                )
            ]
        )
        return session

    def record_uploaded(self, env: EnvId, session: SessionId, names: Sequence[ObjectName]) -> int:
        """Record objects against an open session so they are GC roots.

        **Objects that deduplicated away are recorded too.** A hash the client
        offered and was told it already had is still content this commit will
        reach, and the collector has to know that before the ref moves —
        otherwise a sweep between the offer and the commit would take content the
        new commit depends on, and nothing would notice until a rollout failed.

        Written in append-only pages rather than one growing body: a first commit
        of a large environment offers tens of thousands of chunks, and rewriting
        that list on every batch would be quadratic as well as oversized.
        Returns the total recorded.
        """
        if not names:
            return self._count_uploaded(env, session)

        current = self.get_session(env, session)
        partition = str(env)
        pages = self._store.query(partition, keys.session_page_prefix(session), limit=100_000)
        already = {n for page in pages for n in page.body["names"]}
        fresh = [str(n) for n in dict.fromkeys(names) if str(n) not in already]
        if not fresh:
            return len(already)

        writes: list[WriteOp] = []
        page_index = len(pages)
        for start in range(0, len(fresh), _SESSION_PAGE_SIZE):
            writes.append(
                Put(
                    Key(partition, keys.session_page_sk(session, page_index)),
                    ItemKind.SESSION_PAGE,
                    {"names": fresh[start : start + _SESSION_PAGE_SIZE]},
                    condition=Absent(),
                    expires_at_us=current.expires_at_us,
                )
            )
            page_index += 1
        self._store.transact_write(writes)
        return len(already) + len(fresh)

    def uploaded_objects(self, env: EnvId, session: SessionId) -> list[ObjectName]:
        """Every object recorded against a session, in the order recorded."""
        pages = self._store.query(str(env), keys.session_page_prefix(session), limit=100_000)
        return [ObjectName.parse(n) for page in pages for n in page.body["names"]]

    def _count_uploaded(self, env: EnvId, session: SessionId) -> int:
        return len(self.uploaded_objects(env, session))

    def renew_session(
        self, env: EnvId, session: SessionId, *, ttl_us: int = WRITE_LEASE_TTL_US
    ) -> WriteSession:
        """Push an open session's expiry out from *now*.

        The pages it has already recorded are re-stamped with the new expiry too.
        Leaving them on the old one would expire the record of what the lease
        protects while the lease itself was still open — the objects would stay
        nominally leased and the collector would no longer be able to see which
        ones, which is the same as not protecting them.

        A session that has already expired is not renewable: what it protected
        may already be collected, so handing back a fresh lease would report a
        safety that no longer exists.
        """
        current = self.get_session(env, session)
        now = self._clock.now_us()
        if current.expires_at_us <= now:
            raise NotFound("write session has expired", session=str(session))

        renewed = replace(current, expires_at_us=now + ttl_us)
        partition = str(env)
        pages = self._store.query(partition, keys.session_page_prefix(session), limit=100_000)
        self._store.transact_write(
            [
                Put(
                    Key(partition, keys.session_sk(session)),
                    ItemKind.SESSION,
                    renewed.to_body(),
                    expires_at_us=renewed.expires_at_us,
                ),
                *[
                    Put(
                        page.key,
                        ItemKind.SESSION_PAGE,
                        page.body,
                        expires_at_us=renewed.expires_at_us,
                    )
                    for page in pages
                ],
            ]
        )
        return renewed

    def get_session(self, env: EnvId, session: SessionId) -> WriteSession:
        item = self._store.get(Key(str(env), keys.session_sk(session)))
        if item is None:
            raise NotFound("write session does not exist", session=str(session))
        return WriteSession.from_body(item.body)

    def end_session(self, env: EnvId, session: SessionId) -> None:
        """Close a session and drop its pages.

        The pages go with it: they exist only to keep uploaded objects alive
        while the write is in flight, and leaving them behind would pin content
        forever for a session nobody can reference.
        """
        partition = str(env)
        pages = self._store.query(partition, keys.session_page_prefix(session), limit=100_000)
        self._store.transact_write(
            [
                *[Delete(page.key) for page in pages],
                Delete(Key(partition, keys.session_sk(session))),
            ]
        )

    def list_sessions(self, env: EnvId) -> list[WriteSession]:
        items = self._store.query(str(env), f"{ItemKind.SESSION}#", limit=1000)
        return [WriteSession.from_body(item.body) for item in items]

    # ── ephemeral refs ───────────────────────────────────────────────────────

    def expired_refs(self, *, limit: int = 1000) -> list[tuple[EnvId, Ref]]:
        """Ephemeral refs past their TTL.

        At 0.1M active environments running ten experiments each,
        abandoned refs would otherwise accumulate forever and pin their content
        along with them.
        """
        now = self._clock.now_us()
        found: list[tuple[EnvId, Ref]] = []
        for item in _iter_expired(self._store, ItemKind.REF, now, limit):
            ref = Ref.from_body(item.body)
            if ref.lifecycle is RefLifecycle.EPHEMERAL:
                found.append((EnvId(item.key.pk), ref))
        return found


def _fingerprint(request: Mapping[str, Any]) -> str:
    """Detect "same key, different payload".

    Over the *canonical* encoding, so two spellings of the same request are the
    same fingerprint — otherwise a correct retry with reordered JSON keys would
    be reported as a client bug.
    """
    return blake3(canonical_json(request).encode()).hexdigest()


def _iter_expired(store: MetadataStore, kind: str, now_us: int, limit: int) -> list[Item]:
    scanner = getattr(store, "iter_expired", None)
    if scanner is None:  # pragma: no cover - only the SQLite store ships today
        return []
    return list(scanner(kind, now_us, limit=limit))
