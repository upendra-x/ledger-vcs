"""The content-addressed store — the only name-aware type in the system.

Everything that matters about an object's name being the hash of its bytes, and
about verifying that on the way in and the way out, happens here in two methods.

**put** enforces both. It rehashes the bytes against the name they were offered
under, *and* re-encodes them to check they are canonical. The second check is the
one nothing forces, and it is not optional:

    A client can construct a tree whose entries are out of order, hash those
    exact bytes, and offer them under that hash. The hash check passes — the
    bytes really are what they claim. But now one logical tree has two valid
    names, deduplication has silently split, and the *next* honest writer to
    build that directory stores it a second time.

**get** verifies again on delivery, so a damaged medium or a poisoned cache is an
error at the point of use rather than bad data in a rollout. This
is one of the two signals that must never be silent.

Ingest verification is not merely tidy — under global deduplication it is a
tenancy boundary. A writer able to store arbitrary bytes under a chosen name
damages more than its own environment: the next honest writer to offer that hash
is told "present", uploads nothing, and inherits content that fails verification
on every read. An unrelated environment is made permanently unreadable by a name
it never chose. Verification is what makes a hash mean the same thing to
everyone, and it is nearly free — BLAKE3 runs at gigabytes per second against a
peak of a few hundred writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from src.errors import CorruptObject, NotCanonical
from src.format.codec import decode, decode_as, encode, name_of_encoded, peek_kind, verify
from src.metrics import COUNTERS
from src.store.catalog import CatalogEntry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.clock import Clock
    from src.format.model import LedgerObject
    from src.ids import ObjectName
    from src.store.backend import RawBlobBackend
    from src.store.catalog import WriteCatalog
    from src.store.tombstone import TombstoneStore

__all__ = ["ObjectStore", "PutOutcome"]


@final
@dataclass(frozen=True, slots=True)
class PutOutcome:
    """What a write actually did.

    ``created`` being False is the normal and desirable case: it means
    deduplication worked. The write path reports these counts directly, which is
    how the "objects uploaded ÷ objects offered" metric is measured
    rather than estimated.
    """

    name: ObjectName
    created: bool
    #: The object's own length — what it *is*, and what deduplication is
    #: measured in, because it does not depend on how the bytes were packed.
    size: int
    #: What **this call** added to the medium: the occupied length on a create,
    #: and zero when the object was already there. Smaller than ``size`` when the
    #: backend compresses at rest, and reported separately rather
    #: than folded into it — one is a fact about the content and the other about
    #: this deployment, and conflating them makes "is compression paying for
    #: itself" unanswerable.
    #:
    #: Zero rather than the object's occupied length on a dedup hit, because
    #: every caller wants "what did this cost me". Reporting the existing size
    #: made a re-ingest look like it had stored the whole corpus again, and every
    #: call site had grown the same ``if created else 0`` guard to undo it.
    stored_size: int = 0


@final
class ObjectStore:
    """Immutable, content-addressed objects.

    Deliberately ``@final`` and not an interface. There is nothing to swap here:
    the pluggable parts are the *backend* (filesystem now, S3 later), the
    catalog and the tombstone store, all injected. Making this itself an
    abstraction would create a second place where verification could be omitted,
    which is precisely the hole content addressing cannot afford.
    """

    __slots__ = ("_backend", "_catalog", "_clock", "_tombstones")

    def __init__(
        self,
        backend: RawBlobBackend,
        *,
        catalog: WriteCatalog,
        tombstones: TombstoneStore,
        clock: Clock,
    ) -> None:
        """All collaborators are required, and ``tombstones`` deliberately has no
        default.

        A default would let a production wiring silently get a no-op tombstone
        store, which reintroduces the resurrection bug in ``store.tombstone``
        with no symptom until a rollout fails half an hour in.
        """
        self._backend = backend
        self._catalog = catalog
        self._tombstones = tombstones
        self._clock = clock

    # ── writing ──────────────────────────────────────────────────────────────

    def put(self, name: ObjectName, framed: bytes) -> PutOutcome:
        """Store bytes under the name the caller claims for them.

        Rejects a mismatch (``CorruptObject``) and a non-canonical encoding
        (``NotCanonical``). Idempotent: storing the same object twice is a no-op,
        which is what makes the whole upload phase of the write protocol safe to
        retry without any idempotency machinery.
        """
        actual = name_of_encoded(framed)
        if actual != name:
            raise CorruptObject(
                "offered bytes do not hash to the name they were sent under",
                claimed=str(name),
                actual=str(actual),
                length=len(framed),
            )

        # Re-encode to prove canonicality. `decode` already rejects every
        # non-canonical spelling, so this is belt-and-braces against a decoder
        # that accepts something `encode` would not reproduce — the exact
        # asymmetry that would let one object have two names.
        obj = decode(framed)
        if encode(obj) != framed:
            raise NotCanonical(
                "bytes hash correctly but are not the canonical encoding of what "
                "they mean; accepting them would give this object a second name",
                name=str(name),
            )

        return self.put_encoded(name, framed)

    def put_object(self, obj: LedgerObject) -> PutOutcome:
        """Encode and store, computing the name. For server-side construction —
        the image layout builder and the importer.
        """
        framed = encode(obj)
        return self.put_encoded(name_of_encoded(framed), framed)

    def put_encoded(self, name: ObjectName, framed: bytes) -> PutOutcome:
        """Store bytes **this process just produced**, without re-proving them.

        ``put`` rehashes the bytes and then decodes and re-encodes them to prove
        the encoding is canonical. Both checks exist for bytes that arrived from
        a caller, where a deliberately non-canonical spelling would give one
        logical object two names and split deduplication for everybody.

        Neither can fail for bytes that came out of ``encode`` in this process:
        the name was derived from these exact bytes a moment ago, and ``encode``
        only ever produces the canonical spelling. Re-deriving both is pure cost,
        and it is not a small one — every interior tree and blob-index node goes
        through here, so a large ingest was encoding each of them three times and
        decoding once.

        **Never call this with bytes a caller supplied.** ``put`` is the door for
        those, and the HTTP object route uses it.
        """
        key = _key_for(name)
        if self._backend.exists_many([key]):
            # Present already. Still clear any tombstone: if this object was
            # swept and is now being written again, leaving the tombstone would
            # make it permanently 'missing' and the write could never converge.
            self._tombstones.clear([name])
            return PutOutcome(name=name, created=False, size=len(framed), stored_size=0)

        occupied = self._backend.write(key, framed)
        COUNTERS.increment("ledger_objects_uploaded_total")
        self._tombstones.clear([name])
        self._catalog.record(
            [
                CatalogEntry(
                    # Physical, because the catalog is the "stored" side of the
                    # collection diff and of every cost number: what a sweep
                    # returns is medium bytes, not logical ones.
                    name=name,
                    size=occupied,
                    kind=peek_kind(framed).value,
                    written_at_us=self._clock.now_us(),
                )
            ]
        )
        return PutOutcome(name=name, created=True, size=len(framed), stored_size=occupied)

    # ── reading ──────────────────────────────────────────────────────────────

    def get(self, name: ObjectName) -> bytes:
        """Fetch and verify. Raises ``CorruptObject`` if the bytes have rotted."""
        COUNTERS.increment("ledger_object_reads_total")
        framed = self._backend.read(_key_for(name))
        try:
            verify(name, framed)
        except CorruptObject:
            # Never downgraded to a miss: a miss invites a retry, while this
            # means a storage medium or a cache is actively wrong and someone
            # has to look at it. Counted, not gauged — a gauge would
            # read zero on the next scrape and nobody would know it happened.
            COUNTERS.increment("ledger_object_verification_failures_total")
            raise
        return framed

    def get_object(self, name: ObjectName) -> LedgerObject:
        return decode(self.get(name))

    def get_as[T: LedgerObject](self, name: ObjectName, expected: type[T]) -> T:
        """Fetch, verify, and require a particular object type.

        Worth having because a tree entry with the wrong ``kind`` points at an
        object of an unexpected type, and that should be a clear error at the
        fetch rather than an ``AttributeError`` during materialization.
        """
        return decode_as(self.get(name), expected)

    def get_range(self, name: ObjectName, offset: int, length: int) -> bytes:
        """Read part of an object's *payload*, skipping the frame header.

        Verification here is by whole object, because chunks are capped at 4 MiB
        and a ranged read descends to whole chunks anyway. BLAKE3's
        internal Merkle tree would allow verifying a range without the whole
        object, which is what a multi-gigabyte single object would need — and
        the reason no such object exists.
        """
        from src.format.codec import HEADER_BYTES

        framed = self.get(name)
        payload_start = HEADER_BYTES + offset
        return framed[payload_start : payload_start + length]

    # ── existence: the single predicate ──────────────────────────────────────

    def missing(self, names: Sequence[ObjectName]) -> frozenset[ObjectName]:
        """Which of ``names`` the client must upload.

        **This is the one existence predicate in the system, and its polarity is
        deliberate.** It answers `missing`, not `present`, because the write path
        asks "what do I still owe you" — and because getting the sense backwards
        in one of several copies would either publish commits whose content was
        never stored, or deadlock every write. There is one implementation, and
        this is it.

            missing(n)  ⟺  not stored(n)  or  tombstoned(n)

        The tombstone term is what stops deduplication from resurrecting an
        object whose children have already been swept (see ``store.tombstone``).
        """
        if not names:
            return frozenset()

        COUNTERS.increment("ledger_objects_offered_total", len(names))
        unique = list(dict.fromkeys(names))
        present_keys = self._backend.exists_many([_key_for(n) for n in unique])
        absent = {n for n in unique if _key_for(n) not in present_keys}
        return frozenset(absent | self._tombstones.filter_tombstoned(unique))

    # ── administration ───────────────────────────────────────────────────────

    def delete(self, names: Iterable[ObjectName]) -> int:
        """Remove objects. Only the collector should call this.

        Recording tombstones is the *collector's* responsibility rather than
        this method's, because the expiry depends on the epoch that swept them —
        and burying that policy here would make it invisible.
        """
        materialised = list(names)
        removed = self._backend.delete([_key_for(n) for n in materialised])
        self._catalog.forget(materialised)
        return removed

    @property
    def catalog(self) -> WriteCatalog:
        return self._catalog

    @property
    def tombstones(self) -> TombstoneStore:
        return self._tombstones


def _key_for(name: ObjectName) -> str:
    """An object's backend key is its own hash — the "the
    name IS the
    address", which is why a standalone object needs no index entry at all.
    """
    return name.hex
