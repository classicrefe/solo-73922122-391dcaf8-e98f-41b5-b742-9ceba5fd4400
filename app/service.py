"""Domain service: validate, persist, recompute and query energy intervals.

Invariants
----------
* A batch is applied atomically.  Every conflict is detected during validation,
  before any write, so a rejected batch leaves the database untouched.
* ``(source_record_id)`` is unique.  Re-sending identical content is an
  idempotent no-op; re-sending different content rejects the whole batch.
* ``(meter_id, read_at_us)`` is unique.  Two different source records for the
  same meter and instant reject the whole batch.
* Intervals are always derived from the persisted readings of the commit that
  wrote them.  They are recomputed for every meter touched by a mutation.
* Recomputing a meter means deleting and re-creating *all* of its intervals
  inside the same write transaction.  That makes the interval table a pure
  function of the readings table: after any crash or restart there is no
  possibility of a half-updated interval set.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from .config import Settings
from .db import (
    bump_commit_seq,
    connection,
    current_commit_seq,
    read_transaction,
    write_transaction,
)
from .models import (
    ConflictDetail,
    CorrectionRecordResult,
    CorrectionResponse,
    IntervalOut,
    MeterStateResponse,
    MeterSummary,
    MutationResponse,
    QueryWindow,
    ReadingIn,
    ReadingOut,
    RecordResult,
)
from .timeutil import (
    MAX_EPOCH_US,
    MIN_EPOCH_US,
    isoformat_utc,
    microseconds_to_iso,
    normalize_utc,
    to_microseconds,
)

BREAK_METER_REPLACED = "meter_replaced"
BREAK_COUNTER_ROLLBACK = "counter_rollback"

# Temporary instants used while a time correction moves a batch out of the
# way of its own target slots.  They lie far below MIN_EPOCH_US and therefore
# never collide with a representable stored instant; the values are invisible
# to readers because the whole move is one transaction.
_PARKING_BASE = -1_000_000_000_000_000_000


class BatchRejected(Exception):
    """The whole request is refused; nothing was written."""

    def __init__(self, message: str, conflicts: Sequence[ConflictDetail]):
        super().__init__(message)
        self.message = message
        self.conflicts = list(conflicts)


def _hash_content(
    meter_id: str, read_at_us: int, reading: int, meter_replaced: bool
) -> str:
    payload = json.dumps(
        {
            "meter_id": meter_id,
            "read_at_us": read_at_us,
            "reading": reading,
            "meter_replaced": bool(meter_replaced),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content_hash(reading: ReadingIn) -> str:
    return _hash_content(
        reading.meter_id,
        to_microseconds(reading.read_at),
        reading.reading,
        bool(reading.meter_replaced),
    )


@dataclass
class _Prepared:
    reading: ReadingIn
    read_at_us: int
    content_hash: str


@dataclass
class _Validation:
    to_insert: list[_Prepared] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    conflicts: list[ConflictDetail] = field(default_factory=list)


def _existing_by_source(
    conn: sqlite3.Connection, source_ids: Iterable[str]
) -> dict[str, sqlite3.Row]:
    ids = list(dict.fromkeys(source_ids))
    found: dict[str, sqlite3.Row] = {}
    chunk = 400
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        placeholders = ",".join("?" for _ in part)
        rows = conn.execute(
            f"SELECT id, source_record_id, meter_id, read_at_us, reading, "
            f"meter_replaced, content_hash, original_content_hash "
            f"FROM readings "
            f"WHERE source_record_id IN ({placeholders})",
            part,
        ).fetchall()
        for row in rows:
            found[row["source_record_id"]] = row
    return found


def _validate_batch(
    conn: sqlite3.Connection, readings: Sequence[ReadingIn]
) -> _Validation:
    result = _Validation()

    # ---- duplicates inside the request itself -------------------------
    by_source: dict[str, _Prepared] = {}
    by_slot: dict[tuple[str, int], _Prepared] = {}
    for reading in readings:
        prepared = _Prepared(
            reading=reading,
            read_at_us=to_microseconds(reading.read_at),
            content_hash=_content_hash(reading),
        )
        earlier = by_source.get(reading.source_record_id)
        if earlier is not None:
            if earlier.content_hash == prepared.content_hash:
                # Exact duplicate in the same batch: collapse to one.
                continue
            result.conflicts.append(
                ConflictDetail(
                    type="source_record_conflict",
                    source_record_id=reading.source_record_id,
                    conflicting_source_record_id=reading.source_record_id,
                    detail=(
                        "同一源记录号在本批内出现多次且内容不同"
                    ),
                )
            )
            continue
        by_source[reading.source_record_id] = prepared

        slot = (reading.meter_id, prepared.read_at_us)
        same_slot = by_slot.get(slot)
        if same_slot is not None:
            # Two distinct source records for one meter at one instant — even
            # with identical content — are an ambiguous input; reject.
            result.conflicts.append(
                ConflictDetail(
                    type="meter_time_conflict",
                    source_record_id=reading.source_record_id,
                    conflicting_source_record_id=same_slot.reading.source_record_id,
                    detail=(
                        f"电表 {reading.meter_id} 在 "
                        f"{isoformat_utc(reading.read_at)} 已有另一条本批读数"
                    ),
                )
            )
            continue
        by_slot[slot] = prepared

    if result.conflicts:
        return result

    # ---- compare with stored readings ---------------------------------
    existing = _existing_by_source(conn, by_source.keys())
    for source_id, prepared in by_source.items():
        stored = existing.get(source_id)
        if stored is not None:
            # A reading that was time-corrected accepts either its current
            # content (the corrected instant) or the content it was originally
            # submitted with.  Retransmitting the pre-correction gateway
            # message therefore stays an idempotent no-op and must not move
            # the reading back.
            same_as_stored = (
                stored["meter_id"] == prepared.reading.meter_id
                and stored["content_hash"] == prepared.content_hash
            )
            same_as_original = (
                stored["meter_id"] == prepared.reading.meter_id
                and stored["original_content_hash"] is not None
                and stored["original_content_hash"] == prepared.content_hash
            )
            if same_as_stored or same_as_original:
                result.unchanged.append(source_id)
            else:
                result.conflicts.append(
                    ConflictDetail(
                        type="source_record_conflict",
                        source_record_id=source_id,
                        conflicting_source_record_id=source_id,
                        detail="源记录号已存在但内容不同",
                    )
                )
            continue
        result.to_insert.append(prepared)

    if result.conflicts:
        return result

    # ---- (meter, instant) against stored readings ----------------------
    for prepared in result.to_insert:
        reading = prepared.reading
        row = conn.execute(
            "SELECT source_record_id FROM readings "
            "WHERE meter_id = ? AND read_at_us = ?",
            (reading.meter_id, prepared.read_at_us),
        ).fetchone()
        if row is not None:
            result.conflicts.append(
                ConflictDetail(
                    type="meter_time_conflict",
                    source_record_id=reading.source_record_id,
                    conflicting_source_record_id=row["source_record_id"],
                    detail=(
                        f"电表 {reading.meter_id} 在 "
                        f"{isoformat_utc(reading.read_at)} 已有读数 "
                        f"{row['source_record_id']}"
                    ),
                )
            )
    return result


def _recompute_intervals(
    conn: sqlite3.Connection, meter_id: str, commit_seq: int
) -> None:
    """Rebuild every interval of one meter from its persisted readings."""
    conn.execute("DELETE FROM intervals WHERE meter_id = ?", (meter_id,))
    rows = conn.execute(
        "SELECT id, read_at_us, reading, meter_replaced "
        "FROM readings WHERE meter_id = ? ORDER BY read_at_us ASC",
        (meter_id,),
    ).fetchall()

    payload: list[tuple] = []
    previous: sqlite3.Row | None = None
    for row in rows:
        if previous is None:
            previous = row
            continue
        if bool(row["meter_replaced"]):
            is_break, usage, reason = 1, None, BREAK_METER_REPLACED
        elif row["reading"] < previous["reading"]:
            is_break, usage, reason = 1, None, BREAK_COUNTER_ROLLBACK
        else:
            is_break, usage, reason = 0, row["reading"] - previous["reading"], None
        payload.append(
            (
                meter_id,
                previous["id"],
                row["id"],
                previous["read_at_us"],
                row["read_at_us"],
                previous["reading"],
                row["reading"],
                usage,
                is_break,
                reason,
                commit_seq,
            )
        )
        previous = row

    if payload:
        conn.executemany(
            "INSERT INTO intervals (meter_id, from_reading_id, to_reading_id, "
            "from_at_us, to_at_us, from_value, to_value, usage, is_break, "
            "break_reason, commit_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            payload,
        )


def _meter_summary(conn: sqlite3.Connection, meter_id: str) -> MeterSummary:
    readings = conn.execute(
        "SELECT COUNT(*) AS n, MIN(read_at_us) AS first_at, "
        "MAX(read_at_us) AS last_at FROM readings WHERE meter_id = ?",
        (meter_id,),
    ).fetchone()
    intervals = conn.execute(
        "SELECT COUNT(*) AS n, "
        "COALESCE(SUM(is_break = 0), 0) AS usable, "
        "COALESCE(SUM(is_break = 1), 0) AS breaks, "
        "COALESCE(SUM(usage), 0) AS total "
        "FROM intervals WHERE meter_id = ?",
        (meter_id,),
    ).fetchone()
    reading_count = int(readings["n"])
    return MeterSummary(
        meter_id=meter_id,
        reading_count=reading_count,
        interval_count=int(intervals["n"]),
        usage_interval_count=int(intervals["usable"]),
        break_count=int(intervals["breaks"]),
        usage_total=int(intervals["total"]),
        first_read_at=(
            microseconds_to_iso(readings["first_at"])
            if reading_count
            else None
        ),
        last_read_at=(
            microseconds_to_iso(readings["last_at"])
            if reading_count
            else None
        ),
    )


def _mutation_response(
    conn: sqlite3.Connection,
    commit_seq: int,
    changed: bool,
    records: list[RecordResult],
    meters: Iterable[str],
) -> MutationResponse:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.status] += 1
    summaries = [
        _meter_summary(conn, meter_id) for meter_id in sorted(set(meters))
    ]
    return MutationResponse(
        commit_seq=commit_seq,
        changed=changed,
        counts=dict(counts),
        records=records,
        affected_meters=summaries,
    )


def ingest(
    settings: Settings, readings: Sequence[ReadingIn]
) -> MutationResponse:
    """Atomically insert a batch of readings and refresh affected intervals."""
    with connection(settings) as conn:
        with write_transaction(conn):
            validation = _validate_batch(conn, readings)
            if validation.conflicts:
                raise BatchRejected("整批读数被拒绝", validation.conflicts)

            records = [
                RecordResult(
                    source_record_id=source_id, status="unchanged"
                )
                for source_id in validation.unchanged
            ]
            if not validation.to_insert:
                # Pure idempotent retry: nothing to write, no new commit.
                return _mutation_response(
                    conn,
                    current_commit_seq(conn),
                    changed=False,
                    records=records,
                    meters=[],
                )

            commit_seq = bump_commit_seq(conn)
            meters: set[str] = set()
            for prepared in validation.to_insert:
                reading = prepared.reading
                conn.execute(
                    "INSERT INTO readings (source_record_id, meter_id, "
                    "read_at_us, reading, meter_replaced, content_hash, "
                    "original_content_hash, created_commit_seq, "
                    "updated_commit_seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        reading.source_record_id,
                        reading.meter_id,
                        prepared.read_at_us,
                        reading.reading,
                        1 if reading.meter_replaced else 0,
                        prepared.content_hash,
                        prepared.content_hash,
                        commit_seq,
                    ),
                )
                meters.add(reading.meter_id)
                records.append(
                    RecordResult(
                        source_record_id=reading.source_record_id,
                        status="created",
                    )
                )
            for meter_id in sorted(meters):
                _recompute_intervals(conn, meter_id, commit_seq)

            return _mutation_response(
                conn,
                commit_seq,
                changed=True,
                records=records,
                meters=meters,
            )


def correct_times(
    settings: Settings,
    source_record_ids: Sequence[str],
    offset_seconds: int,
    expected_commit_seq: int,
) -> CorrectionResponse:
    """Atomically move a batch of readings by one UTC offset.

    Semantics
    ---------
    * All listed readings move by the same ``offset_seconds``.  The batch is
      validated as a whole against the *final* state: a target instant may be
      occupied by another reading of the same batch, as long as that reading
      moves away too and the final instants are unique.
    * Rejected as a whole (409) when a listed source record is missing, the
      expected commit sequence is stale, a moved instant leaves the supported
      UTC range, or a final instant collides with a reading outside the batch.
      Nothing is written on rejection.
    * A reading whose target equals its current instant is reported as
      ``unchanged`` (only possible if the offset is 0, which the request
      schema rejects; kept so the service layer stays total).  If every listed
      reading is already there, no commit is made.
    * The original content fingerprint is preserved, so a retransmission of
      the pre-correction gateway message stays an idempotent no-op.
    """
    ids = list(dict.fromkeys(source_record_ids))
    delta_us = int(offset_seconds) * 1_000_000

    with connection(settings) as conn:
        with write_transaction(conn):
            commit_seq = current_commit_seq(conn)
            if expected_commit_seq != commit_seq:
                raise BatchRejected(
                    "提交序号已过期，整批校时被拒绝",
                    [
                        ConflictDetail(
                            type="stale_commit_seq",
                            source_record_id=None,
                            conflicting_source_record_id=None,
                            detail=(
                                f"请求基于提交序号 {expected_commit_seq}，"
                                f"当前已提交序号为 {commit_seq}；"
                                f"期间已有其他提交，请重新查询后再试"
                            ),
                        )
                    ],
                )

            existing = _existing_by_source(conn, ids)
            conflicts: list[ConflictDetail] = []

            # ---- every listed record must exist ------------------------
            for source_id in ids:
                if source_id not in existing:
                    conflicts.append(
                        ConflictDetail(
                            type="record_not_found",
                            source_record_id=source_id,
                            conflicting_source_record_id=None,
                            detail=f"源记录 {source_id} 不存在，无法校时",
                        )
                    )
            if conflicts:
                raise BatchRejected("整批校时被拒绝", conflicts)

            # ---- compute the final instant of every listed reading -----
            moved: dict[str, tuple[sqlite3.Row, int]] = {}
            for source_id in ids:
                row = existing[source_id]
                current_us = int(row["read_at_us"])
                target_us = current_us + delta_us
                if target_us < MIN_EPOCH_US or target_us > MAX_EPOCH_US:
                    conflicts.append(
                        ConflictDetail(
                            type="time_out_of_range",
                            source_record_id=source_id,
                            conflicting_source_record_id=None,
                            detail=(
                                f"源记录 {source_id} 校正后时刻 "
                                f"{_render_epoch(target_us)} 超出支持的 UTC 范围"
                            ),
                        )
                    )
                moved[source_id] = (row, target_us)
            if conflicts:
                raise BatchRejected("整批校时被拒绝", conflicts)

            # ---- final (meter, instant) must be unique within the batch ---
            # With a uniform offset two readings of one meter can only collide
            # if they already shared an instant, which the unique index
            # forbids.  The check is kept as a defensive invariant: the batch
            # is accepted on its *final* slots, never on intermediate states.
            final_slot_owner: dict[tuple[str, int], str] = {}
            for source_id in ids:
                row, target_us = moved[source_id]
                slot = (row["meter_id"], target_us)
                other = final_slot_owner.get(slot)
                if other is not None:
                    conflicts.append(
                        ConflictDetail(
                            type="meter_time_conflict",
                            source_record_id=source_id,
                            conflicting_source_record_id=other,
                            detail=(
                                f"电表 {row['meter_id']} 校正后两条记录落在同一时刻 "
                                f"{_render_epoch(target_us)}"
                            ),
                        )
                    )
                else:
                    final_slot_owner[slot] = source_id
            if conflicts:
                raise BatchRejected("整批校时被拒绝", conflicts)

            # Readings *outside* the batch keep their instants, so a member's
            # target slot must not be owned by one of them.  Slots currently
            # owned by batch members are fine: those readings move away too,
            # which is exactly the "final state" rule.
            outside_slots = _slots_outside_batch(conn, ids)
            for source_id in ids:
                row, target_us = moved[source_id]
                slot = (row["meter_id"], target_us)
                other = outside_slots.get(slot)
                if other is not None:
                    conflicts.append(
                        ConflictDetail(
                            type="meter_time_conflict",
                            source_record_id=source_id,
                            conflicting_source_record_id=other,
                            detail=(
                                f"电表 {row['meter_id']} 校正后时刻 "
                                f"{_render_epoch(target_us)} "
                                f"已有批外读数 {other}"
                            ),
                        )
                    )
            if conflicts:
                raise BatchRejected("整批校时被拒绝", conflicts)

            # ---- apply ------------------------------------------------
            changed = False
            records: list[CorrectionRecordResult] = []
            meters: set[str] = set()
            # The order of UPDATEs must not trip the (meter_id, read_at_us)
            # unique index: move every reading out of the way first, then to
            # its final instant.  Everything happens in one transaction, so
            # no observer ever sees the intermediate state.
            pending = []
            for source_id in ids:
                row, target_us = moved[source_id]
                current_us = int(row["read_at_us"])
                if current_us == target_us:
                    continue
                pending.append((source_id, row, current_us, target_us))
                meters.add(row["meter_id"])

            if pending:
                changed = True
                new_commit_seq = bump_commit_seq(conn)
                # Step 1: park the batch below MIN_EPOCH_US, where no stored
                # instant can live, so the moves cannot trip the
                # (meter_id, read_at_us) unique index while the batch is being
                # rearranged.
                for _source_id, row, _current_us, _target_us in pending:
                    conn.execute(
                        "UPDATE readings SET read_at_us = ? WHERE id = ?",
                        (_PARKING_BASE - int(row["id"]), int(row["id"])),
                    )
                # Step 2: move each reading to its final instant.
                try:
                    for source_id, row, current_us, target_us in pending:
                        conn.execute(
                            "UPDATE readings SET read_at_us = ?, "
                            "content_hash = ?, updated_commit_seq = ? "
                            "WHERE id = ?",
                            (
                                target_us,
                                _hash_content(
                                    row["meter_id"],
                                    target_us,
                                    int(row["reading"]),
                                    bool(row["meter_replaced"]),
                                ),
                                new_commit_seq,
                                int(row["id"]),
                            ),
                        )
                except sqlite3.IntegrityError as exc:
                    # The unique index is the last line of defence; a race or
                    # a hand-edited database must still reject the whole batch.
                    raise BatchRejected(
                        "整批校时被拒绝",
                        [
                            ConflictDetail(
                                type="meter_time_conflict",
                                source_record_id=None,
                                conflicting_source_record_id=None,
                                detail=(
                                    "校正后的时刻与已有读数冲突，整批已回滚: "
                                    f"{exc}"
                                ),
                            )
                        ],
                    ) from exc
                for meter_id in sorted(meters):
                    _recompute_intervals(conn, meter_id, new_commit_seq)
            else:
                new_commit_seq = commit_seq

            for source_id in ids:
                row, target_us = moved[source_id]
                current_us = int(row["read_at_us"])
                if current_us == target_us:
                    status = "unchanged"
                else:
                    status = "corrected"
                records.append(
                    CorrectionRecordResult(
                        source_record_id=source_id,
                        status=status,
                        previous_read_at=microseconds_to_iso(current_us),
                        read_at=microseconds_to_iso(target_us),
                        moved=(status == "corrected"),
                    )
                )

            counts: dict[str, int] = defaultdict(int)
            for record in records:
                counts[record.status] += 1
            summaries = [
                _meter_summary(conn, meter_id) for meter_id in sorted(meters)
            ]
            return CorrectionResponse(
                commit_seq=new_commit_seq,
                changed=changed,
                offset_seconds=int(offset_seconds),
                counts=dict(counts),
                records=records,
                affected_meters=summaries,
            )


def _render_epoch(epoch_us: int) -> str:
    try:
        return microseconds_to_iso(epoch_us)
    except (OverflowError, ValueError):
        return f"epoch_us={epoch_us}"


def _slots_outside_batch(
    conn: sqlite3.Connection, source_ids: Sequence[str]
) -> dict[tuple[str, int], str]:
    """Map every (meter, instant) currently owned by a reading not in the batch."""
    ids = list(dict.fromkeys(source_ids))
    result: dict[tuple[str, int], str] = {}
    chunk = 400
    # Collect the distinct meters touched by the batch first.
    meters: set[str] = set()
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        placeholders = ",".join("?" for _ in part)
        rows = conn.execute(
            f"SELECT DISTINCT meter_id FROM readings "
            f"WHERE source_record_id IN ({placeholders})",
            part,
        ).fetchall()
        meters.update(row["meter_id"] for row in rows)
    if not meters:
        return result
    for meter_id in meters:
        rows = conn.execute(
            "SELECT source_record_id, read_at_us FROM readings "
            "WHERE meter_id = ?",
            (meter_id,),
        ).fetchall()
        for row in rows:
            if row["source_record_id"] in ids:
                continue
            result[(meter_id, int(row["read_at_us"]))] = row["source_record_id"]
    return result


def undo(
    settings: Settings, source_record_ids: Sequence[str]
) -> MutationResponse:
    """Atomically remove readings by source record id and refresh intervals."""
    ids = list(dict.fromkeys(source_record_ids))
    with connection(settings) as conn:
        with write_transaction(conn):
            existing = _existing_by_source(conn, ids)
            records: list[RecordResult] = []
            meters: set[str] = set()
            to_delete: list[int] = []
            for source_id in ids:
                row = existing.get(source_id)
                if row is None:
                    records.append(
                        RecordResult(
                            source_record_id=source_id, status="not_found"
                        )
                    )
                    continue
                to_delete.append(int(row["id"]))
                meters.add(row["meter_id"])
                records.append(
                    RecordResult(
                        source_record_id=source_id, status="removed"
                    )
                )

            if not to_delete:
                return _mutation_response(
                    conn,
                    current_commit_seq(conn),
                    changed=False,
                    records=records,
                    meters=[],
                )

            commit_seq = bump_commit_seq(conn)
            # intervals reference readings with ON DELETE CASCADE; the affected
            # meters are rebuilt below from the remaining readings.
            conn.executemany(
                "DELETE FROM intervals WHERE from_reading_id = ? "
                "OR to_reading_id = ?",
                [(rid, rid) for rid in to_delete],
            )
            conn.executemany(
                "DELETE FROM readings WHERE id = ?", [(rid,) for rid in to_delete]
            )
            for meter_id in sorted(meters):
                _recompute_intervals(conn, meter_id, commit_seq)

            return _mutation_response(
                conn,
                commit_seq,
                changed=True,
                records=records,
                meters=meters,
            )


def list_meters(settings: Settings) -> list[MeterSummary]:
    """Summarise every meter that currently has at least one reading."""
    with connection(settings) as conn:
        with read_transaction(conn):
            rows = conn.execute(
                "SELECT DISTINCT meter_id FROM readings ORDER BY meter_id"
            ).fetchall()
            return [_meter_summary(conn, row["meter_id"]) for row in rows]


def _parse_window_boundary(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} 不是合法的 ISO-8601 时刻: {raw!r}") from exc
    return to_microseconds(normalize_utc(parsed))


def _interval_out(row: sqlite3.Row, from_source: str, to_source: str) -> IntervalOut:
    return IntervalOut(
        from_source_record_id=from_source,
        to_source_record_id=to_source,
        from_read_at=microseconds_to_iso(row["from_at_us"]),
        to_read_at=microseconds_to_iso(row["to_at_us"]),
        from_reading=int(row["from_value"]),
        to_reading=int(row["to_value"]),
        usage=None if row["usage"] is None else int(row["usage"]),
        is_break=bool(row["is_break"]),
        break_reason=row["break_reason"],
        commit_seq=int(row["commit_seq"]),
    )


def _reading_out(row: sqlite3.Row) -> ReadingOut:
    return ReadingOut(
        source_record_id=row["source_record_id"],
        read_at=microseconds_to_iso(row["read_at_us"]),
        reading=int(row["reading"]),
        meter_replaced=bool(row["meter_replaced"]),
        commit_seq=int(
            row["updated_commit_seq"]
            if row["updated_commit_seq"] is not None
            else row["created_commit_seq"]
        ),
    )


def query_meter(
    settings: Settings,
    meter_id: str,
    from_raw: str | None = None,
    to_raw: str | None = None,
    include_intervals: bool = True,
) -> MeterStateResponse | None:
    """Read one consistent snapshot of a meter's readings and intervals."""
    from_us = _parse_window_boundary(from_raw, "from")
    to_us = _parse_window_boundary(to_raw, "to")
    if from_us is not None and to_us is not None and from_us > to_us:
        raise ValueError("from 必须早于或等于 to")
    windowed = from_us is not None or to_us is not None

    with connection(settings) as conn:
        with read_transaction(conn):
            commit_seq = current_commit_seq(conn)
            meter_exists = conn.execute(
                "SELECT 1 FROM readings WHERE meter_id = ? LIMIT 1",
                (meter_id,),
            ).fetchone()
            if meter_exists is None:
                return None

            clauses = ["meter_id = ?"]
            params: list[object] = [meter_id]
            if from_us is not None:
                clauses.append("read_at_us >= ?")
                params.append(from_us)
            if to_us is not None:
                clauses.append("read_at_us <= ?")
                params.append(to_us)
            reading_rows = conn.execute(
                "SELECT source_record_id, read_at_us, reading, meter_replaced, "
                "created_commit_seq, updated_commit_seq "
                f"FROM readings WHERE {' AND '.join(clauses)} "
                "ORDER BY read_at_us ASC",
                params,
            ).fetchall()
            readings = [_reading_out(row) for row in reading_rows]

            interval_outs: list[IntervalOut] = []
            if include_intervals:
                # Only intervals fully inside the requested window are returned,
                # so every interval endpoint is also present in `readings`.
                i_clauses = ["i.meter_id = ?"]
                i_params: list[object] = [meter_id]
                if from_us is not None:
                    i_clauses.append("i.from_at_us >= ?")
                    i_params.append(from_us)
                if to_us is not None:
                    i_clauses.append("i.to_at_us <= ?")
                    i_params.append(to_us)
                interval_rows = conn.execute(
                    "SELECT i.from_at_us, i.to_at_us, i.from_value, i.to_value, "
                    "i.usage, i.is_break, i.break_reason, i.commit_seq, "
                    "f.source_record_id AS from_source, "
                    "t.source_record_id AS to_source "
                    "FROM intervals i "
                    "JOIN readings f ON f.id = i.from_reading_id "
                    "JOIN readings t ON t.id = i.to_reading_id "
                    f"WHERE {' AND '.join(i_clauses)} "
                    "ORDER BY i.to_at_us ASC",
                    i_params,
                ).fetchall()
                interval_outs = [
                    _interval_out(row, row["from_source"], row["to_source"])
                    for row in interval_rows
                ]

            breaks = [item for item in interval_outs if item.is_break]
            if include_intervals:
                usage_total = sum(
                    item.usage for item in interval_outs
                    if not item.is_break and item.usage is not None
                )
            else:
                # Intervals were not selected; still report the meter total
                # from the same snapshot.
                total_row = conn.execute(
                    "SELECT COALESCE(SUM(usage), 0) FROM intervals "
                    "WHERE meter_id = ? AND is_break = 0"
                    + (" AND from_at_us >= ?" if from_us is not None else "")
                    + (" AND to_at_us <= ?" if to_us is not None else ""),
                    [
                        value
                        for value in (meter_id, from_us, to_us)
                        if value is not None
                    ],
                ).fetchone()
                usage_total = int(total_row[0])
            return MeterStateResponse(
                meter_id=meter_id,
                commit_seq=commit_seq,
                window=QueryWindow(
                    from_=(
                        microseconds_to_iso(from_us)
                        if from_us is not None
                        else None
                    ),
                    to=(
                        microseconds_to_iso(to_us) if to_us is not None else None
                    ),
                    applied=windowed,
                ),
                readings=readings,
                intervals=interval_outs,
                breaks=breaks,
                usage_total=int(usage_total),
            )
