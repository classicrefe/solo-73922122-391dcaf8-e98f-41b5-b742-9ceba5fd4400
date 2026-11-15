"""Functional tests for batch time correction."""

from __future__ import annotations

import pytest

from app.db import connection
from datetime import timedelta

from tests.common import BASE, assert_db_invariants, assert_meter_consistent


def at(second: int) -> str:
    return (BASE + timedelta(seconds=second)).isoformat().replace("+00:00", "Z")


def reading(meter: str, source: str, second: int, value: int, replaced: bool = False) -> dict:
    return {
        "meter_id": meter,
        "source_record_id": source,
        "read_at": at(second),
        "reading": value,
        "meter_replaced": replaced,
    }


def post(client, readings):
    return client.post("/api/v1/readings", json={"readings": readings})


def correct(client, ids, offset, seq):
    return client.post(
        "/api/v1/readings/correct-time",
        json={
            "source_record_ids": ids,
            "offset_seconds": offset,
            "expected_commit_seq": seq,
        },
    )


def undo(client, ids):
    return client.post("/api/v1/readings/undo", json={"source_record_ids": ids})


def get(client, meter, **params):
    return client.get(f"/api/v1/meters/{meter}", params=params)


def seq_of(client, meter="M1"):
    return get(client, meter).json()["commit_seq"]


def test_correction_moves_times_and_recomputes_intervals(client):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 60, 160),
            reading("M1", "c", 120, 250),
        ],
    )
    seq = seq_of(client)

    resp = correct(client, ["b", "c"], 300, seq)  # +5 minutes
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["changed"] is True
    assert body["offset_seconds"] == 300
    assert body["counts"] == {"corrected": 2}
    assert body["commit_seq"] > seq
    assert [r["source_record_id"] for r in body["records"]] == ["b", "c"]
    assert [r["previous_read_at"] for r in body["records"]] == [at(60), at(120)]
    assert [r["read_at"] for r in body["records"]] == [at(360), at(420)]

    state = get(client, "M1").json()
    assert state["commit_seq"] == body["commit_seq"]
    assert [r["read_at"] for r in state["readings"]] == [at(0), at(360), at(420)]
    # a -> b is captured at 360 with 60, b -> c at 420 with 90.
    assert [i["usage"] for i in state["intervals"]] == [60, 90]
    assert state["usage_total"] == 150
    assert_meter_consistent(state)


def test_correction_reorders_readings_across_meters(client):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M2", "x", 120, 10),
            reading("M1", "b", 240, 160),
        ],
    )
    seq = get(client, "M1").json()["commit_seq"]
    resp = correct(client, ["b"], -120, seq)  # M1/b: 240 -> 120
    assert resp.status_code == 200
    state = get(client, "M1").json()
    assert [r["source_record_id"] for r in state["readings"]] == ["a", "b"]
    assert [i["usage"] for i in state["intervals"]] == [60]
    # M2 is untouched and still has no interval.
    assert get(client, "M2").json()["intervals"] == []


def test_target_slot_temporarily_held_by_another_batch_member(client):
    """Every target is occupied by a batch member that moves away too."""
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 60, 160),
            reading("M1", "c", 120, 250),
        ],
    )
    seq = seq_of(client)
    resp = correct(client, ["a", "b", "c"], 60, seq)  # 0->60, 60->120, 120->180
    assert resp.status_code == 200, resp.json()
    assert resp.json()["counts"] == {"corrected": 3}

    state = get(client, "M1").json()
    assert [r["read_at"] for r in state["readings"]] == [at(60), at(120), at(180)]
    assert [i["usage"] for i in state["intervals"]] == [60, 90]
    assert_meter_consistent(state)


def test_collision_with_reading_outside_the_batch_rejects(client, settings):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 60, 160),
            reading("M1", "c", 120, 250),
        ],
    )
    seq = seq_of(client)
    before = get(client, "M1").json()

    # a at 0 -> 60, but b (outside the batch) already sits there.
    resp = correct(client, ["a"], 60, seq)
    assert resp.status_code == 409, resp.json()
    conflicts = resp.json()["conflicts"]
    assert conflicts[0]["type"] == "meter_time_conflict"
    assert conflicts[0]["conflicting_source_record_id"] == "b"

    # Nothing changed, not even the commit sequence.
    after = get(client, "M1").json()
    assert after["commit_seq"] == before["commit_seq"] == seq
    assert after["readings"] == before["readings"]
    with connection(settings) as conn:
        assert_db_invariants(conn, "M1")


def test_missing_record_rejects_whole_batch(client, settings):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M2", "x", 0, 10),
        ],
    )
    seq = get(client, "M1").json()["commit_seq"]
    resp = correct(client, ["a", "nope"], 60, seq)
    assert resp.status_code == 409, resp.json()
    assert resp.json()["conflicts"][0]["type"] == "record_not_found"
    # Neither reading moved.
    assert get(client, "M1").json()["readings"][0]["read_at"] == at(0)
    assert get(client, "M2").json()["readings"][0]["read_at"] == at(0)


def test_stale_commit_sequence_rejects(client):
    post(client, [reading("M1", "a", 0, 100)])
    seq = seq_of(client)
    correct(client, ["a"], 60, seq)
    # The first correction consumed the sequence the second request relied on.
    stale = correct(client, ["a"], 60, seq)
    assert stale.status_code == 409
    assert stale.json()["conflicts"][0]["type"] == "stale_commit_seq"

    current = seq_of(client)
    assert correct(client, ["a"], 60, current).status_code == 200
    assert get(client, "M1").json()["readings"][0]["read_at"] == at(120)


def test_undo_between_query_and_correction_makes_sequence_stale(client):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 60, 160),
        ],
    )
    seq = seq_of(client)
    undo(client, ["b"])
    resp = correct(client, ["a"], 60, seq)
    assert resp.status_code == 409
    assert resp.json()["conflicts"][0]["type"] == "stale_commit_seq"
    # The newly committed state after undo is the base for the next attempt.
    fresh = get(client, "M1").json()
    assert correct(client, ["a"], 60, fresh["commit_seq"]).status_code == 200
    assert get(client, "M1").json()["readings"][0]["read_at"] == at(60)


def test_correction_between_readings_does_not_break_breakpoints(client, settings):
    post(
        client,
        [
            reading("M1", "a", 0, 1000),
            reading("M1", "b", 60, 1200),
            reading("M1", "c", 120, 30, replaced=True),
            reading("M1", "d", 180, 80),
        ],
    )
    seq = seq_of(client)
    # Move the replacement reading forward in time; the break travels with it.
    resp = correct(client, ["c", "d"], 60, seq)
    assert resp.status_code == 200
    state = get(client, "M1").json()
    assert [r["read_at"] for r in state["readings"]] == [
        at(0), at(60), at(180), at(240)
    ]
    assert [i["break_reason"] for i in state["intervals"]] == [
        None, "meter_replaced", None
    ]
    assert [i["usage"] for i in state["intervals"]] == [200, None, 50]
    assert state["usage_total"] == 250
    assert_meter_consistent(state)

    # Moving the replacement reading before the others changes which interval
    # is the break; the recomputation follows the reading's new position.
    seq = state["commit_seq"]
    resp = correct(client, ["c"], -3600, seq)
    assert resp.status_code == 200
    state = get(client, "M1").json()
    assert [r["source_record_id"] for r in state["readings"]] == ["c", "a", "b", "d"]
    assert [i["is_break"] for i in state["intervals"]] == [False, False, True]
    assert state["breaks"][0]["break_reason"] == "counter_rollback"
    assert [i["usage"] for i in state["intervals"]] == [970, 200, None]
    assert_meter_consistent(state)
    with connection(settings) as conn:
        assert_db_invariants(conn, "M1")


def test_correction_keeps_intervals_a_function_of_readings(client, settings):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 120, 160),
            reading("M1", "c", 240, 250),
        ],
    )
    seq = seq_of(client)
    correct(client, ["c", "a"], 3600, seq)  # reorder wildly, same offset
    with connection(settings) as conn:
        assert_db_invariants(conn, "M1")


def test_out_of_range_target_rejects(client):
    post(
        client,
        [
            {
                "meter_id": "M1",
                "source_record_id": "late",
                "read_at": "9999-12-31T23:59:59Z",
                "reading": 1,
            }
        ],
    )
    seq = seq_of(client)
    resp = correct(client, ["late"], 60, seq)
    assert resp.status_code == 409
    assert resp.json()["conflicts"][0]["type"] == "time_out_of_range"
    assert get(client, "M1").json()["readings"][0]["read_at"] == "9999-12-31T23:59:59Z"

    post(
        client,
        [
            {
                "meter_id": "M2",
                "source_record_id": "early",
                "read_at": "0001-01-01T00:00:00Z",
                "reading": 1,
            }
        ],
    )
    seq = get(client, "M2").json()["commit_seq"]
    resp = correct(client, ["early"], -60, seq)
    assert resp.status_code == 409
    assert resp.json()["conflicts"][0]["type"] == "time_out_of_range"


def test_negative_offset(client):
    post(client, [reading("M1", "a", 60, 100)])
    seq = seq_of(client)
    resp = correct(client, ["a"], -60, seq)
    assert resp.status_code == 200
    assert get(client, "M1").json()["readings"][0]["read_at"] == at(0)


def test_original_gateway_retransmission_stays_idempotent(client):
    original = [reading("M1", "a", 0, 10), reading("M1", "b", 60, 20)]
    post(client, original)
    seq = seq_of(client)

    corrected = correct(client, ["a", "b"], 120, seq)
    assert corrected.status_code == 200
    after_correction = corrected.json()["commit_seq"]

    # The gateway re-sends the pre-correction message: accepted, unchanged,
    # and the corrected instants must not move back.
    retry = post(client, original)
    assert retry.status_code == 200, retry.json()
    assert retry.json()["changed"] is False
    assert retry.json()["counts"] == {"unchanged": 2}
    assert retry.json()["commit_seq"] == after_correction

    state = get(client, "M1").json()
    assert state["commit_seq"] == after_correction
    assert [r["read_at"] for r in state["readings"]] == [at(120), at(180)]

    # A retransmission carrying the corrected instant is also a no-op.
    retry_corrected = post(
        client, [reading("M1", "a", 120, 10), reading("M1", "b", 180, 20)]
    )
    assert retry_corrected.status_code == 200
    assert retry_corrected.json()["counts"] == {"unchanged": 2}
    assert retry_corrected.json()["commit_seq"] == after_correction

    # Content other than the original or the current reading still conflicts.
    conflict = post(client, [reading("M1", "a", 240, 10)])
    assert conflict.status_code == 409


def test_second_correction_keeps_original_retransmission_idempotent(client):
    original = [reading("M1", "a", 0, 10)]
    post(client, original)
    correct(client, ["a"], 60, seq_of(client))
    correct(client, ["a"], 60, seq_of(client))
    assert get(client, "M1").json()["readings"][0]["read_at"] == at(120)

    retry = post(client, original)
    assert retry.status_code == 200
    assert retry.json()["counts"] == {"unchanged": 1}
    assert get(client, "M1").json()["readings"][0]["read_at"] == at(120)


def test_undo_after_correction_then_reinsert(client):
    post(client, [reading("M1", "a", 0, 10), reading("M1", "b", 60, 20)])
    seq = seq_of(client)
    correct(client, ["b"], 60, seq)
    # Moving a reading does not change its cumulative value, only the instants.
    assert [i["usage"] for i in get(client, "M1").json()["intervals"]] == [10]

    undo(client, ["b"])
    assert [r["source_record_id"] for r in get(client, "M1").json()["readings"]] == ["a"]

    # After undo the source id is free; submitting the *original* message
    # creates a fresh reading at its original instant.
    resp = post(client, [reading("M1", "b", 60, 45)])
    assert resp.status_code == 200
    assert resp.json()["counts"] == {"created": 1}
    assert [i["usage"] for i in get(client, "M1").json()["intervals"]] == [35]


def test_correction_across_meters_reports_both(client):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 60, 160),
            reading("M2", "x", 0, 5),
            reading("M2", "y", 60, 15),
        ],
    )
    seq = get(client, "M1").json()["commit_seq"]
    resp = correct(client, ["b", "y"], 120, seq)
    assert resp.status_code == 200
    assert [m["meter_id"] for m in resp.json()["affected_meters"]] == ["M1", "M2"]
    assert [i["usage"] for i in get(client, "M1").json()["intervals"]] == [60]
    assert [i["usage"] for i in get(client, "M2").json()["intervals"]] == [10]
    assert get(client, "M1").json()["readings"][-1]["read_at"] == at(180)


def test_duplicate_ids_collapse(client):
    post(client, [reading("M1", "a", 0, 10), reading("M1", "b", 60, 20)])
    resp = correct(client, ["a", "a", "b"], 60, seq_of(client))
    assert resp.status_code == 200
    assert resp.json()["counts"] == {"corrected": 2}
    assert len(resp.json()["records"]) == 2


def test_batch_size_and_offset_validation(client):
    post(client, [reading("M1", f"s{i}", i * 60, i) for i in range(100)])
    seq = seq_of(client)

    too_many = correct(client, [f"s{i}" for i in range(101)], 60, seq)
    assert too_many.status_code == 422
    assert correct(client, [], 60, seq).status_code == 422
    assert correct(client, ["s0"], 0, seq).status_code == 422
    assert correct(client, ["s0"], 3601, seq).status_code == 422
    assert correct(client, ["s0"], -3601, seq).status_code == 422
    assert correct(client, ["s0"], 1.5, seq).status_code == 422
    assert client.post(
        "/api/v1/readings/correct-time",
        json={"source_record_ids": ["s0"], "offset_seconds": 60},
    ).status_code == 422
    # Booleans are not integers, even though Python treats True as 1.
    assert client.post(
        "/api/v1/readings/correct-time",
        json={
            "source_record_ids": ["s0"],
            "offset_seconds": True,
            "expected_commit_seq": seq,
        },
    ).status_code == 422

    # Boundary: exactly 100 ids and exactly 3600 seconds are accepted.
    resp = correct(client, [f"s{i}" for i in range(100)], 3600, seq)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["counts"] == {"corrected": 100}


def test_zero_offset_is_rejected(client):
    post(client, [reading("M1", "a", 0, 10)])
    assert correct(client, ["a"], 0, seq_of(client)).status_code == 422


def test_failure_rolls_back_whole_correction(db_path, settings, monkeypatch):
    import app.service as service
    from app.db import initialize
    from app.models import ReadingIn
    from app.timeutil import to_microseconds
    from datetime import datetime

    initialize(settings)
    service.ingest(
        settings,
        [ReadingIn.model_validate(item) for item in
         (reading("M1", "a", 0, 100), reading("M1", "b", 60, 160))],
    )

    original = service._recompute_intervals

    def explode(conn, meter_id, commit_seq):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_recompute_intervals", explode)
    with pytest.raises(RuntimeError):
        service.correct_times(settings, ["a", "b"], 60, 1)
    monkeypatch.setattr(service, "_recompute_intervals", original)

    expected = [
        to_microseconds(datetime.fromisoformat(at(0).replace("Z", "+00:00"))),
        to_microseconds(datetime.fromisoformat(at(60).replace("Z", "+00:00"))),
    ]
    with connection(settings) as conn:
        assert [
            row[0]
            for row in conn.execute(
                "SELECT read_at_us FROM readings ORDER BY read_at_us"
            )
        ] == expected
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'commit_seq'"
        ).fetchone()[0] == "1"
        assert_db_invariants(conn, "M1")


def test_query_snapshot_never_sees_partial_correction(client, settings):
    post(
        client,
        [
            reading("M1", "a", 0, 100),
            reading("M1", "b", 60, 160),
            reading("M1", "c", 120, 250),
        ],
    )
    seq = seq_of(client)
    resp = correct(client, ["b", "c"], 600, seq)
    assert resp.status_code == 200
    state = get(client, "M1").json()
    assert state["commit_seq"] == resp.json()["commit_seq"]
    assert [r["read_at"] for r in state["readings"]] == [at(0), at(660), at(720)]
    assert_meter_consistent(state)
    for interval in state["intervals"]:
        # Every returned interval belongs to the same commit as the state.
        assert interval["commit_seq"] == state["commit_seq"]


def test_uncommitted_correction_is_rolled_back_on_connection_loss(settings):
    """A crash between parking and final placement cannot leak a move."""
    import sqlite3

    from app.db import initialize

    initialize(settings)
    raw = sqlite3.connect(settings.db_path, timeout=10)
    raw.isolation_level = None
    raw.execute(
        "INSERT INTO readings (source_record_id, meter_id, read_at_us, reading, "
        "meter_replaced, content_hash, original_content_hash, created_commit_seq) "
        "VALUES ('a', 'M1', 0, 10, 0, 'h1', 'h1', 1)"
    )
    raw.commit()
    # A correction holds the write lock, parks the row, then dies before it
    # reaches the final instant: the committed original must survive.
    raw.execute("BEGIN IMMEDIATE")
    raw.execute(
        "UPDATE readings SET read_at_us = -1000000000000000000 "
        "WHERE source_record_id = 'a'"
    )
    raw.close()  # connection death == rollback, parking must not persist

    with connection(settings) as conn:
        rows = conn.execute("SELECT source_record_id, read_at_us FROM readings").fetchall()
        assert [(row[0], row[1]) for row in rows] == [("a", 0)]
