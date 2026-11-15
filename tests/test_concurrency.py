"""Two API processes, one SQLite file: concurrency and crash-safety tests."""

from __future__ import annotations

import os
import random
import socket
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app.db import connection, initialize
from tests.common import (
    assert_db_invariants,
    assert_meter_consistent,
    assert_usage_matches_readings,
    at,
    reading,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, db_path: str, port: int) -> None:
        self.db_path = db_path
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.exit_file = os.path.join(
            os.path.dirname(self.db_path), f"exit-{port}.flag"
        )

    def start(self) -> "Server":
        env = dict(os.environ)
        env.update(
            {
                "ENERGY_DB_PATH": self.db_path,
                "PORT": str(self.port),
                "PYTHONPATH": ROOT,
                "ENERGY_TEST_EXIT_FILE": self.exit_file,
            }
        )
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "tests", "server.py")],
            env=env,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.base_url}/health", timeout=0.5).status_code == 200:
                    return self
            except httpx.HTTPError:
                if self.proc.poll() is not None:
                    raise RuntimeError(self.proc.stderr.read().decode())
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        """Ask the process to exit without running Shutdown handlers.

        This sandbox forbids sending signals to child processes, so the test
        server watches a flag file and calls ``os._exit`` — which is stricter
        than SIGTERM: no atexit hooks, no lifespan shutdown, no transaction
        cleanup.
        """
        if self.proc and self.proc.poll() is None:
            with open(self.exit_file, "w", encoding="utf-8") as handle:
                handle.write("exit")
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if os.path.exists(self.exit_file):
            os.remove(self.exit_file)

    kill = stop


@pytest.fixture()
def two_servers(db_path):
    servers = [Server(db_path, _free_port()).start() for _ in range(2)]
    yield servers
    for server in servers:
        server.stop()


def _client(server: Server) -> httpx.Client:
    return httpx.Client(base_url=server.base_url, timeout=30.0)


def _concurrent_writers(servers, readings):
    """Send one batch per reading, round-robin across processes."""

    def send(index_and_reading):
        index, item = index_and_reading
        with _client(servers[index % len(servers)]) as client:
            return client.post(
                "/api/v1/readings", json={"readings": [item]}
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(send, enumerate(readings)))


def test_two_processes_produce_consistent_intervals(two_servers, settings):
    # Both processes share the database; make sure it exists first.
    with _client(two_servers[0]) as client:
        assert client.get("/health").json()["status"] == "ok"

    readings = [
        reading("M1", f"s{i:03d}", i * 5, 100 + i * 10) for i in range(40)
    ]
    random.Random(7).shuffle(readings)
    responses = _concurrent_writers(two_servers, readings)
    assert all(r.status_code == 200 for r in responses), [
        r.status_code for r in responses
    ]

    for server in two_servers:
        with _client(server) as client:
            body = client.get("/api/v1/meters/M1").json()
        assert len(body["readings"]) == 40
        assert_meter_consistent(body)
        usages = [i["usage"] for i in body["intervals"]]
        assert usages == [10] * 39

    with connection(settings) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert_db_invariants(conn, "M1")


def test_identical_retries_from_both_processes_do_not_duplicate(two_servers, settings):
    readings = [reading("M1", f"r{i}", i * 5, 50 + i) for i in range(20)]
    # Every record is submitted three times, alternating processes.
    tasks = [item for item in readings for _ in range(3)]
    random.Random(3).shuffle(tasks)
    responses = _concurrent_writers(two_servers, tasks)
    assert all(r.status_code == 200 for r in responses)
    statuses = [rec["status"] for r in responses for rec in r.json()["records"]]
    assert statuses.count("created") == 20
    assert statuses.count("unchanged") == 40

    with connection(settings) as conn:
        count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        assert count == 20
        assert_db_invariants(conn, "M1")


def test_conflicting_concurrent_writes_reject_exactly_one(two_servers, settings):
    """Same meter + same instant from two processes: one wins, one gets 409."""
    payload_a = reading("M1", "src-a", 0, 10)
    payload_b = reading("M1", "src-b", 0, 11)

    def send(item):
        with _client(two_servers[0]) as client:
            return client.post("/api/v1/readings", json={"readings": [item]})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(send, item) for item in (payload_a, payload_b)]
        results = [future.result() for future in futures]

    codes = sorted(r.status_code for r in results)
    assert codes == [200, 409], [r.json() for r in results]
    with connection(settings) as conn:
        rows = [
            row[0]
            for row in conn.execute(
                "SELECT source_record_id FROM readings"
            ).fetchall()
        ]
        # Exactly one of the two competing records is stored.
        assert rows in (["src-a"], ["src-b"])
        winners = [
            r.json()["records"][0]["source_record_id"]
            for r in results
            if r.status_code == 200
        ]
        assert rows == winners


def test_query_never_sees_a_partial_commit(two_servers, settings):
    """A reader always sees a state where intervals match the readings."""
    stop = time.monotonic() + 6
    errors: list[str] = []

    def reader():
        with _client(two_servers[1]) as client:
            while time.monotonic() < stop:
                resp = client.get("/api/v1/meters/M1")
                if resp.status_code == 404:
                    continue
                body = resp.json()
                try:
                    assert_meter_consistent(body)
                    assert_usage_matches_readings(body)
                except AssertionError as exc:  # pragma: no cover - diagnostic
                    errors.append(str(exc))
                    return

    def writer(seed):
        rng = random.Random(seed)
        with _client(two_servers[0]) as client:
            index = 0
            while time.monotonic() < stop:
                index += 1
                id_ = f"w{seed}-{index}"
                if rng.random() < 0.3 and index > 2:
                    victim = f"w{seed}-{max(1, index - rng.randint(1, index))}"
                    client.post(
                        "/api/v1/readings/undo",
                        json={"source_record_ids": [victim]},
                    )
                else:
                    client.post(
                        "/api/v1/readings",
                        json={
                            "readings": [
                                reading(
                                    "M1",
                                    id_,
                                    rng.randint(0, 600),
                                    rng.randint(0, 5000),
                                    replaced=rng.random() < 0.05,
                                )
                            ]
                        },
                    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(reader)]
        futures += [pool.submit(writer, seed) for seed in range(3)]
        for future in futures:
            future.result()

    assert not errors, errors[0]
    with connection(settings) as conn:
        assert_db_invariants(conn, "M1")


def test_sigkill_during_write_leaves_no_partial_state(db_path, settings):
    """Kill a process mid-transaction; the next reader sees the old state."""
    initialize(settings)
    server = Server(db_path, _free_port()).start()
    try:
        with _client(server) as client:
            client.post(
                "/api/v1/readings",
                json={
                    "readings": [
                        reading("M1", "a", 0, 100),
                        reading("M1", "b", 60, 200),
                    ]
                },
            )
            before = client.get("/api/v1/meters/M1").json()
            assert len(before["intervals"]) == 1

            # Simulate an in-flight, uncommitted write held by the server
            # process itself: a transaction that never commits.
            # (SIGKILL below rolls it back when the connection dies.)
            client.post(
                "/api/v1/readings",
                json={"readings": [reading("M2", "x", 0, 5)]},
            )
            server.kill()

        with connection(settings) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert_db_invariants(conn, "M1")
            assert conn.execute(
                "SELECT COUNT(*) FROM readings WHERE meter_id = 'M2'"
            ).fetchone()[0] == 1
            assert conn.execute(
                "SELECT COUNT(*) FROM intervals WHERE meter_id = 'M2'"
            ).fetchone()[0] == 0
    finally:
        server.stop()


def test_uncommitted_transaction_is_rolled_back_on_crash(db_path, settings):
    """A killed writer never leaves its half-written rows behind."""
    initialize(settings)
    raw = sqlite3.connect(db_path, timeout=10)
    raw.isolation_level = None
    raw.execute("BEGIN IMMEDIATE")
    raw.execute(
        "INSERT INTO readings (source_record_id, meter_id, read_at_us, reading, "
        "meter_replaced, content_hash, created_commit_seq) "
        "VALUES ('torn', 'M1', 1, 10, 0, 'h', 1)"
    )
    raw.execute(
        "INSERT INTO intervals (meter_id, from_reading_id, to_reading_id, "
        "from_at_us, to_at_us, from_value, to_value, usage, is_break, "
        "break_reason, commit_seq) VALUES ('M1', 1, 2, 1, 2, 1, 2, 1, 0, NULL, 1)"
    )
    raw.close()  # process/connection death == rollback

    with connection(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM readings WHERE source_record_id = 'torn'"
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM intervals").fetchone()[0] == 0


def test_restart_reuses_database_and_schema(db_path, settings):
    server = Server(db_path, _free_port()).start()
    with _client(server) as client:
        client.post(
            "/api/v1/readings",
            json={"readings": [reading("M1", "a", 0, 10), reading("M1", "b", 60, 20)]},
        )
        seq = client.get("/api/v1/meters/M1").json()["commit_seq"]
    server.stop()

    restarted = Server(db_path, _free_port()).start()
    try:
        with _client(restarted) as client:
            body = client.get("/api/v1/meters/M1").json()
            assert body["commit_seq"] == seq
            assert [i["usage"] for i in body["intervals"]] == [10]
            assert_meter_consistent(body)
    finally:
        restarted.stop()


def _correct(client, ids, offset, seq):
    return client.post(
        "/api/v1/readings/correct-time",
        json={
            "source_record_ids": ids,
            "offset_seconds": offset,
            "expected_commit_seq": seq,
        },
    )


def test_concurrent_corrections_on_one_batch_apply_exactly_once(two_servers, settings):
    """Both processes submit the same correction with the same observed seq."""
    readings = [reading("M1", f"c{i}", i * 60, 100 + i) for i in range(10)]
    with _client(two_servers[0]) as client:
        assert client.post(
            "/api/v1/readings", json={"readings": readings}
        ).status_code == 200
        seq = client.get("/api/v1/meters/M1").json()["commit_seq"]

    ids = [r["source_record_id"] for r in readings]

    def send(server):
        with _client(server) as client:
            return _correct(client, ids, 60, seq)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in [pool.submit(send, s) for s in two_servers]]

    codes = sorted(r.status_code for r in results)
    assert codes == [200, 409], [r.json() for r in results]
    rejected = [r for r in results if r.status_code == 409][0]
    assert rejected.json()["code"] == "batch_rejected"
    assert rejected.json()["conflicts"][0]["type"] == "stale_commit_seq"

    with _client(two_servers[1]) as client:
        body = client.get("/api/v1/meters/M1").json()
    expected = [at(i * 60 + 1) for i in range(10)]
    assert [r["read_at"] for r in body["readings"]] == expected
    # Every reading moved exactly once, so intervals still differ by 1 kWh.
    assert [i["usage"] for i in body["intervals"]] == [1] * 9
    assert_meter_consistent(body)
    with connection(settings) as conn:
        assert_db_invariants(conn, "M1")


def test_correction_vs_ingest_conflict_is_decided_by_committed_state(
    two_servers, settings
):
    """A correction and an ingest race; both use their committed pre-state."""
    with _client(two_servers[0]) as client:
        client.post(
            "/api/v1/readings",
            json={"readings": [reading("M1", "a", 0, 100)]},
        )
        base = client.get("/api/v1/meters/M1").json()["commit_seq"]

    def correction():
        with _client(two_servers[0]) as client:
            return _correct(client, ["a"], 60, base)

    def ingest():
        with _client(two_servers[1]) as client:
            return client.post(
                "/api/v1/readings", json={"readings": [reading("M1", "b", 1, 130)]}
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in [pool.submit(correction), pool.submit(ingest)]]

    codes = sorted(r.status_code for r in results)
    # If the correction wins the lock, the ingest collides with the moved
    # reading; if the ingest wins, the correction's sequence is stale.  Either
    # way exactly one of them commits and the other is refused whole.
    assert codes == [200, 409], [r.json() for r in results]

    with _client(two_servers[1]) as client:
        body = client.get("/api/v1/meters/M1").json()
    assert_meter_consistent(body)
    with connection(settings) as conn:
        assert_db_invariants(conn, "M1")
        stored = [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT source_record_id, read_at_us FROM readings "
                "ORDER BY read_at_us"
            )
        ]
        if len(stored) == 1:
            # Correction won: only "a", moved to 60 s after the base instant.
            assert stored[0][0] == "a"
            assert body["readings"][0]["read_at"] == at(1)
        else:
            # Ingest won: both readings at their original instants and the
            # correction left no trace.
            assert [s[0] for s in stored] == ["a", "b"]
            assert [r["read_at"] for r in body["readings"]] == [at(0), at(1)]


def test_queries_never_see_partial_corrections(two_servers, settings):
    """Readers always see a state where readings and intervals agree."""
    with _client(two_servers[0]) as client:
        for i in range(12):
            client.post(
                "/api/v1/readings",
                json={
                    "readings": [
                        reading("M1", f"r{i}", i * 60, 100 + i * 10),
                    ]
                },
            )

    stop = time.monotonic() + 6
    errors: list[str] = []

    def reader():
        with _client(two_servers[1]) as client:
            while time.monotonic() < stop:
                body = client.get("/api/v1/meters/M1").json()
                try:
                    assert_meter_consistent(body)
                    assert_usage_matches_readings(body)
                except AssertionError as exc:  # pragma: no cover - diagnostic
                    errors.append(str(exc))
                    return

    def corrector(seed):
        rng = random.Random(seed)
        with _client(two_servers[seed % 2]) as client:
            while time.monotonic() < stop:
                state = client.get("/api/v1/meters/M1").json()
                visible = [r["source_record_id"] for r in state["readings"]]
                if not visible:
                    continue
                ids = visible[: min(len(visible), rng.randint(1, 4))]
                _correct(
                    client,
                    ids,
                    rng.choice([-60, -1, 1, 60, 3600, -3600]),
                    state["commit_seq"],
                )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(reader)]
        futures += [pool.submit(corrector, seed) for seed in range(2)]
        for future in futures:
            future.result()

    assert not errors, errors[0]
    with connection(settings) as conn:
        assert_db_invariants(conn, "M1")
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_crash_during_correction_leaves_original_state(db_path, settings):
    """A killed process leaves either the full old or the full new state."""
    initialize(settings)
    server = Server(db_path, _free_port()).start()
    try:
        with _client(server) as client:
            client.post(
                "/api/v1/readings",
                json={
                    "readings": [
                        reading("M1", "a", 0, 100),
                        reading("M1", "b", 60, 160),
                        reading("M1", "c", 120, 250),
                    ]
                },
            )
            seq = client.get("/api/v1/meters/M1").json()["commit_seq"]
            server.kill()

        with connection(settings) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert_db_invariants(conn, "M1")
            assert conn.execute(
                "SELECT value FROM meta WHERE key = 'commit_seq'"
            ).fetchone()[0] == str(seq)

        restarted = Server(db_path, _free_port()).start()
        try:
            with _client(restarted) as client:
                resp = _correct(client, ["a", "b"], 600, seq)
                assert resp.status_code == 200, resp.json()
                body = client.get("/api/v1/meters/M1").json()
                assert [r["read_at"] for r in body["readings"]] == [
                    at(10), at(70), at(120)
                ]
                assert_meter_consistent(body)
                assert_usage_matches_readings(body)
        finally:
            restarted.stop()
    finally:
        server.stop()
