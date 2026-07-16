from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from espresso_rl.adapters.postgres_repositories import PostgresLiveShotSessionRepository
from espresso_rl.adapters.sqlite_repositories import SQLiteLiveShotSessionRepository, SQLiteStore
from espresso_rl.application.live_telemetry import LIVE_SESSION_RETENTION_MS, LiveShotTelemetryService
from espresso_rl.domain.live_telemetry import (
    LiveShotEndedEvent,
    LiveShotSampleEvent,
    LiveShotSession,
    LiveShotSessionStatus,
    LiveShotStartedEvent,
)


class LiveShotTelemetryTests(unittest.TestCase):
    def test_session_records_gaps_is_idempotent_and_reconciles(self) -> None:
        now = [1_800_000_100_000]
        with tempfile.TemporaryDirectory() as tmp, SQLiteStore(Path(tmp) / "espresso.db") as store:
            repository = SQLiteLiveShotSessionRepository(store)
            service = LiveShotTelemetryService(repository, clock_ms=lambda: now[0])
            started = _started()
            service.start(started)
            service.start(started)

            first = _sample(0, 250)
            service.append(first)
            service.append(first)
            service.append(_sample(2, 750))
            service.end(_ended(2, 1_000))
            service.end(_ended(2, 1_000))

            session = repository.get_session("shot_live_1")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.status, LiveShotSessionStatus.ENDED)
            self.assertEqual(session.sample_count, 2)
            self.assertEqual(session.gap_count, 1)

            self.assertTrue(
                service.reconcile_completed_shot(
                    "shot_live_1",
                    "install_1",
                    "gaggimate:AA_BB",
                )
            )
            reconciled = repository.get_session("shot_live_1")
            self.assertIsNotNone(reconciled)
            assert reconciled is not None
            self.assertEqual(reconciled.status, LiveShotSessionStatus.RECONCILED)
            self.assertIsNone(repository.get_sample("shot_live_1", 0))

    def test_conflicting_duplicate_and_sequence_regression_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, SQLiteStore(Path(tmp) / "espresso.db") as store:
            repository = SQLiteLiveShotSessionRepository(store)
            service = LiveShotTelemetryService(repository, clock_ms=lambda: 1_800_000_100_000)
            service.start(_started())
            service.append(_sample(1, 500))

            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                service.append(replace(_sample(1, 500), pressure_bar=8.0))
            with self.assertRaisesRegex(ValueError, "regressed"):
                service.append(_sample(0, 250))

    def test_stale_active_session_expires_on_next_start(self) -> None:
        now = [1_800_000_100_000]
        with tempfile.TemporaryDirectory() as tmp, SQLiteStore(Path(tmp) / "espresso.db") as store:
            repository = SQLiteLiveShotSessionRepository(store)
            service = LiveShotTelemetryService(repository, clock_ms=lambda: now[0])
            service.start(_started())
            service.append(_sample(0, 250))
            now[0] += LIVE_SESSION_RETENTION_MS + 1
            service.start(
                replace(
                    _started(),
                    shot_id="shot_live_2",
                    timestamp_ms=now[0],
                )
            )
            expired = repository.get_session("shot_live_1")
            self.assertIsNotNone(expired)
            assert expired is not None
            self.assertEqual(expired.status, LiveShotSessionStatus.EXPIRED)
            self.assertIsNone(repository.get_sample("shot_live_1", 0))

    def test_sample_and_session_update_roll_back_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, SQLiteStore(Path(tmp) / "espresso.db") as store:
            repository = SQLiteLiveShotSessionRepository(store)
            service = LiveShotTelemetryService(repository, clock_ms=lambda: 1_800_000_100_000)
            service.start(_started())
            store.conn.executescript(
                """
                CREATE TRIGGER reject_live_session_progress
                BEFORE UPDATE ON live_shot_sessions
                WHEN NEW.last_sequence IS NOT NULL
                BEGIN
                    SELECT RAISE(ABORT, 'simulated session write failure');
                END;
                """
            )

            with self.assertRaisesRegex(Exception, "simulated session write failure"):
                service.append(_sample(0, 250))

            session = repository.get_session("shot_live_1")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertIsNone(session.last_sequence)
            self.assertEqual(session.sample_count, 0)
            self.assertIsNone(repository.get_sample("shot_live_1", 0))

    def test_authoritative_completed_shot_discards_conflicting_live_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, SQLiteStore(Path(tmp) / "espresso.db") as store:
            repository = SQLiteLiveShotSessionRepository(store)
            service = LiveShotTelemetryService(repository, clock_ms=lambda: 1_800_000_100_000)
            service.start(_started())
            service.append(_sample(0, 250))

            self.assertFalse(
                service.reconcile_completed_shot(
                    "shot_live_1",
                    "different_install",
                    "gaggimate:OTHER",
                )
            )
            session = repository.get_session("shot_live_1")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.status, LiveShotSessionStatus.EXPIRED)
            self.assertIsNone(repository.get_sample("shot_live_1", 0))

    def test_postgres_sample_and_session_progress_share_one_transaction(self) -> None:
        connection = _FakePostgresConnection()
        repository = PostgresLiveShotSessionRepository(_FakePostgresStore(connection))
        session = LiveShotSession(
            shot_id="shot_live_1",
            install_id="install_1",
            machine_id="gaggimate:AA_BB",
            started_at_ms=1_800_000_000_000,
            sample_interval_ms=250,
            weight_source="hardware_scale",
            flow_source="hardware_scale",
            last_sequence=0,
            sample_count=1,
            updated_at_ms=1_800_000_100_000,
        )

        repository.append_sample(_sample(0, 250), session)

        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)
        self.assertEqual(len(connection.statements), 2)

        failing_connection = _FakePostgresConnection(fail_on_session=True)
        failing_repository = PostgresLiveShotSessionRepository(
            _FakePostgresStore(failing_connection)
        )
        with self.assertRaisesRegex(RuntimeError, "simulated Postgres failure"):
            failing_repository.append_sample(_sample(0, 250), session)
        self.assertEqual(failing_connection.commits, 0)
        self.assertEqual(failing_connection.rollbacks, 1)


def _started() -> LiveShotStartedEvent:
    return LiveShotStartedEvent(
        shot_id="shot_live_1",
        install_id="install_1",
        machine_id="gaggimate:AA_BB",
        timestamp_ms=1_800_000_000_000,
        sample_interval_ms=250,
        weight_source="hardware_scale",
        flow_source="hardware_scale",
    )


def _sample(sequence: int, elapsed_ms: int) -> LiveShotSampleEvent:
    return LiveShotSampleEvent(
        shot_id="shot_live_1",
        install_id="install_1",
        machine_id="gaggimate:AA_BB",
        timestamp_ms=1_800_000_000_000 + elapsed_ms,
        sequence=sequence,
        elapsed_ms=elapsed_ms,
        pressure_bar=9.0,
        pressure_target_bar=9.0,
        pump_flow_ml_s=4.0,
        pump_flow_target_ml_s=4.0,
        beverage_flow_g_s=2.0,
        weight_g=5.0,
        temperature_c=93.0,
        temperature_target_c=93.0,
        pump_target_mode=1,
        valve_open=True,
    )


def _ended(final_sequence: int, elapsed_ms: int) -> LiveShotEndedEvent:
    return LiveShotEndedEvent(
        shot_id="shot_live_1",
        install_id="install_1",
        machine_id="gaggimate:AA_BB",
        timestamp_ms=1_800_000_000_000 + elapsed_ms,
        final_sequence=final_sequence,
        elapsed_ms=elapsed_ms,
        end_state="finished",
    )


class _FakePostgresConnection:
    def __init__(self, *, fail_on_session: bool = False) -> None:
        self.fail_on_session = fail_on_session
        self.statements: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement: str, parameters: object = None) -> "_FakePostgresConnection":
        del parameters
        normalized = " ".join(statement.split())
        self.statements.append(normalized)
        if self.fail_on_session and "INSERT INTO live_shot_sessions" in normalized:
            raise RuntimeError("simulated Postgres failure")
        return self

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakePostgresStore:
    def __init__(self, connection: _FakePostgresConnection) -> None:
        self.conn = connection
