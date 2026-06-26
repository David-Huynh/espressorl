from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from espresso_rl.adapters.sqlite_repositories import (
    _recommendation_to_row,
    _row_to_recommendation,
    _row_to_shot,
    _row_to_upload_item,
    _shot_to_row,
    _upload_item_to_row,
)
from espresso_rl.domain.community import (
    AdminActionLogEntry,
    CommunityAbuseEvent,
    CommunityInstallStats,
    CommunityPrior,
    CommunityRawUpload,
    CommunityRecommendationRecord,
    CommunityRejectionSummary,
    CommunityTrainingRow,
    CommunityValidatedShot,
    InstallTrustScore,
    community_rejection_categories,
)
from espresso_rl.domain.models import Recommendation, ShotRecord, UploadQueueItem, UploadQueueStatus


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("postgres_dsn is required when storage_backend=postgres")
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg[binary] is required for Postgres storage") from exc

        self.conn = psycopg.connect(dsn, row_factory=dict_row)
        self.conn.autocommit = False
        self._create_tables()

    def _create_tables(self) -> None:
        schema_path = Path(__file__).with_name("postgres_schema.sql")
        for statement in schema_path.read_text().split(";"):
            if statement.strip():
                self.conn.execute(statement)
        for column, definition in {
            "bean_context_name": "TEXT",
            "grinder_context_id": "TEXT",
            "shot_type": "TEXT NOT NULL DEFAULT 'espresso'",
            "exclude_from_local_optimization": "BOOLEAN NOT NULL DEFAULT FALSE",
            "optimization_weight": "DOUBLE PRECISION NOT NULL DEFAULT 1.0",
            "rating_prompt_allowed": "BOOLEAN NOT NULL DEFAULT TRUE",
            "grind_followed": "BOOLEAN",
            "dose_followed": "BOOLEAN",
            "yield_followed": "BOOLEAN",
            "grind_recommendation_trust": "DOUBLE PRECISION NOT NULL DEFAULT 0.0",
            "dose_recommendation_trust": "DOUBLE PRECISION NOT NULL DEFAULT 0.0",
            "yield_recommendation_trust": "DOUBLE PRECISION NOT NULL DEFAULT 0.0",
            "weight_source": "TEXT",
            "flow_source": "TEXT",
            "flow_units": "TEXT",
            "pump_flow_source": "TEXT",
            "pump_flow_units": "TEXT",
            "pump_flow_calibration_required": "BOOLEAN NOT NULL DEFAULT FALSE",
            "profile_flow_valid": "BOOLEAN NOT NULL DEFAULT TRUE",
            "profile_flow_masked": "BOOLEAN NOT NULL DEFAULT FALSE",
            "profile_id": "TEXT",
            "profile_label": "TEXT",
            "profile_type": "TEXT",
            "profile_phase_count": "INTEGER",
            "final_phase_index": "INTEGER",
            "final_phase_name": "TEXT",
            "final_phase_type": "TEXT",
            "final_phase_elapsed_s": "DOUBLE PRECISION",
            "final_pump_target": "TEXT",
            "final_target_pressure": "DOUBLE PRECISION",
            "final_target_flow": "DOUBLE PRECISION",
            "final_valve_open": "BOOLEAN",
            "profile_temperature_c": "DOUBLE PRECISION",
            "final_phase_temperature_c": "DOUBLE PRECISION",
            "beverage_flow_profile_blob": "BYTEA",
            "temperature_profile_blob": "BYTEA",
            "target_temperature_profile_blob": "BYTEA",
            "pump_target_mode_profile_blob": "BYTEA",
            "shot_end_state": "TEXT",
            "grinder_calibration_mode": "TEXT NOT NULL DEFAULT 'relative_calibrated'",
            "grinder_step_direction": "TEXT NOT NULL DEFAULT 'higher_is_finer'",
            "grinder_reference_label": "TEXT NOT NULL DEFAULT 'reference'",
            "current_absolute_step": "DOUBLE PRECISION",
            "absolute_reference_step": "DOUBLE PRECISION",
        }.items():
            self.conn.execute(f"ALTER TABLE shots ADD COLUMN IF NOT EXISTS {column} {definition}")
        self.conn.execute("ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS grinder_context_id TEXT")
        for column, definition in {
            "grinder_calibration_mode": "TEXT NOT NULL DEFAULT 'relative_calibrated'",
            "grinder_step_direction": "TEXT NOT NULL DEFAULT 'higher_is_finer'",
            "grinder_reference_label": "TEXT NOT NULL DEFAULT 'reference'",
            "current_absolute_step": "DOUBLE PRECISION",
            "absolute_reference_step": "DOUBLE PRECISION",
            "projected_absolute_step": "DOUBLE PRECISION",
        }.items():
            self.conn.execute(f"ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS {column} {definition}")
        for column, definition in {
            "validated_at": "TIMESTAMPTZ",
            "rejected_at": "TIMESTAMPTZ",
            "validation_summary": "JSONB NOT NULL DEFAULT '{}'::jsonb",
            "validation_errors": "JSONB NOT NULL DEFAULT '[]'::jsonb",
        }.items():
            self.conn.execute(
                f"ALTER TABLE community_raw_uploads ADD COLUMN IF NOT EXISTS {column} {definition}"
            )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_training_dataset_source_validation_id
                ON training_dataset (source_validation_id)
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_community_priors_context_key
                ON community_priors (context_key)
            """
        )
        self.conn.commit()


def _nullable_clause(
    column: str,
    value: str | None,
    params: list[object],
    placeholder: str = "%s",
) -> str:
    if value is None:
        return f"{column} IS NULL"
    params.append(value)
    return f"{column}={placeholder}"


class PostgresShotRepository:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def upsert(self, shot: ShotRecord) -> None:
        row = _shot_to_row(shot)
        _upsert(self._store.conn, "shots", "shot_id", row)

    def get(self, shot_id: str) -> ShotRecord | None:
        row = self._store.conn.execute("SELECT * FROM shots WHERE shot_id=%s", (shot_id,)).fetchone()
        return _row_to_shot(row) if row else None

    def list_recent(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None = None,
        limit: int = 200,
        grinder_context_id: str | None = None,
    ) -> list[ShotRecord]:
        params: list[object] = [install_id, machine_id]
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params)
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params)
        params.append(limit)
        rows = self._store.conn.execute(
            f"""
            SELECT * FROM shots
            WHERE install_id=%s AND machine_id=%s AND {bean_clause} AND {grinder_clause}
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            tuple(params),
        ).fetchall()
        return list(reversed([_row_to_shot(row) for row in rows]))

    def list_machine_shots(
        self,
        install_id: str,
        machine_id: str,
        limit: int = 500,
    ) -> list[ShotRecord]:
        rows = self._store.conn.execute(
            """
            SELECT * FROM shots
            WHERE install_id=%s AND machine_id=%s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (install_id, machine_id, limit),
        ).fetchall()
        return list(reversed([_row_to_shot(row) for row in rows]))


class PostgresLocalDataRepository:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def list_machine_shots(
        self,
        install_id: str,
        machine_id: str,
        limit: int = 500,
    ) -> list[ShotRecord]:
        rows = self._store.conn.execute(
            """
            SELECT * FROM shots
            WHERE install_id=%s AND machine_id=%s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (install_id, machine_id, limit),
        ).fetchall()
        shots = list(reversed([_row_to_shot(row) for row in rows]))
        _mark_rejected_uploads_postgres(self._store.conn, shots)
        return shots

    def delete_shot(
        self,
        install_id: str,
        machine_id: str,
        shot_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, int]:
        try:
            shot_count = int(
                self._store.conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM shots
                    WHERE install_id=%s AND machine_id=%s AND shot_id=%s
                    """,
                    (install_id, machine_id, shot_id),
                ).fetchone()["count"]
            )
            upload_count = 0
            if shot_count:
                upload_count = int(
                    self._store.conn.execute(
                        """
                        SELECT COUNT(*) AS count FROM upload_queue
                        WHERE local_record_type='shot' AND local_record_id=%s
                        """,
                        (shot_id,),
                    ).fetchone()["count"]
                )
            if dry_run:
                return {"shots": shot_count, "upload_queue": upload_count}
            if shot_count:
                self._store.conn.execute(
                    "DELETE FROM upload_queue WHERE local_record_type='shot' AND local_record_id=%s",
                    (shot_id,),
                )
                self._store.conn.execute(
                    "DELETE FROM shots WHERE install_id=%s AND machine_id=%s AND shot_id=%s",
                    (install_id, machine_id, shot_id),
                )
            self._store.conn.commit()
            return {"shots": shot_count, "upload_queue": upload_count}
        except Exception:
            self._store.conn.rollback()
            raise

    def exclude_shot_from_optimization(
        self,
        install_id: str,
        machine_id: str,
        shot_id: str,
        *,
        now: int,
        dry_run: bool = False,
    ) -> dict[str, int]:
        try:
            shot_count = int(
                self._store.conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM shots
                    WHERE install_id=%s AND machine_id=%s AND shot_id=%s
                    """,
                    (install_id, machine_id, shot_id),
                ).fetchone()["count"]
            )
            if dry_run:
                return {"shots": shot_count}
            if shot_count:
                self._store.conn.execute(
                    """
                    UPDATE shots
                    SET exclude_from_local_optimization=TRUE,
                        optimization_weight=0,
                        recommendation_attribution_weight=0,
                        updated_at=%s
                    WHERE install_id=%s AND machine_id=%s AND shot_id=%s
                    """,
                    (now, install_id, machine_id, shot_id),
                )
            self._store.conn.commit()
            return {"shots": shot_count}
        except Exception:
            self._store.conn.rollback()
            raise

    def purge_useless_shots(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None = None,
        *,
        limit: int = 100,
        dry_run: bool = False,
        grinder_context_id: str | None = None,
    ) -> dict[str, int]:
        try:
            context_clause = ""
            params: list[object] = [install_id, machine_id]
            if bean_context_id is not None or grinder_context_id is not None:
                clauses = [
                    _nullable_clause("bean_context_id", bean_context_id, params),
                    _nullable_clause("grinder_context_id", grinder_context_id, params),
                ]
                context_clause = "AND " + " AND ".join(clauses)
            params.append(limit)
            rows = self._store.conn.execute(
                f"""
                SELECT shot_id FROM shots
                WHERE install_id=%s AND machine_id=%s
                  {context_clause}
                  AND (
                    shot_type != 'espresso'
                    OR exclude_from_local_optimization = TRUE
                    OR optimization_weight <= 0
                  )
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                tuple(params),
            ).fetchall()
            shot_ids = [row["shot_id"] for row in rows]
            upload_count = _count_upload_queue_for_shots_postgres(self._store.conn, shot_ids)
            if dry_run:
                return {"shots": len(shot_ids), "upload_queue": upload_count}
            if shot_ids:
                self._store.conn.execute(
                    "DELETE FROM upload_queue WHERE local_record_type='shot' AND local_record_id = ANY(%s)",
                    (shot_ids,),
                )
                self._store.conn.execute(
                    "DELETE FROM shots WHERE shot_id = ANY(%s) AND install_id=%s AND machine_id=%s",
                    (shot_ids, install_id, machine_id),
                )
            self._store.conn.commit()
            return {"shots": len(shot_ids), "upload_queue": upload_count}
        except Exception:
            self._store.conn.rollback()
            raise

    def reset_optimizer_context(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        *,
        now: int,
        dry_run: bool = False,
        grinder_context_id: str | None = None,
    ) -> dict[str, int]:
        try:
            context_params: list[object] = [install_id, machine_id, bean_context_id]
            grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, context_params)
            shot_count = int(
                self._store.conn.execute(
                    f"""
                    SELECT COUNT(*) AS count FROM shots
                    WHERE install_id=%s AND machine_id=%s AND bean_context_id=%s AND {grinder_clause}
                      AND (
                        exclude_from_local_optimization = FALSE
                        OR optimization_weight > 0
                        OR recommendation_attribution_weight > 0
                      )
                    """,
                    tuple(context_params),
                ).fetchone()["count"]
            )
            rec_context_params: list[object] = [install_id, machine_id, bean_context_id]
            rec_grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, rec_context_params)
            recommendation_count = int(
                self._store.conn.execute(
                    f"""
                    SELECT COUNT(*) AS count FROM recommendations
                    WHERE install_id=%s AND machine_id=%s AND bean_context_id=%s AND {rec_grinder_clause}
                      AND status IN ('pending', 'shown', 'accepted', 'edited')
                    """,
                    tuple(rec_context_params),
                ).fetchone()["count"]
            )
            if dry_run:
                return {"shots": shot_count, "recommendations": recommendation_count}
            self._store.conn.execute(
                f"""
                UPDATE shots
                SET exclude_from_local_optimization=TRUE,
                    optimization_weight=0,
                    recommendation_attribution_weight=0,
                    updated_at=%s
                WHERE install_id=%s AND machine_id=%s AND bean_context_id=%s AND {grinder_clause}
                """,
                (now, *context_params),
            )
            self._store.conn.execute(
                f"""
                UPDATE recommendations
                SET status='superseded',
                    superseded_at=COALESCE(superseded_at, %s),
                    updated_at=%s
                WHERE install_id=%s AND machine_id=%s AND bean_context_id=%s AND {rec_grinder_clause}
                  AND status IN ('pending', 'shown', 'accepted', 'edited')
                """,
                (now, now, *rec_context_params),
            )
            self._store.conn.commit()
            return {"shots": shot_count, "recommendations": recommendation_count}
        except Exception:
            self._store.conn.rollback()
            raise


class PostgresRecommendationRepository:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def upsert(self, recommendation: Recommendation) -> None:
        row = _recommendation_to_row(recommendation)
        _upsert(self._store.conn, "recommendations", "recommendation_id", row)

    def get(self, recommendation_id: str) -> Recommendation | None:
        row = self._store.conn.execute(
            "SELECT * FROM recommendations WHERE recommendation_id=%s",
            (recommendation_id,),
        ).fetchone()
        return _row_to_recommendation(row) if row else None

    def get_latest(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None = None,
    ) -> Recommendation | None:
        params: list[object] = [install_id, machine_id]
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params)
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params)
        row = self._store.conn.execute(
            f"""
            SELECT * FROM recommendations
            WHERE install_id=%s AND machine_id=%s AND {bean_clause} AND {grinder_clause}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return _row_to_recommendation(row) if row else None

    def get_current(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
        grinder_context_id: str | None = None,
    ) -> Recommendation | None:
        params: list[object] = [install_id, machine_id]
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params)
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params)
        params.append(now)
        row = self._store.conn.execute(
            f"""
            SELECT * FROM recommendations
            WHERE install_id=%s AND machine_id=%s AND {bean_clause} AND {grinder_clause}
              AND status IN ('pending', 'shown', 'accepted', 'edited')
              AND (expires_at IS NULL OR expires_at > %s)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return _row_to_recommendation(row) if row else None

    def supersede_active(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
        except_recommendation_id: str | None = None,
        grinder_context_id: str | None = None,
    ) -> None:
        params: list[Any] = [now, now, install_id, machine_id]
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params)
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params)
        except_clause = ""
        if except_recommendation_id is not None:
            except_clause = "AND recommendation_id != %s"
            params.append(except_recommendation_id)
        self._store.conn.execute(
            f"""
            UPDATE recommendations
            SET status='superseded', superseded_at=%s, updated_at=%s
            WHERE install_id=%s AND machine_id=%s AND {bean_clause} AND {grinder_clause}
              AND status IN ('pending', 'shown')
              {except_clause}
            """,
            tuple(params),
        )
        self._store.conn.commit()


class PostgresUploadQueueRepository:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def enqueue(self, item: UploadQueueItem) -> None:
        conn = self._store.conn
        try:
            # Idempotency: this exact content was already uploaded; nothing to do.
            if conn.execute(
                """
                SELECT 1 FROM upload_queue
                WHERE local_record_type=%s AND local_record_id=%s AND payload_hash=%s
                  AND status=%s
                LIMIT 1
                """,
                (
                    item.local_record_type,
                    item.local_record_id,
                    item.payload_hash,
                    UploadQueueStatus.UPLOADED.value,
                ),
            ).fetchone():
                conn.commit()
                return
            # Never clobber an in-flight send of this exact content.
            if conn.execute(
                "SELECT 1 FROM upload_queue WHERE upload_id=%s AND status=%s LIMIT 1",
                (item.upload_id, UploadQueueStatus.UPLOADING.value),
            ).fetchone():
                conn.commit()
                return
            # Coalesce: drop superseded unsent versions for this record. Rejected
            # rows are also rearmed by a new snapshot, which lets a payload/schema
            # fix drain local data without manual SQL. UPLOADING rows are left
            # untouched (a worker may be mid-send); UPLOADED rows are kept as the
            # "already sent" memory.
            conn.execute(
                """
                DELETE FROM upload_queue
                WHERE local_record_type=%s AND local_record_id=%s AND status IN (%s, %s, %s)
                """,
                (
                    item.local_record_type,
                    item.local_record_id,
                    UploadQueueStatus.PENDING.value,
                    UploadQueueStatus.FAILED.value,
                    UploadQueueStatus.REJECTED.value,
                ),
            )
            row = _upload_item_to_row(item)
            columns = list(row)
            column_sql = ", ".join(columns)
            value_sql = ", ".join(f"%({column})s" for column in columns)
            update_sql = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns if column != "upload_id")
            conn.execute(
                f"""
                INSERT INTO upload_queue ({column_sql})
                VALUES ({value_sql})
                ON CONFLICT (upload_id) DO UPDATE SET {update_sql}
                """,
                row,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def list_ready(self, now: int, limit: int = 100) -> list[UploadQueueItem]:
        rows = self._store.conn.execute(
            """
            SELECT * FROM upload_queue
            WHERE status IN ('pending', 'failed')
              AND (next_retry_at IS NULL OR next_retry_at <= %s)
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (now, limit),
        ).fetchall()
        return [_row_to_upload_item(row) for row in rows]

    def update_status(
        self,
        upload_id: str,
        status: UploadQueueStatus,
        now: int,
        error_message: str | None = None,
        next_retry_at: int | None = None,
    ) -> None:
        existing = self._store.conn.execute(
            "SELECT attempt_count FROM upload_queue WHERE upload_id=%s",
            (upload_id,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"unknown upload_id {upload_id}")
        attempt_count = int(existing["attempt_count"])
        last_attempt_at = None
        # Count one attempt per completed try (FAILED/REJECTED), not for the
        # transient UPLOADING transition, so a single cycle increments once.
        if status in {UploadQueueStatus.FAILED, UploadQueueStatus.REJECTED}:
            attempt_count += 1
            last_attempt_at = now
        self._store.conn.execute(
            """
            UPDATE upload_queue
            SET status=%s, attempt_count=%s, last_attempt_at=COALESCE(%s, last_attempt_at),
                next_retry_at=%s, error_message=%s, updated_at=%s
            WHERE upload_id=%s
            """,
            (
                UploadQueueStatus(status).value,
                attempt_count,
                last_attempt_at,
                next_retry_at,
                error_message,
                now,
                upload_id,
            ),
        )
        self._store.conn.commit()

    def count_by_status(self) -> dict[UploadQueueStatus, int]:
        rows = self._store.conn.execute(
            "SELECT status, COUNT(*) AS count FROM upload_queue GROUP BY status"
        ).fetchall()
        return {UploadQueueStatus(row["status"]): int(row["count"]) for row in rows}

    def list_by_status(self, status: UploadQueueStatus, limit: int = 100) -> list[UploadQueueItem]:
        rows = self._store.conn.execute(
            """
            SELECT * FROM upload_queue
            WHERE status=%s
            ORDER BY updated_at DESC, created_at DESC
            LIMIT %s
            """,
            (UploadQueueStatus(status).value, limit),
        ).fetchall()
        return [_row_to_upload_item(row) for row in rows]

    def requeue(
        self,
        upload_id: str,
        now: int,
        error_message: str | None = None,
    ) -> None:
        existing = self._store.conn.execute(
            "SELECT 1 FROM upload_queue WHERE upload_id=%s AND status=%s",
            (upload_id, UploadQueueStatus.REJECTED.value),
        ).fetchone()
        if existing is None:
            raise ValueError(f"unknown rejected upload_id {upload_id}")
        self._store.conn.execute(
            """
            UPDATE upload_queue
            SET status=%s, attempt_count=0, last_attempt_at=NULL, next_retry_at=NULL,
                error_message=%s, updated_at=%s
            WHERE upload_id=%s
            """,
            (
                UploadQueueStatus.PENDING.value,
                error_message,
                now,
                upload_id,
            ),
        )
        self._store.conn.commit()

    def mark_rejected_preflight_failed(
        self,
        upload_id: str,
        now: int,
        error_message: str,
    ) -> None:
        existing = self._store.conn.execute(
            "SELECT 1 FROM upload_queue WHERE upload_id=%s AND status=%s",
            (upload_id, UploadQueueStatus.REJECTED.value),
        ).fetchone()
        if existing is None:
            raise ValueError(f"unknown rejected upload_id {upload_id}")
        self._store.conn.execute(
            """
            UPDATE upload_queue
            SET error_message=%s, updated_at=%s
            WHERE upload_id=%s AND status=%s
            """,
            (
                error_message[:500],
                now,
                upload_id,
                UploadQueueStatus.REJECTED.value,
            ),
        )
        self._store.conn.commit()

    def purge_rejected_artifacts(
        self,
        now: int,
        limit: int = 100,
        local_record_id: str | None = None,
    ) -> dict[str, int]:
        del now  # The delete itself is intentionally timestamp-free; audit stays in logs/UI events.
        conn = self._store.conn
        inspected = 0
        purged_uploads = 0
        purged_shots = 0
        purged_recommendations = 0
        kept_linked_records = 0
        try:
            params: list[object] = [UploadQueueStatus.REJECTED.value]
            record_filter = ""
            if local_record_id:
                record_filter = "AND local_record_id=%s"
                params.append(local_record_id)
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT upload_id, local_record_type, local_record_id
                FROM upload_queue
                WHERE status=%s
                  {record_filter}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT %s
                """,
                tuple(params),
            ).fetchall()
            inspected = len(rows)
            for row in rows:
                record_type = row["local_record_type"]
                record_id = row["local_record_id"]
                deleted_linked = False
                if record_type == "shot":
                    shot = conn.execute(
                        "DELETE FROM shots WHERE shot_id=%s RETURNING shot_id",
                        (record_id,),
                    ).fetchone()
                    if shot is not None:
                        purged_shots += 1
                        deleted_linked = True
                elif record_type == "recommendation":
                    recommendation = conn.execute(
                        """
                        DELETE FROM recommendations
                        WHERE recommendation_id=%s
                          AND status IN ('ignored', 'expired', 'superseded')
                          AND NOT EXISTS (
                            SELECT 1 FROM shots
                            WHERE shots.recommendation_id = recommendations.recommendation_id
                          )
                        RETURNING recommendation_id
                        """,
                        (record_id,),
                    ).fetchone()
                    if recommendation is not None:
                        purged_recommendations += 1
                        deleted_linked = True
                if record_type in {"shot", "recommendation"} and not deleted_linked:
                    kept_linked_records += 1

                deleted = conn.execute(
                    "DELETE FROM upload_queue WHERE upload_id=%s AND status=%s RETURNING upload_id",
                    (row["upload_id"], UploadQueueStatus.REJECTED.value),
                ).fetchone()
                if deleted is not None:
                    purged_uploads += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return {
            "inspected": inspected,
            "purged_uploads": purged_uploads,
            "purged_shots": purged_shots,
            "purged_recommendations": purged_recommendations,
            "kept_linked_records": kept_linked_records,
        }


class PostgresCommunityWarehouse:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def upsert_raw_upload(self, upload: CommunityRawUpload) -> None:
        self._store.conn.execute(
            """
            INSERT INTO community_raw_uploads (
                install_id, upload_id, payload_hash, event_type,
                payload_json, supabase_received_at, status
            ) VALUES (
                %(install_id)s, %(upload_id)s, %(payload_hash)s, %(event_type)s,
                %(payload_json)s::jsonb, %(supabase_received_at)s, 'mirrored'
            )
            ON CONFLICT (install_id, upload_id) DO UPDATE SET
                payload_hash=EXCLUDED.payload_hash,
                event_type=EXCLUDED.event_type,
                payload_json=EXCLUDED.payload_json,
                supabase_received_at=EXCLUDED.supabase_received_at,
                mirrored_at=now(),
                status='mirrored'
            """,
            {
                "install_id": upload.install_id,
                "upload_id": upload.upload_id,
                "payload_hash": upload.payload_hash,
                "event_type": upload.event_type,
                "payload_json": json.dumps(upload.payload_json, sort_keys=True),
                "supabase_received_at": upload.received_at,
            },
        )
        self._store.conn.commit()

    def list_raw_uploads(self, status: str = "mirrored", limit: int = 100) -> list[CommunityRawUpload]:
        rows = self._store.conn.execute(
            """
            SELECT install_id, upload_id, payload_hash, event_type, payload_json, supabase_received_at
            FROM community_raw_uploads
            WHERE status=%s
            ORDER BY mirrored_at ASC
            LIMIT %s
            """,
            (status, limit),
        ).fetchall()
        return [_row_to_community_raw_upload(row) for row in rows]

    def mark_raw_upload_validated(
        self,
        upload: CommunityRawUpload,
        validation_summary: dict[str, Any],
    ) -> None:
        self._store.conn.execute(
            """
            UPDATE community_raw_uploads
            SET status='validated', validated_at=now(), rejected_at=NULL,
                validation_summary=%s::jsonb, validation_errors='[]'::jsonb
            WHERE install_id=%s AND upload_id=%s
            """,
            (
                json.dumps(validation_summary, sort_keys=True),
                upload.install_id,
                upload.upload_id,
            ),
        )
        self._store.conn.commit()

    def mark_raw_upload_rejected(
        self,
        upload: CommunityRawUpload,
        validation_errors: list[str],
    ) -> None:
        self._store.conn.execute(
            """
            UPDATE community_raw_uploads
            SET status='rejected', rejected_at=now(),
                validation_errors=%s::jsonb
            WHERE install_id=%s AND upload_id=%s
            """,
            (
                json.dumps(validation_errors[:50], sort_keys=True),
                upload.install_id,
                upload.upload_id,
            ),
        )
        self._store.conn.commit()

    def upsert_validated_shot(self, shot: CommunityValidatedShot) -> int:
        row = self._store.conn.execute(
            """
            INSERT INTO community_validated_shots (
                install_id, upload_id, shot_id, payload_json, trust_weight, validation_summary
            ) VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb)
            ON CONFLICT (install_id, shot_id) DO UPDATE SET
                upload_id=EXCLUDED.upload_id,
                payload_json=EXCLUDED.payload_json,
                trust_weight=EXCLUDED.trust_weight,
                validation_summary=EXCLUDED.validation_summary
            RETURNING validation_id
            """,
            (
                shot.install_id,
                shot.upload_id,
                shot.shot_id,
                json.dumps(shot.payload_json, sort_keys=True),
                shot.trust_weight,
                json.dumps(shot.validation_summary, sort_keys=True),
            ),
        ).fetchone()
        self._store.conn.commit()
        return int(row["validation_id"])

    def upsert_community_recommendation(self, recommendation: CommunityRecommendationRecord) -> None:
        self._store.conn.execute(
            """
            INSERT INTO community_recommendations (
                install_id, recommendation_id, upload_id, payload_json
            ) VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (install_id, recommendation_id) DO UPDATE SET
                upload_id=EXCLUDED.upload_id,
                payload_json=EXCLUDED.payload_json
            """,
            (
                recommendation.install_id,
                recommendation.recommendation_id,
                recommendation.upload_id,
                json.dumps(recommendation.payload_json, sort_keys=True),
            ),
        )
        self._store.conn.commit()

    def upsert_install_trust_score(self, score: InstallTrustScore) -> None:
        self._store.conn.execute(
            """
            INSERT INTO install_trust_scores (install_id, trust_score, reason, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (install_id) DO UPDATE SET
                trust_score=EXCLUDED.trust_score,
                reason=EXCLUDED.reason,
                updated_at=now()
            """,
            (score.install_id, score.trust_score, score.reason),
        )
        self._store.conn.commit()

    def install_stats(self, install_id: str) -> CommunityInstallStats:
        validated_row = self._store.conn.execute(
            "SELECT COUNT(*) AS count FROM community_validated_shots WHERE install_id=%s",
            (install_id,),
        ).fetchone()
        rejected_row = self._store.conn.execute(
            """
            SELECT COUNT(*) AS count FROM community_raw_uploads
            WHERE install_id=%s AND status='rejected'
            """,
            (install_id,),
        ).fetchone()
        abuse_row = self._store.conn.execute(
            "SELECT COUNT(*) AS count FROM abuse_events WHERE install_id=%s",
            (install_id,),
        ).fetchone()
        return CommunityInstallStats(
            validated_shots=int(validated_row["count"]),
            rejected_uploads=int(rejected_row["count"]),
            abuse_events=int(abuse_row["count"]),
        )

    def record_abuse_event(self, event: CommunityAbuseEvent) -> None:
        self._store.conn.execute(
            """
            INSERT INTO abuse_events (install_id, upload_id, payload_hash, reason, detail)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            """,
            (
                event.install_id,
                event.upload_id,
                event.payload_hash,
                event.reason,
                json.dumps(event.detail, sort_keys=True),
            ),
        )
        self._store.conn.commit()

    def upsert_training_row(
        self,
        source_validation_id: int,
        payload_json: dict[str, Any],
        trust_weight: float,
    ) -> None:
        self._store.conn.execute(
            """
            INSERT INTO training_dataset (source_validation_id, payload_json, trust_weight)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (source_validation_id) DO UPDATE SET
                payload_json=EXCLUDED.payload_json,
                trust_weight=EXCLUDED.trust_weight
            """,
            (
                source_validation_id,
                json.dumps(payload_json, sort_keys=True),
                trust_weight,
            ),
        )
        self._store.conn.commit()

    def list_training_rows(self, limit: int = 5000) -> list[CommunityTrainingRow]:
        rows = self._store.conn.execute(
            """
            SELECT
                td.training_row_id,
                td.source_validation_id,
                cvs.install_id,
                td.payload_json,
                td.trust_weight,
                cru.payload_hash
            FROM training_dataset td
            JOIN community_validated_shots cvs
                ON cvs.validation_id = td.source_validation_id
            LEFT JOIN community_raw_uploads cru
                ON cru.install_id = cvs.install_id AND cru.upload_id = cvs.upload_id
            WHERE td.trust_weight > 0
            ORDER BY td.created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [_row_to_training_row(row) for row in rows]

    def upsert_community_prior(self, prior: CommunityPrior) -> None:
        self._store.conn.execute(
            """
            INSERT INTO community_priors (context_key, prior_json, confidence, created_at)
            VALUES (%s, %s::jsonb, %s, now())
            ON CONFLICT (context_key) DO UPDATE SET
                prior_json=EXCLUDED.prior_json,
                confidence=EXCLUDED.confidence,
                created_at=now()
            """,
            (
                prior.context_key,
                json.dumps(prior.prior_json, sort_keys=True),
                prior.confidence,
            ),
        )
        self._store.conn.commit()

    def list_community_priors(self, context_key: str, limit: int = 10) -> list[CommunityPrior]:
        rows = self._store.conn.execute(
            """
            SELECT context_key, prior_json, confidence
            FROM community_priors
            WHERE context_key=%s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (context_key, limit),
        ).fetchall()
        return [_row_to_community_prior(row) for row in rows]

    def raw_upload_counts_by_status(self) -> dict[str, int]:
        rows = self._store.conn.execute(
            "SELECT status, COUNT(*) AS count FROM community_raw_uploads GROUP BY status"
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def raw_upload_purge_eligible_counts(
        self,
        *,
        validated_retention_days: int = 14,
        rejected_retention_days: int = 30,
    ) -> dict[str, int]:
        rows = self._store.conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM community_raw_uploads
            WHERE (
                    status='validated'
                    AND validated_at IS NOT NULL
                    AND validated_at < now() - make_interval(days => %s)
                )
                OR (
                    status='rejected'
                    AND rejected_at IS NOT NULL
                    AND rejected_at < now() - make_interval(days => %s)
                )
            GROUP BY status
            """,
            (
                max(1, int(validated_retention_days)),
                max(1, int(rejected_retention_days)),
            ),
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def purge_raw_uploads(
        self,
        *,
        validated_retention_days: int = 14,
        rejected_retention_days: int = 30,
    ) -> int:
        try:
            cursor = self._store.conn.execute(
                """
                DELETE FROM community_raw_uploads
                WHERE (
                        status='validated'
                        AND validated_at IS NOT NULL
                        AND validated_at < now() - make_interval(days => %s)
                    )
                    OR (
                        status='rejected'
                        AND rejected_at IS NOT NULL
                        AND rejected_at < now() - make_interval(days => %s)
                    )
                """,
                (
                    max(1, int(validated_retention_days)),
                    max(1, int(rejected_retention_days)),
                ),
            )
            self._store.conn.commit()
            return max(0, int(cursor.rowcount or 0))
        except Exception:
            self._store.conn.rollback()
            raise

    def validated_shot_count(self) -> int:
        return _count_table(self._store.conn, "community_validated_shots")

    def training_row_count(self) -> int:
        return _count_table(self._store.conn, "training_dataset")

    def community_prior_count(self) -> int:
        return _count_table(self._store.conn, "community_priors")

    def abuse_event_count(self) -> int:
        return _count_table(self._store.conn, "abuse_events")

    def latest_rejections(self, limit: int = 10) -> list[CommunityRejectionSummary]:
        rows = self._store.conn.execute(
            """
            SELECT install_id, upload_id, event_type, validation_errors, rejected_at
            FROM community_raw_uploads
            WHERE status='rejected'
            ORDER BY rejected_at DESC NULLS LAST, mirrored_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [_row_to_rejection_summary(row) for row in rows]

    def record_admin_action(self, entry: AdminActionLogEntry) -> None:
        self._store.conn.execute(
            """
            INSERT INTO admin_action_log (
                action_type, requested_at, requested_by, dry_run, status,
                rows_seen, rows_changed, warnings_count, error_summary
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                entry.action_type,
                entry.requested_at,
                entry.requested_by,
                entry.dry_run,
                entry.status,
                entry.rows_seen,
                entry.rows_changed,
                entry.warnings_count,
                entry.error_summary,
            ),
        )
        self._store.conn.commit()

    def latest_admin_actions(self, limit: int = 10) -> list[AdminActionLogEntry]:
        rows = self._store.conn.execute(
            """
            SELECT action_type, requested_at, requested_by, dry_run, status,
                   rows_seen, rows_changed, warnings_count, error_summary
            FROM admin_action_log
            ORDER BY requested_at DESC, action_id DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [_row_to_admin_action(row) for row in rows]


def _upsert(conn, table: str, key: str, row: dict[str, Any]) -> None:
    columns = list(row)
    column_sql = ", ".join(columns)
    value_sql = ", ".join(f"%({column})s" for column in columns)
    update_sql = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns if column != key)
    try:
        conn.execute(
            f"""
            INSERT INTO {table} ({column_sql})
            VALUES ({value_sql})
            ON CONFLICT ({key}) DO UPDATE SET {update_sql}
            """,
            row,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _count_table(conn, table: str) -> int:
    if table not in {
        "community_validated_shots",
        "training_dataset",
        "community_priors",
        "abuse_events",
    }:
        raise ValueError("unsupported count table")
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def _mark_rejected_uploads_postgres(conn, shots: list[ShotRecord]) -> None:
    shot_ids = [shot.shot_id for shot in shots]
    if not shot_ids:
        return
    rows = conn.execute(
        """
        SELECT DISTINCT local_record_id
        FROM upload_queue
        WHERE local_record_type='shot'
          AND status='rejected'
          AND local_record_id = ANY(%s)
        """,
        (shot_ids,),
    ).fetchall()
    rejected = {row["local_record_id"] for row in rows}
    for shot in shots:
        setattr(shot, "_rejected_upload", shot.shot_id in rejected)


def _count_upload_queue_for_shots_postgres(conn, shot_ids: list[str]) -> int:
    if not shot_ids:
        return 0
    return int(
        conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM upload_queue
            WHERE local_record_type='shot'
              AND local_record_id = ANY(%s)
            """,
            (shot_ids,),
        ).fetchone()["count"]
    )


def _row_to_community_raw_upload(row: dict[str, Any]) -> CommunityRawUpload:
    received_at = row.get("supabase_received_at")
    payload_json = row["payload_json"]
    if isinstance(payload_json, str):
        payload_json = json.loads(payload_json)
    return CommunityRawUpload(
        install_id=row["install_id"],
        upload_id=row["upload_id"],
        payload_hash=row["payload_hash"],
        event_type=row["event_type"],
        payload_json=payload_json,
        received_at=str(received_at) if received_at is not None else None,
    )


def _row_to_training_row(row: dict[str, Any]) -> CommunityTrainingRow:
    payload_json = row["payload_json"]
    if isinstance(payload_json, str):
        payload_json = json.loads(payload_json)
    return CommunityTrainingRow(
        training_row_id=int(row["training_row_id"]),
        source_validation_id=int(row["source_validation_id"]),
        install_id=row["install_id"],
        payload_json=payload_json,
        trust_weight=float(row["trust_weight"]),
        payload_hash=row.get("payload_hash"),
    )


def _row_to_community_prior(row: dict[str, Any]) -> CommunityPrior:
    prior_json = row["prior_json"]
    if isinstance(prior_json, str):
        prior_json = json.loads(prior_json)
    return CommunityPrior(
        context_key=row["context_key"],
        prior_json=prior_json,
        confidence=float(row["confidence"]),
    )


def _row_to_rejection_summary(row: dict[str, Any]) -> CommunityRejectionSummary:
    validation_errors = row["validation_errors"]
    if isinstance(validation_errors, str):
        validation_errors = json.loads(validation_errors)
    if not isinstance(validation_errors, list):
        validation_errors = []
    rejected_at = row.get("rejected_at")
    return CommunityRejectionSummary(
        install_id=row["install_id"],
        upload_id=row["upload_id"],
        event_type=row["event_type"],
        validation_errors=community_rejection_categories(validation_errors[:20]),
        rejected_at=str(rejected_at) if rejected_at is not None else None,
    )


def _row_to_admin_action(row: dict[str, Any]) -> AdminActionLogEntry:
    return AdminActionLogEntry(
        action_type=row["action_type"],
        requested_at=int(row["requested_at"]),
        requested_by=row["requested_by"],
        dry_run=bool(row["dry_run"]),
        status=row["status"],
        rows_seen=int(row["rows_seen"]),
        rows_changed=int(row["rows_changed"]),
        warnings_count=int(row["warnings_count"]),
        error_summary=row.get("error_summary"),
    )
