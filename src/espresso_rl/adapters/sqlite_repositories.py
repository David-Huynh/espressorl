from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from espresso_rl.domain.models import (
    PROFILE_DTYPE,
    PROFILE_SHAPE,
    FollowThroughState,
    Recommendation,
    RecommendationApplyStatus,
    RecommendationDecision,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
    ShotType,
    UploadQueueItem,
    UploadQueueStatus,
)


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shots (
                shot_id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                install_id TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                machine_adapter TEXT NOT NULL,
                bean_context_id TEXT,
                bean_context_name TEXT,
                grinder_context_id TEXT,
                profile_resampled_blob BLOB NOT NULL,
                raw_profile_available INTEGER NOT NULL,
                raw_profile_hash TEXT,
                relative_grind_steps_from_reference REAL,
                relative_grind_um_from_reference REAL,
                microns_per_step REAL NOT NULL,
                dose_in_g REAL NOT NULL,
                beverage_out_g REAL,
                brew_ratio REAL,
                target_yield_g REAL NOT NULL,
                target_ratio REAL,
                shot_time_s REAL,
                recommendation_id TEXT,
                recommended_grind_delta_steps_from_current INTEGER,
                recommended_grind_delta_um_from_current REAL,
                recommended_projected_relative_step_from_reference REAL,
                recommended_dose_g REAL,
                recommended_target_yield_g REAL,
                recommended_target_ratio REAL,
                recommendation_decision TEXT NOT NULL,
                recommendation_followed TEXT NOT NULL,
                recommendation_attribution_weight REAL NOT NULL,
                human_rating INTEGER,
                taste_tags_json TEXT NOT NULL,
                feedback_recorded INTEGER NOT NULL DEFAULT 0,
                profile_score REAL,
                profile_mse REAL,
                reward REAL,
                reward_confidence REAL NOT NULL,
                shot_type TEXT NOT NULL DEFAULT 'espresso',
                exclude_from_local_optimization INTEGER NOT NULL DEFAULT 0,
                optimization_weight REAL NOT NULL DEFAULT 1.0,
                rating_prompt_allowed INTEGER NOT NULL DEFAULT 1,
                grind_followed INTEGER,
                dose_followed INTEGER,
                yield_followed INTEGER,
                grind_recommendation_trust REAL NOT NULL DEFAULT 0.0,
                dose_recommendation_trust REAL NOT NULL DEFAULT 0.0,
                yield_recommendation_trust REAL NOT NULL DEFAULT 0.0,
                weight_source TEXT,
                flow_source TEXT,
                flow_units TEXT,
                pump_flow_source TEXT,
                pump_flow_units TEXT,
                pump_flow_calibration_required INTEGER NOT NULL DEFAULT 0,
                profile_flow_valid INTEGER NOT NULL DEFAULT 1,
                profile_flow_masked INTEGER NOT NULL DEFAULT 0,
                profile_id TEXT,
                profile_label TEXT,
                profile_type TEXT,
                profile_phase_count INTEGER,
                final_phase_index INTEGER,
                final_phase_name TEXT,
                final_phase_type TEXT,
                final_phase_elapsed_s REAL,
                final_pump_target TEXT,
                final_target_pressure REAL,
                final_target_flow REAL,
                final_valve_open INTEGER,
                profile_temperature_c REAL,
                final_phase_temperature_c REAL,
                shot_end_state TEXT,
                grinder_calibration_mode TEXT NOT NULL DEFAULT 'relative_calibrated',
                grinder_step_direction TEXT NOT NULL DEFAULT 'higher_is_finer',
                grinder_reference_label TEXT NOT NULL DEFAULT 'reference',
                current_absolute_step REAL,
                absolute_reference_step REAL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                expires_at INTEGER,
                install_id TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                bean_context_id TEXT,
                grinder_context_id TEXT,
                grind_delta_steps_from_current INTEGER NOT NULL,
                grind_delta_um_from_current REAL NOT NULL,
                projected_relative_step_from_reference REAL NOT NULL,
                projected_relative_grind_um_from_reference REAL NOT NULL,
                next_dose_g REAL NOT NULL,
                target_yield_g REAL NOT NULL,
                target_ratio REAL NOT NULL,
                mode TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                shown_count INTEGER NOT NULL,
                accepted_at INTEGER,
                ignored_at INTEGER,
                edited_at INTEGER,
                used_at INTEGER,
                superseded_at INTEGER,
                source_shot_id TEXT,
                apply_status TEXT NOT NULL DEFAULT 'unknown',
                apply_acknowledged_at INTEGER,
                applied_fields_json TEXT NOT NULL DEFAULT '{}',
                manual_fields_json TEXT NOT NULL DEFAULT '[]',
                apply_error TEXT,
                grinder_calibration_mode TEXT NOT NULL DEFAULT 'relative_calibrated',
                grinder_step_direction TEXT NOT NULL DEFAULT 'higher_is_finer',
                grinder_reference_label TEXT NOT NULL DEFAULT 'reference',
                current_absolute_step REAL,
                absolute_reference_step REAL,
                projected_absolute_step REAL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_queue (
                upload_id TEXT PRIMARY KEY,
                local_record_type TEXT NOT NULL,
                local_record_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                last_attempt_at INTEGER,
                next_retry_at INTEGER,
                error_message TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        self._ensure_column("upload_queue", "payload_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("recommendations", "apply_status", "TEXT NOT NULL DEFAULT 'unknown'")
        self._ensure_column("recommendations", "apply_acknowledged_at", "INTEGER")
        self._ensure_column("recommendations", "applied_fields_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("recommendations", "manual_fields_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("recommendations", "apply_error", "TEXT")
        self._ensure_column("recommendations", "grinder_context_id", "TEXT")
        self._ensure_column("recommendations", "grinder_calibration_mode", "TEXT NOT NULL DEFAULT 'relative_calibrated'")
        self._ensure_column("recommendations", "grinder_step_direction", "TEXT NOT NULL DEFAULT 'higher_is_finer'")
        self._ensure_column("recommendations", "grinder_reference_label", "TEXT NOT NULL DEFAULT 'reference'")
        self._ensure_column("recommendations", "current_absolute_step", "REAL")
        self._ensure_column("recommendations", "absolute_reference_step", "REAL")
        self._ensure_column("recommendations", "projected_absolute_step", "REAL")
        self._ensure_column("shots", "grinder_context_id", "TEXT")
        self._ensure_column("shots", "bean_context_name", "TEXT")
        self._ensure_column("shots", "grinder_calibration_mode", "TEXT NOT NULL DEFAULT 'relative_calibrated'")
        self._ensure_column("shots", "grinder_step_direction", "TEXT NOT NULL DEFAULT 'higher_is_finer'")
        self._ensure_column("shots", "grinder_reference_label", "TEXT NOT NULL DEFAULT 'reference'")
        self._ensure_column("shots", "current_absolute_step", "REAL")
        self._ensure_column("shots", "absolute_reference_step", "REAL")
        self._ensure_column("shots", "shot_type", "TEXT NOT NULL DEFAULT 'espresso'")
        self._ensure_column("shots", "exclude_from_local_optimization", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("shots", "optimization_weight", "REAL NOT NULL DEFAULT 1.0")
        self._ensure_column("shots", "rating_prompt_allowed", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("shots", "feedback_recorded", "INTEGER NOT NULL DEFAULT 0")
        self.conn.execute(
            "UPDATE shots SET feedback_recorded=1 WHERE feedback_recorded=0 AND human_rating IS NOT NULL"
        )
        self._ensure_column("shots", "grind_followed", "INTEGER")
        self._ensure_column("shots", "dose_followed", "INTEGER")
        self._ensure_column("shots", "yield_followed", "INTEGER")
        self._ensure_column("shots", "grind_recommendation_trust", "REAL NOT NULL DEFAULT 0.0")
        self._ensure_column("shots", "dose_recommendation_trust", "REAL NOT NULL DEFAULT 0.0")
        self._ensure_column("shots", "yield_recommendation_trust", "REAL NOT NULL DEFAULT 0.0")
        self._ensure_column("shots", "weight_source", "TEXT")
        self._ensure_column("shots", "flow_source", "TEXT")
        self._ensure_column("shots", "flow_units", "TEXT")
        self._ensure_column("shots", "pump_flow_source", "TEXT")
        self._ensure_column("shots", "pump_flow_units", "TEXT")
        self._ensure_column("shots", "pump_flow_calibration_required", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("shots", "profile_flow_valid", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("shots", "profile_flow_masked", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("shots", "profile_id", "TEXT")
        self._ensure_column("shots", "profile_label", "TEXT")
        self._ensure_column("shots", "profile_type", "TEXT")
        self._ensure_column("shots", "profile_phase_count", "INTEGER")
        self._ensure_column("shots", "final_phase_index", "INTEGER")
        self._ensure_column("shots", "final_phase_name", "TEXT")
        self._ensure_column("shots", "final_phase_type", "TEXT")
        self._ensure_column("shots", "final_phase_elapsed_s", "REAL")
        self._ensure_column("shots", "final_pump_target", "TEXT")
        self._ensure_column("shots", "final_target_pressure", "REAL")
        self._ensure_column("shots", "final_target_flow", "REAL")
        self._ensure_column("shots", "final_valve_open", "INTEGER")
        self._ensure_column("shots", "profile_temperature_c", "REAL")
        self._ensure_column("shots", "final_phase_temperature_c", "REAL")
        self._ensure_column("shots", "shot_end_state", "TEXT")
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_shots_context_grinder_time
            ON shots (install_id, machine_id, bean_context_id, grinder_context_id, timestamp DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recommendations_context_grinder_time
            ON recommendations (install_id, machine_id, bean_context_id, grinder_context_id, created_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_upload_queue_record
            ON upload_queue (local_record_type, local_record_id)
            """
        )
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in rows}
        if column not in existing:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _nullable_clause(
    column: str,
    value: str | None,
    params: list[object],
    placeholder: str,
) -> str:
    if value is None:
        return f"{column} IS NULL"
    params.append(value)
    return f"{column}={placeholder}"


class SQLiteShotRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def upsert(self, shot: ShotRecord) -> None:
        self._store.conn.execute(
            """
            INSERT OR REPLACE INTO shots (
                shot_id, timestamp, install_id, machine_id, machine_adapter,
                bean_context_id, bean_context_name, grinder_context_id, profile_resampled_blob, raw_profile_available,
                raw_profile_hash, relative_grind_steps_from_reference, relative_grind_um_from_reference, microns_per_step,
                dose_in_g, beverage_out_g, brew_ratio, target_yield_g,
                target_ratio, shot_time_s, recommendation_id,
                recommended_grind_delta_steps_from_current, recommended_grind_delta_um_from_current,
                recommended_projected_relative_step_from_reference, recommended_dose_g,
                recommended_target_yield_g, recommended_target_ratio,
                recommendation_decision, recommendation_followed,
                recommendation_attribution_weight, human_rating, taste_tags_json,
                feedback_recorded,
                profile_score, profile_mse, reward, reward_confidence,
                shot_type, exclude_from_local_optimization, optimization_weight,
                rating_prompt_allowed, grind_followed, dose_followed,
                yield_followed, grind_recommendation_trust,
                dose_recommendation_trust, yield_recommendation_trust,
                weight_source, flow_source, flow_units, pump_flow_source,
                pump_flow_units, pump_flow_calibration_required,
                profile_flow_valid, profile_flow_masked, profile_id,
                profile_label, profile_type, profile_phase_count,
                final_phase_index, final_phase_name, final_phase_type,
                final_phase_elapsed_s, final_pump_target,
                final_target_pressure, final_target_flow, final_valve_open,
                profile_temperature_c, final_phase_temperature_c,
                shot_end_state, grinder_calibration_mode,
                grinder_step_direction, grinder_reference_label,
                current_absolute_step, absolute_reference_step,
                created_at, updated_at
            ) VALUES (
                :shot_id, :timestamp, :install_id, :machine_id, :machine_adapter,
                :bean_context_id, :bean_context_name, :grinder_context_id, :profile_resampled_blob, :raw_profile_available,
                :raw_profile_hash, :relative_grind_steps_from_reference, :relative_grind_um_from_reference, :microns_per_step,
                :dose_in_g, :beverage_out_g, :brew_ratio, :target_yield_g,
                :target_ratio, :shot_time_s, :recommendation_id,
                :recommended_grind_delta_steps_from_current, :recommended_grind_delta_um_from_current,
                :recommended_projected_relative_step_from_reference, :recommended_dose_g,
                :recommended_target_yield_g, :recommended_target_ratio,
                :recommendation_decision, :recommendation_followed,
                :recommendation_attribution_weight, :human_rating, :taste_tags_json,
                :feedback_recorded,
                :profile_score, :profile_mse, :reward, :reward_confidence,
                :shot_type, :exclude_from_local_optimization, :optimization_weight,
                :rating_prompt_allowed, :grind_followed, :dose_followed,
                :yield_followed, :grind_recommendation_trust,
                :dose_recommendation_trust, :yield_recommendation_trust,
                :weight_source, :flow_source, :flow_units, :pump_flow_source,
                :pump_flow_units, :pump_flow_calibration_required,
                :profile_flow_valid, :profile_flow_masked, :profile_id,
                :profile_label, :profile_type, :profile_phase_count,
                :final_phase_index, :final_phase_name, :final_phase_type,
                :final_phase_elapsed_s, :final_pump_target,
                :final_target_pressure, :final_target_flow, :final_valve_open,
                :profile_temperature_c, :final_phase_temperature_c,
                :shot_end_state, :grinder_calibration_mode,
                :grinder_step_direction, :grinder_reference_label,
                :current_absolute_step, :absolute_reference_step,
                :created_at, :updated_at
            )
            """,
            _shot_to_row(shot),
        )
        self._store.conn.commit()

    def get(self, shot_id: str) -> ShotRecord | None:
        row = self._store.conn.execute("SELECT * FROM shots WHERE shot_id=?", (shot_id,)).fetchone()
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
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params, "?")
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params, "?")
        params.append(limit)
        rows = self._store.conn.execute(
            f"""
            SELECT * FROM shots
            WHERE install_id=? AND machine_id=? AND {bean_clause} AND {grinder_clause}
            ORDER BY timestamp DESC
            LIMIT ?
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
            WHERE install_id=? AND machine_id=?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (install_id, machine_id, limit),
        ).fetchall()
        return list(reversed([_row_to_shot(row) for row in rows]))


class SQLiteLocalDataRepository:
    def __init__(self, store: SQLiteStore) -> None:
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
            WHERE install_id=? AND machine_id=?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (install_id, machine_id, limit),
        ).fetchall()
        shots = list(reversed([_row_to_shot(row) for row in rows]))
        _mark_rejected_uploads_sqlite(self._store.conn, shots)
        return shots

    def delete_shot(
        self,
        install_id: str,
        machine_id: str,
        shot_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, int]:
        shot_count = int(
            self._store.conn.execute(
                """
                SELECT COUNT(*) AS count FROM shots
                WHERE install_id=? AND machine_id=? AND shot_id=?
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
                    WHERE local_record_type='shot' AND local_record_id=?
                    """,
                    (shot_id,),
                ).fetchone()["count"]
            )
        if dry_run:
            return {"shots": shot_count, "upload_queue": upload_count}
        if shot_count:
            self._store.conn.execute(
                "DELETE FROM upload_queue WHERE local_record_type='shot' AND local_record_id=?",
                (shot_id,),
            )
            self._store.conn.execute(
                "DELETE FROM shots WHERE install_id=? AND machine_id=? AND shot_id=?",
                (install_id, machine_id, shot_id),
            )
        self._store.conn.commit()
        return {"shots": shot_count, "upload_queue": upload_count}

    def exclude_shot_from_optimization(
        self,
        install_id: str,
        machine_id: str,
        shot_id: str,
        *,
        now: int,
        dry_run: bool = False,
    ) -> dict[str, int]:
        shot_count = int(
            self._store.conn.execute(
                """
                SELECT COUNT(*) AS count FROM shots
                WHERE install_id=? AND machine_id=? AND shot_id=?
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
                SET exclude_from_local_optimization=1,
                    optimization_weight=0,
                    recommendation_attribution_weight=0,
                    updated_at=?
                WHERE install_id=? AND machine_id=? AND shot_id=?
                """,
                (now, install_id, machine_id, shot_id),
            )
        self._store.conn.commit()
        return {"shots": shot_count}

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
        context_clause = ""
        params: list[object] = [install_id, machine_id]
        if bean_context_id is not None or grinder_context_id is not None:
            clauses = [
                _nullable_clause("bean_context_id", bean_context_id, params, "?"),
                _nullable_clause("grinder_context_id", grinder_context_id, params, "?"),
            ]
            context_clause = "AND " + " AND ".join(clauses)
        params.append(limit)
        rows = self._store.conn.execute(
            f"""
            SELECT shot_id FROM shots
            WHERE install_id=? AND machine_id=?
              {context_clause}
              AND (
                shot_type != 'espresso'
                OR exclude_from_local_optimization = 1
                OR optimization_weight <= 0
              )
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        shot_ids = [row["shot_id"] for row in rows]
        upload_count = _count_upload_queue_for_shots_sqlite(self._store.conn, shot_ids)
        if dry_run:
            return {"shots": len(shot_ids), "upload_queue": upload_count}
        for shot_id in shot_ids:
            self._store.conn.execute(
                "DELETE FROM upload_queue WHERE local_record_type='shot' AND local_record_id=?",
                (shot_id,),
            )
            self._store.conn.execute(
                "DELETE FROM shots WHERE shot_id=? AND install_id=? AND machine_id=?",
                (shot_id, install_id, machine_id),
            )
        self._store.conn.commit()
        return {"shots": len(shot_ids), "upload_queue": upload_count}

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
        context_params: list[object] = [install_id, machine_id, bean_context_id]
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, context_params, "?")
        shot_count = int(
            self._store.conn.execute(
                f"""
                SELECT COUNT(*) AS count FROM shots
                WHERE install_id=? AND machine_id=? AND bean_context_id=? AND {grinder_clause}
                  AND (
                    exclude_from_local_optimization = 0
                    OR optimization_weight > 0
                    OR recommendation_attribution_weight > 0
                  )
                """,
                tuple(context_params),
            ).fetchone()["count"]
        )
        rec_context_params: list[object] = [install_id, machine_id, bean_context_id]
        rec_grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, rec_context_params, "?")
        recommendation_count = int(
            self._store.conn.execute(
                f"""
                SELECT COUNT(*) AS count FROM recommendations
                WHERE install_id=? AND machine_id=? AND bean_context_id=? AND {rec_grinder_clause}
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
            SET exclude_from_local_optimization=1,
                optimization_weight=0,
                recommendation_attribution_weight=0,
                updated_at=?
            WHERE install_id=? AND machine_id=? AND bean_context_id=? AND {grinder_clause}
            """,
            (now, *context_params),
        )
        self._store.conn.execute(
            f"""
            UPDATE recommendations
            SET status='superseded',
                superseded_at=COALESCE(superseded_at, ?),
                updated_at=?
            WHERE install_id=? AND machine_id=? AND bean_context_id=? AND {rec_grinder_clause}
              AND status IN ('pending', 'shown', 'accepted', 'edited')
            """,
            (now, now, *rec_context_params),
        )
        self._store.conn.commit()
        return {"shots": shot_count, "recommendations": recommendation_count}


class SQLiteRecommendationRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def upsert(self, recommendation: Recommendation) -> None:
        self._store.conn.execute(
            """
            INSERT OR REPLACE INTO recommendations (
                recommendation_id, created_at, updated_at, expires_at,
                install_id, machine_id, bean_context_id, grinder_context_id,
                grind_delta_steps_from_current, grind_delta_um_from_current, projected_relative_step_from_reference, projected_relative_grind_um_from_reference, next_dose_g,
                target_yield_g, target_ratio, mode, confidence, reason, status,
                shown_count, accepted_at, ignored_at, edited_at, used_at,
                superseded_at, source_shot_id, apply_status,
                apply_acknowledged_at, applied_fields_json, manual_fields_json,
                apply_error, grinder_calibration_mode, grinder_step_direction,
                grinder_reference_label, current_absolute_step,
                absolute_reference_step, projected_absolute_step
            ) VALUES (
                :recommendation_id, :created_at, :updated_at, :expires_at,
                :install_id, :machine_id, :bean_context_id, :grinder_context_id,
                :grind_delta_steps_from_current, :grind_delta_um_from_current, :projected_relative_step_from_reference, :projected_relative_grind_um_from_reference, :next_dose_g,
                :target_yield_g, :target_ratio, :mode, :confidence, :reason, :status,
                :shown_count, :accepted_at, :ignored_at, :edited_at, :used_at,
                :superseded_at, :source_shot_id, :apply_status,
                :apply_acknowledged_at, :applied_fields_json, :manual_fields_json,
                :apply_error, :grinder_calibration_mode, :grinder_step_direction,
                :grinder_reference_label, :current_absolute_step,
                :absolute_reference_step, :projected_absolute_step
            )
            """,
            _recommendation_to_row(recommendation),
        )
        self._store.conn.commit()

    def get(self, recommendation_id: str) -> Recommendation | None:
        row = self._store.conn.execute(
            "SELECT * FROM recommendations WHERE recommendation_id=?",
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
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params, "?")
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params, "?")
        row = self._store.conn.execute(
            f"""
            SELECT * FROM recommendations
            WHERE install_id=? AND machine_id=? AND {bean_clause} AND {grinder_clause}
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
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params, "?")
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params, "?")
        params.append(now)
        row = self._store.conn.execute(
            f"""
            SELECT * FROM recommendations
            WHERE install_id=? AND machine_id=? AND {bean_clause} AND {grinder_clause}
              AND status IN ('pending', 'shown', 'accepted', 'edited')
              AND (expires_at IS NULL OR expires_at > ?)
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
        params: list = [now, now, install_id, machine_id]
        bean_clause = _nullable_clause("bean_context_id", bean_context_id, params, "?")
        grinder_clause = _nullable_clause("grinder_context_id", grinder_context_id, params, "?")
        except_clause = ""
        if except_recommendation_id is not None:
            except_clause = "AND recommendation_id != ?"
            params.append(except_recommendation_id)
        self._store.conn.execute(
            f"""
            UPDATE recommendations
            SET status='superseded', superseded_at=?, updated_at=?
            WHERE install_id=? AND machine_id=? AND {bean_clause} AND {grinder_clause}
              AND status IN ('pending', 'shown')
              {except_clause}
            """,
            tuple(params),
        )
        self._store.conn.commit()


class SQLiteUploadQueueRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def enqueue(self, item: UploadQueueItem) -> None:
        conn = self._store.conn
        # Idempotency: this exact content was already uploaded; nothing to do.
        if conn.execute(
            """
            SELECT 1 FROM upload_queue
            WHERE local_record_type=? AND local_record_id=? AND payload_hash=? AND status=?
            LIMIT 1
            """,
            (
                item.local_record_type,
                item.local_record_id,
                item.payload_hash,
                UploadQueueStatus.UPLOADED.value,
            ),
        ).fetchone():
            return
        # Never clobber an in-flight send of this exact content.
        if conn.execute(
            "SELECT 1 FROM upload_queue WHERE upload_id=? AND status=? LIMIT 1",
            (item.upload_id, UploadQueueStatus.UPLOADING.value),
        ).fetchone():
            return
        # Coalesce: drop superseded unsent versions for this record so the latest
        # state replaces them. Rejected rows are also rearmed by a new snapshot,
        # which lets a payload/schema fix drain local data without manual SQL.
        # UPLOADING rows are left untouched (a worker may be mid-send) and
        # UPLOADED rows are kept as the "already sent" memory.
        conn.execute(
            """
            DELETE FROM upload_queue
            WHERE local_record_type=? AND local_record_id=? AND status IN (?, ?, ?)
            """,
            (
                item.local_record_type,
                item.local_record_id,
                UploadQueueStatus.PENDING.value,
                UploadQueueStatus.FAILED.value,
                UploadQueueStatus.REJECTED.value,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO upload_queue (
                upload_id, local_record_type, local_record_id, payload_hash, payload_json,
                status, attempt_count, last_attempt_at, next_retry_at,
                error_message, created_at, updated_at
            ) VALUES (
                :upload_id, :local_record_type, :local_record_id, :payload_hash, :payload_json,
                :status, :attempt_count, :last_attempt_at, :next_retry_at,
                :error_message, :created_at, :updated_at
            )
            """,
            _upload_item_to_row(item),
        )
        conn.commit()

    def list_ready(self, now: int, limit: int = 100) -> list[UploadQueueItem]:
        rows = self._store.conn.execute(
            """
            SELECT * FROM upload_queue
            WHERE status IN ('pending', 'failed')
              AND (next_retry_at IS NULL OR next_retry_at <= ?)
            ORDER BY created_at ASC
            LIMIT ?
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
            "SELECT attempt_count FROM upload_queue WHERE upload_id=?",
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
            SET status=?, attempt_count=?, last_attempt_at=COALESCE(?, last_attempt_at),
                next_retry_at=?, error_message=?, updated_at=?
            WHERE upload_id=?
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
            WHERE status=?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
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
            "SELECT 1 FROM upload_queue WHERE upload_id=? AND status=?",
            (upload_id, UploadQueueStatus.REJECTED.value),
        ).fetchone()
        if existing is None:
            raise ValueError(f"unknown rejected upload_id {upload_id}")
        self._store.conn.execute(
            """
            UPDATE upload_queue
            SET status=?, attempt_count=0, last_attempt_at=NULL, next_retry_at=NULL,
                error_message=?, updated_at=?
            WHERE upload_id=?
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
            "SELECT 1 FROM upload_queue WHERE upload_id=? AND status=?",
            (upload_id, UploadQueueStatus.REJECTED.value),
        ).fetchone()
        if existing is None:
            raise ValueError(f"unknown rejected upload_id {upload_id}")
        self._store.conn.execute(
            """
            UPDATE upload_queue
            SET error_message=?, updated_at=?
            WHERE upload_id=? AND status=?
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
        del now
        conn = self._store.conn
        params: list[object] = [UploadQueueStatus.REJECTED.value]
        record_filter = ""
        if local_record_id:
            record_filter = "AND local_record_id=?"
            params.append(local_record_id)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT upload_id, local_record_type, local_record_id
            FROM upload_queue
            WHERE status=?
              {record_filter}
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        inspected = len(rows)
        purged_uploads = 0
        purged_shots = 0
        purged_recommendations = 0
        kept_linked_records = 0

        for row in rows:
            record_type = row["local_record_type"]
            record_id = row["local_record_id"]
            deleted_linked = False
            if record_type == "shot":
                cursor = conn.execute(
                    "DELETE FROM shots WHERE shot_id=?",
                    (record_id,),
                )
                if cursor.rowcount:
                    purged_shots += cursor.rowcount
                    deleted_linked = True
            elif record_type == "recommendation":
                cursor = conn.execute(
                    """
                    DELETE FROM recommendations
                    WHERE recommendation_id=?
                      AND status IN ('ignored', 'expired', 'superseded')
                      AND NOT EXISTS (
                        SELECT 1 FROM shots
                        WHERE shots.recommendation_id = recommendations.recommendation_id
                      )
                    """,
                    (record_id,),
                )
                if cursor.rowcount:
                    purged_recommendations += cursor.rowcount
                    deleted_linked = True
            if record_type in {"shot", "recommendation"} and not deleted_linked:
                kept_linked_records += 1

            cursor = conn.execute(
                "DELETE FROM upload_queue WHERE upload_id=? AND status=?",
                (row["upload_id"], UploadQueueStatus.REJECTED.value),
            )
            purged_uploads += cursor.rowcount

        conn.commit()
        return {
            "inspected": inspected,
            "purged_uploads": purged_uploads,
            "purged_shots": purged_shots,
            "purged_recommendations": purged_recommendations,
            "kept_linked_records": kept_linked_records,
        }


def _shot_to_row(shot: ShotRecord) -> dict:
    return {
        "shot_id": shot.shot_id,
        "timestamp": shot.timestamp,
        "install_id": shot.install_id,
        "machine_id": shot.machine_id,
        "machine_adapter": shot.machine_adapter,
        "bean_context_id": shot.bean_context_id,
        "bean_context_name": shot.bean_context_name,
        "grinder_context_id": shot.grinder_context_id,
        "profile_resampled_blob": shot.profile.astype(PROFILE_DTYPE).tobytes(),
        "raw_profile_available": bool(shot.raw_profile_available),
        "raw_profile_hash": shot.raw_profile_hash,
        "relative_grind_steps_from_reference": shot.relative_grind_steps_from_reference,
        "relative_grind_um_from_reference": shot.relative_grind_um_from_reference,
        "microns_per_step": shot.microns_per_step,
        "dose_in_g": shot.dose_in_g,
        "beverage_out_g": shot.beverage_out_g,
        "brew_ratio": shot.brew_ratio,
        "target_yield_g": shot.target_yield_g,
        "target_ratio": shot.target_ratio,
        "shot_time_s": shot.shot_time_s,
        "recommendation_id": shot.recommendation_id,
        "recommended_grind_delta_steps_from_current": shot.recommended_grind_delta_steps_from_current,
        "recommended_grind_delta_um_from_current": shot.recommended_grind_delta_um_from_current,
        "recommended_projected_relative_step_from_reference": shot.recommended_projected_relative_step_from_reference,
        "recommended_dose_g": shot.recommended_dose_g,
        "recommended_target_yield_g": shot.recommended_target_yield_g,
        "recommended_target_ratio": shot.recommended_target_ratio,
        "recommendation_decision": shot.recommendation_decision.value,
        "recommendation_followed": shot.recommendation_followed.value,
        "recommendation_attribution_weight": shot.recommendation_attribution_weight,
        "human_rating": shot.human_rating,
        "taste_tags_json": json.dumps(shot.taste_tags),
        "feedback_recorded": bool(shot.feedback_recorded),
        "profile_score": shot.profile_score,
        "profile_mse": shot.profile_mse,
        "reward": shot.reward,
        "reward_confidence": shot.reward_confidence,
        "shot_type": shot.shot_type.value,
        "exclude_from_local_optimization": bool(shot.exclude_from_local_optimization),
        "optimization_weight": shot.optimization_weight,
        "rating_prompt_allowed": bool(shot.rating_prompt_allowed),
        "grind_followed": shot.grind_followed,
        "dose_followed": shot.dose_followed,
        "yield_followed": shot.yield_followed,
        "grind_recommendation_trust": shot.grind_recommendation_trust,
        "dose_recommendation_trust": shot.dose_recommendation_trust,
        "yield_recommendation_trust": shot.yield_recommendation_trust,
        "weight_source": shot.weight_source,
        "flow_source": shot.flow_source,
        "flow_units": shot.flow_units,
        "pump_flow_source": shot.pump_flow_source,
        "pump_flow_units": shot.pump_flow_units,
        "pump_flow_calibration_required": bool(shot.pump_flow_calibration_required),
        "profile_flow_valid": bool(shot.profile_flow_valid),
        "profile_flow_masked": bool(shot.profile_flow_masked),
        "profile_id": shot.profile_id,
        "profile_label": shot.profile_label,
        "profile_type": shot.profile_type,
        "profile_phase_count": shot.profile_phase_count,
        "final_phase_index": shot.final_phase_index,
        "final_phase_name": shot.final_phase_name,
        "final_phase_type": shot.final_phase_type,
        "final_phase_elapsed_s": shot.final_phase_elapsed_s,
        "final_pump_target": shot.final_pump_target,
        "final_target_pressure": shot.final_target_pressure,
        "final_target_flow": shot.final_target_flow,
        "final_valve_open": shot.final_valve_open,
        "profile_temperature_c": shot.profile_temperature_c,
        "final_phase_temperature_c": shot.final_phase_temperature_c,
        "shot_end_state": shot.shot_end_state,
        "grinder_calibration_mode": shot.grinder_calibration_mode.value,
        "grinder_step_direction": shot.grinder_step_direction.value,
        "grinder_reference_label": shot.grinder_reference_label,
        "current_absolute_step": shot.current_absolute_step,
        "absolute_reference_step": shot.absolute_reference_step,
        "created_at": shot.created_at,
        "updated_at": shot.updated_at,
    }


def _row_to_shot(row: sqlite3.Row) -> ShotRecord:
    profile = np.frombuffer(row["profile_resampled_blob"], dtype=PROFILE_DTYPE).reshape(PROFILE_SHAPE)
    return ShotRecord(
        shot_id=row["shot_id"],
        timestamp=row["timestamp"],
        install_id=row["install_id"],
        machine_id=row["machine_id"],
        machine_adapter=row["machine_adapter"],
        bean_context_id=row["bean_context_id"],
        bean_context_name=row["bean_context_name"],
        grinder_context_id=row["grinder_context_id"],
        profile=profile.copy(),
        raw_profile_available=bool(row["raw_profile_available"]),
        raw_profile_hash=row["raw_profile_hash"],
        relative_grind_steps_from_reference=row["relative_grind_steps_from_reference"],
        relative_grind_um_from_reference=row["relative_grind_um_from_reference"],
        microns_per_step=row["microns_per_step"],
        dose_in_g=row["dose_in_g"],
        beverage_out_g=row["beverage_out_g"],
        brew_ratio=row["brew_ratio"],
        target_yield_g=row["target_yield_g"],
        target_ratio=row["target_ratio"],
        shot_time_s=row["shot_time_s"],
        recommendation_id=row["recommendation_id"],
        recommended_grind_delta_steps_from_current=row["recommended_grind_delta_steps_from_current"],
        recommended_grind_delta_um_from_current=row["recommended_grind_delta_um_from_current"],
        recommended_projected_relative_step_from_reference=row["recommended_projected_relative_step_from_reference"],
        recommended_dose_g=row["recommended_dose_g"],
        recommended_target_yield_g=row["recommended_target_yield_g"],
        recommended_target_ratio=row["recommended_target_ratio"],
        recommendation_decision=RecommendationDecision(row["recommendation_decision"]),
        recommendation_followed=FollowThroughState(row["recommendation_followed"]),
        recommendation_attribution_weight=row["recommendation_attribution_weight"],
        human_rating=row["human_rating"],
        taste_tags=json.loads(row["taste_tags_json"]),
        feedback_recorded=bool(row["feedback_recorded"]),
        profile_score=row["profile_score"],
        profile_mse=row["profile_mse"],
        reward=row["reward"],
        reward_confidence=row["reward_confidence"],
        shot_type=ShotType(row["shot_type"]),
        exclude_from_local_optimization=bool(row["exclude_from_local_optimization"]),
        optimization_weight=row["optimization_weight"],
        rating_prompt_allowed=bool(row["rating_prompt_allowed"]),
        grind_followed=_optional_int_to_bool(row["grind_followed"]),
        dose_followed=_optional_int_to_bool(row["dose_followed"]),
        yield_followed=_optional_int_to_bool(row["yield_followed"]),
        grind_recommendation_trust=row["grind_recommendation_trust"],
        dose_recommendation_trust=row["dose_recommendation_trust"],
        yield_recommendation_trust=row["yield_recommendation_trust"],
        weight_source=row["weight_source"],
        flow_source=row["flow_source"],
        flow_units=row["flow_units"],
        pump_flow_source=row["pump_flow_source"],
        pump_flow_units=row["pump_flow_units"],
        pump_flow_calibration_required=bool(row["pump_flow_calibration_required"]),
        profile_flow_valid=bool(row["profile_flow_valid"]),
        profile_flow_masked=bool(row["profile_flow_masked"]),
        profile_id=row["profile_id"],
        profile_label=row["profile_label"],
        profile_type=row["profile_type"],
        profile_phase_count=row["profile_phase_count"],
        final_phase_index=row["final_phase_index"],
        final_phase_name=row["final_phase_name"],
        final_phase_type=row["final_phase_type"],
        final_phase_elapsed_s=row["final_phase_elapsed_s"],
        final_pump_target=row["final_pump_target"],
        final_target_pressure=row["final_target_pressure"],
        final_target_flow=row["final_target_flow"],
        final_valve_open=_optional_int_to_bool(row["final_valve_open"]),
        profile_temperature_c=row["profile_temperature_c"],
        final_phase_temperature_c=row["final_phase_temperature_c"],
        shot_end_state=row["shot_end_state"],
        grinder_calibration_mode=row["grinder_calibration_mode"],
        grinder_step_direction=row["grinder_step_direction"],
        grinder_reference_label=row["grinder_reference_label"],
        current_absolute_step=row["current_absolute_step"],
        absolute_reference_step=row["absolute_reference_step"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _optional_bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _optional_int_to_bool(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _recommendation_to_row(recommendation: Recommendation) -> dict:
    return {
        "recommendation_id": recommendation.recommendation_id,
        "created_at": recommendation.created_at,
        "updated_at": recommendation.updated_at,
        "expires_at": recommendation.expires_at,
        "install_id": recommendation.install_id,
        "machine_id": recommendation.machine_id,
        "bean_context_id": recommendation.bean_context_id,
        "grinder_context_id": recommendation.grinder_context_id,
        "grind_delta_steps_from_current": recommendation.grind_delta_steps_from_current,
        "grind_delta_um_from_current": recommendation.grind_delta_um_from_current,
        "projected_relative_step_from_reference": recommendation.projected_relative_step_from_reference,
        "projected_relative_grind_um_from_reference": recommendation.projected_relative_grind_um_from_reference,
        "next_dose_g": recommendation.next_dose_g,
        "target_yield_g": recommendation.target_yield_g,
        "target_ratio": recommendation.target_ratio,
        "mode": recommendation.mode.value,
        "confidence": recommendation.confidence,
        "reason": recommendation.reason,
        "status": recommendation.status.value,
        "shown_count": recommendation.shown_count,
        "accepted_at": recommendation.accepted_at,
        "ignored_at": recommendation.ignored_at,
        "edited_at": recommendation.edited_at,
        "used_at": recommendation.used_at,
        "superseded_at": recommendation.superseded_at,
        "source_shot_id": recommendation.source_shot_id,
        "apply_status": recommendation.apply_status.value,
        "apply_acknowledged_at": recommendation.apply_acknowledged_at,
        "applied_fields_json": json.dumps(recommendation.applied_fields),
        "manual_fields_json": json.dumps(recommendation.manual_fields),
        "apply_error": recommendation.apply_error,
        "grinder_calibration_mode": recommendation.grinder_calibration_mode.value,
        "grinder_step_direction": recommendation.grinder_step_direction.value,
        "grinder_reference_label": recommendation.grinder_reference_label,
        "current_absolute_step": recommendation.current_absolute_step,
        "absolute_reference_step": recommendation.absolute_reference_step,
        "projected_absolute_step": recommendation.projected_absolute_step,
    }


def _row_to_recommendation(row: sqlite3.Row) -> Recommendation:
    return Recommendation(
        recommendation_id=row["recommendation_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
        install_id=row["install_id"],
        machine_id=row["machine_id"],
        bean_context_id=row["bean_context_id"],
        grinder_context_id=row["grinder_context_id"],
        grind_delta_steps_from_current=row["grind_delta_steps_from_current"],
        grind_delta_um_from_current=row["grind_delta_um_from_current"],
        projected_relative_step_from_reference=row["projected_relative_step_from_reference"],
        projected_relative_grind_um_from_reference=row["projected_relative_grind_um_from_reference"],
        next_dose_g=row["next_dose_g"],
        target_yield_g=row["target_yield_g"],
        target_ratio=row["target_ratio"],
        mode=RecommendationMode(row["mode"]),
        confidence=row["confidence"],
        reason=row["reason"],
        status=RecommendationStatus(row["status"]),
        shown_count=row["shown_count"],
        accepted_at=row["accepted_at"],
        ignored_at=row["ignored_at"],
        edited_at=row["edited_at"],
        used_at=row["used_at"],
        superseded_at=row["superseded_at"],
        source_shot_id=row["source_shot_id"],
        apply_status=RecommendationApplyStatus(row["apply_status"]),
        apply_acknowledged_at=row["apply_acknowledged_at"],
        applied_fields=json.loads(row["applied_fields_json"]),
        manual_fields=json.loads(row["manual_fields_json"]),
        apply_error=row["apply_error"],
        grinder_calibration_mode=row["grinder_calibration_mode"],
        grinder_step_direction=row["grinder_step_direction"],
        grinder_reference_label=row["grinder_reference_label"],
        current_absolute_step=row["current_absolute_step"],
        absolute_reference_step=row["absolute_reference_step"],
        projected_absolute_step=row["projected_absolute_step"],
    )


def _mark_rejected_uploads_sqlite(conn: sqlite3.Connection, shots: list[ShotRecord]) -> None:
    shot_ids = [shot.shot_id for shot in shots]
    if not shot_ids:
        return
    placeholders = ",".join("?" for _ in shot_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT local_record_id
        FROM upload_queue
        WHERE local_record_type='shot'
          AND status='rejected'
          AND local_record_id IN ({placeholders})
        """,
        tuple(shot_ids),
    ).fetchall()
    rejected = {row["local_record_id"] for row in rows}
    for shot in shots:
        setattr(shot, "_rejected_upload", shot.shot_id in rejected)


def _count_upload_queue_for_shots_sqlite(conn: sqlite3.Connection, shot_ids: list[str]) -> int:
    if not shot_ids:
        return 0
    placeholders = ",".join("?" for _ in shot_ids)
    return int(
        conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM upload_queue
            WHERE local_record_type='shot'
              AND local_record_id IN ({placeholders})
            """,
            tuple(shot_ids),
        ).fetchone()["count"]
    )


def _upload_item_to_row(item: UploadQueueItem) -> dict:
    return {
        "upload_id": item.upload_id,
        "local_record_type": item.local_record_type,
        "local_record_id": item.local_record_id,
        "payload_hash": item.payload_hash,
        "payload_json": item.payload_json,
        "status": item.status.value,
        "attempt_count": item.attempt_count,
        "last_attempt_at": item.last_attempt_at,
        "next_retry_at": item.next_retry_at,
        "error_message": item.error_message,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def _row_to_upload_item(row: sqlite3.Row) -> UploadQueueItem:
    return UploadQueueItem(
        upload_id=row["upload_id"],
        local_record_type=row["local_record_type"],
        local_record_id=row["local_record_id"],
        payload_hash=row["payload_hash"],
        payload_json=row["payload_json"],
        status=UploadQueueStatus(row["status"]),
        attempt_count=row["attempt_count"],
        last_attempt_at=row["last_attempt_at"],
        next_retry_at=row["next_retry_at"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
