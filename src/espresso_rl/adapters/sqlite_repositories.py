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
    UploadQueueItem,
    UploadQueueStatus,
)


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

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
                profile_resampled_blob BLOB NOT NULL,
                raw_profile_available INTEGER NOT NULL,
                raw_profile_hash TEXT,
                grind_steps REAL,
                grind_um REAL,
                grinder_step_size_um REAL NOT NULL,
                dose_in_g REAL NOT NULL,
                beverage_out_g REAL,
                brew_ratio REAL,
                target_yield_g REAL NOT NULL,
                target_ratio REAL,
                shot_time_s REAL,
                recommendation_id TEXT,
                recommended_grind_delta_steps INTEGER,
                recommended_grind_delta_um REAL,
                recommended_next_grind_steps REAL,
                recommended_dose_g REAL,
                recommended_target_yield_g REAL,
                recommended_target_ratio REAL,
                recommendation_decision TEXT NOT NULL,
                recommendation_followed TEXT NOT NULL,
                recommendation_attribution_weight REAL NOT NULL,
                human_rating INTEGER,
                taste_tags_json TEXT NOT NULL,
                profile_score REAL,
                profile_mse REAL,
                reward REAL,
                reward_confidence REAL NOT NULL,
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
                grind_delta_steps INTEGER NOT NULL,
                grind_delta_um REAL NOT NULL,
                next_grind_steps REAL NOT NULL,
                next_grind_um REAL NOT NULL,
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
                apply_error TEXT
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
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in rows}
        if column not in existing:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


class SQLiteShotRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def upsert(self, shot: ShotRecord) -> None:
        self._store.conn.execute(
            """
            INSERT OR REPLACE INTO shots (
                shot_id, timestamp, install_id, machine_id, machine_adapter,
                bean_context_id, profile_resampled_blob, raw_profile_available,
                raw_profile_hash, grind_steps, grind_um, grinder_step_size_um,
                dose_in_g, beverage_out_g, brew_ratio, target_yield_g,
                target_ratio, shot_time_s, recommendation_id,
                recommended_grind_delta_steps, recommended_grind_delta_um,
                recommended_next_grind_steps, recommended_dose_g,
                recommended_target_yield_g, recommended_target_ratio,
                recommendation_decision, recommendation_followed,
                recommendation_attribution_weight, human_rating, taste_tags_json,
                profile_score, profile_mse, reward, reward_confidence,
                created_at, updated_at
            ) VALUES (
                :shot_id, :timestamp, :install_id, :machine_id, :machine_adapter,
                :bean_context_id, :profile_resampled_blob, :raw_profile_available,
                :raw_profile_hash, :grind_steps, :grind_um, :grinder_step_size_um,
                :dose_in_g, :beverage_out_g, :brew_ratio, :target_yield_g,
                :target_ratio, :shot_time_s, :recommendation_id,
                :recommended_grind_delta_steps, :recommended_grind_delta_um,
                :recommended_next_grind_steps, :recommended_dose_g,
                :recommended_target_yield_g, :recommended_target_ratio,
                :recommendation_decision, :recommendation_followed,
                :recommendation_attribution_weight, :human_rating, :taste_tags_json,
                :profile_score, :profile_mse, :reward, :reward_confidence,
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
    ) -> list[ShotRecord]:
        if bean_context_id is None:
            rows = self._store.conn.execute(
                """
                SELECT * FROM shots
                WHERE install_id=? AND machine_id=? AND bean_context_id IS NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (install_id, machine_id, limit),
            ).fetchall()
        else:
            rows = self._store.conn.execute(
                """
                SELECT * FROM shots
                WHERE install_id=? AND machine_id=? AND bean_context_id=?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (install_id, machine_id, bean_context_id, limit),
            ).fetchall()
        return list(reversed([_row_to_shot(row) for row in rows]))


class SQLiteRecommendationRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def upsert(self, recommendation: Recommendation) -> None:
        self._store.conn.execute(
            """
            INSERT OR REPLACE INTO recommendations (
                recommendation_id, created_at, updated_at, expires_at,
                install_id, machine_id, bean_context_id, grind_delta_steps,
                grind_delta_um, next_grind_steps, next_grind_um, next_dose_g,
                target_yield_g, target_ratio, mode, confidence, reason, status,
                shown_count, accepted_at, ignored_at, edited_at, used_at,
                superseded_at, source_shot_id, apply_status,
                apply_acknowledged_at, applied_fields_json, manual_fields_json,
                apply_error
            ) VALUES (
                :recommendation_id, :created_at, :updated_at, :expires_at,
                :install_id, :machine_id, :bean_context_id, :grind_delta_steps,
                :grind_delta_um, :next_grind_steps, :next_grind_um, :next_dose_g,
                :target_yield_g, :target_ratio, :mode, :confidence, :reason, :status,
                :shown_count, :accepted_at, :ignored_at, :edited_at, :used_at,
                :superseded_at, :source_shot_id, :apply_status,
                :apply_acknowledged_at, :applied_fields_json, :manual_fields_json,
                :apply_error
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

    def get_current(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
    ) -> Recommendation | None:
        params: tuple
        if bean_context_id is None:
            bean_clause = "bean_context_id IS NULL"
            params = (install_id, machine_id, now)
        else:
            bean_clause = "bean_context_id=?"
            params = (install_id, machine_id, bean_context_id, now)
        row = self._store.conn.execute(
            f"""
            SELECT * FROM recommendations
            WHERE install_id=? AND machine_id=? AND {bean_clause}
              AND status IN ('pending', 'shown', 'accepted', 'edited')
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return _row_to_recommendation(row) if row else None

    def supersede_active(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
        except_recommendation_id: str | None = None,
    ) -> None:
        params: list = [now, now, install_id, machine_id]
        bean_clause = "bean_context_id IS NULL" if bean_context_id is None else "bean_context_id=?"
        if bean_context_id is not None:
            params.append(bean_context_id)
        except_clause = ""
        if except_recommendation_id is not None:
            except_clause = "AND recommendation_id != ?"
            params.append(except_recommendation_id)
        self._store.conn.execute(
            f"""
            UPDATE recommendations
            SET status='superseded', superseded_at=?, updated_at=?
            WHERE install_id=? AND machine_id=? AND {bean_clause}
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
        self._store.conn.execute(
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
        self._store.conn.commit()

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
        if status in {UploadQueueStatus.UPLOADING, UploadQueueStatus.FAILED, UploadQueueStatus.REJECTED}:
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


def _shot_to_row(shot: ShotRecord) -> dict:
    return {
        "shot_id": shot.shot_id,
        "timestamp": shot.timestamp,
        "install_id": shot.install_id,
        "machine_id": shot.machine_id,
        "machine_adapter": shot.machine_adapter,
        "bean_context_id": shot.bean_context_id,
        "profile_resampled_blob": shot.profile.astype(PROFILE_DTYPE).tobytes(),
        "raw_profile_available": bool(shot.raw_profile_available),
        "raw_profile_hash": shot.raw_profile_hash,
        "grind_steps": shot.grind_steps,
        "grind_um": shot.grind_um,
        "grinder_step_size_um": shot.grinder_step_size_um,
        "dose_in_g": shot.dose_in_g,
        "beverage_out_g": shot.beverage_out_g,
        "brew_ratio": shot.brew_ratio,
        "target_yield_g": shot.target_yield_g,
        "target_ratio": shot.target_ratio,
        "shot_time_s": shot.shot_time_s,
        "recommendation_id": shot.recommendation_id,
        "recommended_grind_delta_steps": shot.recommended_grind_delta_steps,
        "recommended_grind_delta_um": shot.recommended_grind_delta_um,
        "recommended_next_grind_steps": shot.recommended_next_grind_steps,
        "recommended_dose_g": shot.recommended_dose_g,
        "recommended_target_yield_g": shot.recommended_target_yield_g,
        "recommended_target_ratio": shot.recommended_target_ratio,
        "recommendation_decision": shot.recommendation_decision.value,
        "recommendation_followed": shot.recommendation_followed.value,
        "recommendation_attribution_weight": shot.recommendation_attribution_weight,
        "human_rating": shot.human_rating,
        "taste_tags_json": json.dumps(shot.taste_tags),
        "profile_score": shot.profile_score,
        "profile_mse": shot.profile_mse,
        "reward": shot.reward,
        "reward_confidence": shot.reward_confidence,
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
        profile=profile.copy(),
        raw_profile_available=bool(row["raw_profile_available"]),
        raw_profile_hash=row["raw_profile_hash"],
        grind_steps=row["grind_steps"],
        grind_um=row["grind_um"],
        grinder_step_size_um=row["grinder_step_size_um"],
        dose_in_g=row["dose_in_g"],
        beverage_out_g=row["beverage_out_g"],
        brew_ratio=row["brew_ratio"],
        target_yield_g=row["target_yield_g"],
        target_ratio=row["target_ratio"],
        shot_time_s=row["shot_time_s"],
        recommendation_id=row["recommendation_id"],
        recommended_grind_delta_steps=row["recommended_grind_delta_steps"],
        recommended_grind_delta_um=row["recommended_grind_delta_um"],
        recommended_next_grind_steps=row["recommended_next_grind_steps"],
        recommended_dose_g=row["recommended_dose_g"],
        recommended_target_yield_g=row["recommended_target_yield_g"],
        recommended_target_ratio=row["recommended_target_ratio"],
        recommendation_decision=RecommendationDecision(row["recommendation_decision"]),
        recommendation_followed=FollowThroughState(row["recommendation_followed"]),
        recommendation_attribution_weight=row["recommendation_attribution_weight"],
        human_rating=row["human_rating"],
        taste_tags=json.loads(row["taste_tags_json"]),
        profile_score=row["profile_score"],
        profile_mse=row["profile_mse"],
        reward=row["reward"],
        reward_confidence=row["reward_confidence"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _recommendation_to_row(recommendation: Recommendation) -> dict:
    return {
        "recommendation_id": recommendation.recommendation_id,
        "created_at": recommendation.created_at,
        "updated_at": recommendation.updated_at,
        "expires_at": recommendation.expires_at,
        "install_id": recommendation.install_id,
        "machine_id": recommendation.machine_id,
        "bean_context_id": recommendation.bean_context_id,
        "grind_delta_steps": recommendation.grind_delta_steps,
        "grind_delta_um": recommendation.grind_delta_um,
        "next_grind_steps": recommendation.next_grind_steps,
        "next_grind_um": recommendation.next_grind_um,
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
        grind_delta_steps=row["grind_delta_steps"],
        grind_delta_um=row["grind_delta_um"],
        next_grind_steps=row["next_grind_steps"],
        next_grind_um=row["next_grind_um"],
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
