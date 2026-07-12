from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
    _shot_to_row,
)
from espresso_rl.config import Config
from espresso_rl.domain.models import (
    Recommendation,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
    UploadQueueItem,
    UploadQueueStatus,
)
from espresso_rl.main import upload_queue_for_service
from espresso_rl.domain.taste_goal import TasteGoal


class SQLiteAndBoundaryTests(unittest.TestCase):
    def test_sqlite_round_trips_scalar_free_shot_and_cpbo_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recommendations = SQLiteRecommendationRepository(store)
                shots.upsert(_shot())
                recommendations.upsert(_recommendation())

                stored_shot = shots.get("shot_1")
                stored_recommendation = recommendations.get("rec_1")
                self.assertEqual(stored_shot.relative_grind_steps_from_reference, 2.0)  # type: ignore[union-attr]
                self.assertEqual(stored_recommendation.mode, RecommendationMode.CPBO_BEST_INCUMBENT)  # type: ignore[union-attr]
                self.assertEqual(stored_shot.taste_goal, _taste_goal())  # type: ignore[union-attr]
                self.assertEqual(stored_recommendation.taste_goal, _taste_goal())  # type: ignore[union-attr]
                self.assertIsNone(
                    recommendations.get_current(
                        "install_1",
                        "machine_1",
                        "bean_1",
                        150,
                        grinder_context_id="grinder_1",
                        profile_id="profile_1",
                        taste_goal_fingerprint=TasteGoal.balanced().fingerprint,
                    )
                )
                columns = {
                    row["name"]
                    for row in store.conn.execute("PRAGMA table_info(shots)").fetchall()
                }
                self.assertFalse(
                    columns.intersection(
                        {
                            "human_rating",
                            "taste_tags_json",
                            "feedback_recorded",
                            "reward",
                            "reward_confidence",
                            "profile_score",
                            "profile_mse",
                        }
                    )
                )

    def test_shared_shot_row_uses_driver_neutral_booleans(self) -> None:
        row = _shot_to_row(_shot())
        self.assertIs(type(row["raw_profile_available"]), bool)
        self.assertIs(type(row["grind_observed"]), bool)
        self.assertNotIn("human_rating", row)
        self.assertNotIn("reward", row)

    def test_upload_queue_tracks_retry_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(_upload())
                ready = queue.list_ready(now=10, limit=10)
                self.assertEqual([item.upload_id for item in ready], ["upload_1"])
                queue.update_status(
                    "upload_1",
                    UploadQueueStatus.FAILED,
                    now=10,
                    next_retry_at=30,
                    error_message="offline",
                )
                self.assertEqual(queue.count_by_status(), {UploadQueueStatus.FAILED: 1})
                self.assertEqual(queue.list_ready(now=20, limit=10), [])
                self.assertEqual(len(queue.list_ready(now=30, limit=10)), 1)

    def test_admin_role_never_pushes_to_public_upload_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                config = Config(
                    mqtt_host="localhost",
                    deployment_role="admin",
                    storage_backend="postgres",
                    postgres_dsn="postgresql://example.invalid/db",
                    data_dir=Path(tmp),
                )
                self.assertIsNone(upload_queue_for_service(config, queue))

    def test_postgres_schema_is_preference_only_and_algorithm_neutral(self) -> None:
        schema = (
            Path(__file__).parents[1]
            / "src"
            / "espresso_rl"
            / "adapters"
            / "postgres_schema.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("CREATE TABLE IF NOT EXISTS community_validated_shots", schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS community_comparisons", schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS cpbo_comparisons", schema)
        self.assertNotIn("human_rating", schema)
        self.assertNotIn("taste_tags_json", schema)
        self.assertNotIn("dreamer_shadow", schema)
        self.assertNotIn("training_dataset", schema)

    def test_core_layers_do_not_import_adapters_or_infrastructure(self) -> None:
        package = Path(__file__).parents[1] / "src" / "espresso_rl"
        forbidden_prefixes = (
            "espresso_rl.adapters",
            "paho",
            "psycopg",
            "fastapi",
            "supabase",
        )
        for layer in ("domain", "application", "optimizers", "ports"):
            for source in (package / layer).glob("*.py"):
                tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)
                if layer == "application":
                    forbidden = forbidden_prefixes
                else:
                    forbidden = forbidden_prefixes + ("espresso_rl.application",)
                self.assertFalse(
                    any(name.startswith(forbidden) for name in imports),
                    f"{source} imports infrastructure: {imports}",
                )

    def test_cpbo_math_does_not_consume_taste_goal_as_a_model_feature(self) -> None:
        optimizer_dir = Path(__file__).parents[1] / "src" / "espresso_rl" / "optimizers"
        references = {
            source.name
            for source in optimizer_dir.glob("cpbo*.py")
            if "taste_goal" in source.read_text(encoding="utf-8")
        }
        self.assertEqual(references, set())


def _shot() -> ShotRecord:
    return ShotRecord(
        shot_id="shot_1",
        timestamp=100,
        install_id="install_1",
        machine_id="machine_1",
        machine_adapter="gaggimate",
        bean_context_id="bean_1",
        grinder_context_id="grinder_1",
        profile_id="profile_1",
        taste_goal=_taste_goal(),
        profile=np.zeros((5, 100), dtype=np.float32),
        raw_profile_available=True,
        raw_profile_hash="a" * 64,
        microns_per_step=12.5,
        relative_grind_steps_from_reference=2.0,
        dose_in_g=18.0,
        target_yield_g=36.0,
        beverage_out_g=35.8,
        shot_time_s=30.0,
        created_at=100,
        updated_at=100,
    )


def _recommendation() -> Recommendation:
    return Recommendation(
        recommendation_id="rec_1",
        created_at=100,
        updated_at=100,
        expires_at=200,
        install_id="install_1",
        machine_id="machine_1",
        bean_context_id="bean_1",
        grinder_context_id="grinder_1",
        profile_id="profile_1",
        taste_goal=_taste_goal(),
        grind_delta_steps_from_current=1.0,
        grind_delta_um_from_current=12.5,
        projected_relative_step_from_reference=3.0,
        projected_relative_grind_um_from_reference=37.5,
        next_dose_g=18.0,
        target_yield_g=36.0,
        target_ratio=2.0,
        mode=RecommendationMode.CPBO_BEST_INCUMBENT,
        confidence=0.5,
        reason="test",
        status=RecommendationStatus.PENDING,
    )


def _upload() -> UploadQueueItem:
    return UploadQueueItem(
        upload_id="upload_1",
        local_record_type="shot",
        local_record_id="shot_1",
        payload_hash="a" * 64,
        payload_json='{"event_type":"shot_record"}',
        status=UploadQueueStatus.PENDING,
        created_at=1,
        updated_at=1,
    )


def _taste_goal() -> TasteGoal:
    return TasteGoal.custom({"sweet": "high", "bitter": "low"})


if __name__ == "__main__":
    unittest.main()
