from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.sqlite_repositories import (
    SQLitePreferentialOptimizationRepository,
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
)
from espresso_rl.application.cpbo_runtime import CPBORuntimeBridge, strict_context_from_shot
from espresso_rl.application.preference_optimization import ConsecutivePreferenceOptimizationService
from espresso_rl.application.runtime_coordinator import (
    AutoTuningRuntimeCoordinator,
    PostShotOptimizationResult,
)
from espresso_rl.application.services import EspressoRLService
from espresso_rl.config import Config
from espresso_rl.domain.cpbo import (
    AcquisitionDiagnostics,
    ComparisonMode,
    ModelRecommendation,
    RecipeParameter,
    RecipePoint,
    RecipeSpace,
    Suggestion,
    SuggestionComputation,
    TrustRegionDiagnostics,
)
from espresso_rl.domain.models import Recipe, Recommendation, RecommendationStatus
from espresso_rl.optimizers.cpbo_config import TrustRegionConfig
from espresso_rl.optimizers.cpbo_trust_region import update_trust_region


FIXTURE = Path(__file__).parent / "fixtures" / "gaggimate_shot_profile.json"


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> None:
        self.published.append((topic, payload, qos, retain))


class FakeMessage:
    def __init__(self, topic: str, payload: dict) -> None:
        self.topic = topic
        self.payload = json.dumps(payload).encode("utf-8")


class DeterministicEngine:
    def __init__(self) -> None:
        self._grinds = [3.0, 4.0]

    def suggest(self, *, run, recipes, shots, comparisons, state, now):
        anchor = (
            state.previous_valid_shot_id
            if run.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS
            else state.incumbent_shot_id
        )
        recipe = RecipePoint.create(
            run.run_id,
            run.recipe_space,
            self._grinds.pop(0),
            18.0,
            38.0,
            created_at=now,
        )
        suggestion = Suggestion(
            suggestion_id=f"suggestion_{state.iteration + 1}",
            optimization_run_id=run.run_id,
            recipe=recipe,
            anchor_shot_id=anchor,
            comparison_mode=run.comparison_mode,
            acquisition=AcquisitionDiagnostics(
                acquisition_value=0.2,
                unclipped_acquisition_value=0.2,
                outcome_probabilities={
                    "new_better": 0.45,
                    "tie": 0.1,
                    "anchor_better": 0.45,
                },
                learned_gamma=0.2,
                kernel_weights={"raw": 0.8, "physics": 0.2, "trace": 0.0},
                raw_kernel_lengthscales=(1.0, 1.0, 1.0),
                physics_kernel_lengthscales=(1.0,),
                trace_kernel_enabled=False,
                fit_warnings=(),
                maximum_strategy="paper_gumbel",
                truncation_fallback_count=0,
            ),
            trust_region=TrustRegionDiagnostics(
                length=state.trust_region_state.length,
                lower_bounds=(0.0, 0.0, 0.0),
                upper_bounds=(1.0, 1.0, 1.0),
                success_count=state.trust_region_state.success_count,
                failure_count=state.trust_region_state.failure_count,
                restart_pending=state.trust_region_state.restart_pending,
                full_domain_proposal=False,
            ),
            model_version="test_cpbo",
            iteration=state.iteration + 1,
            created_at=now,
        )
        return SuggestionComputation(suggestion, '{"model":"safe"}', None)

    def recommend_evaluated(self, *, run, recipes, shots, comparisons, state):
        shot_id = state.incumbent_shot_id or state.previous_valid_shot_id
        shot = next(row for row in shots if row.shot_id == shot_id)
        recipe = next(row for row in recipes if row.recipe_id == shot.recipe_id)
        return ModelRecommendation(run.run_id, recipe, "test", True, shot_id)

    def update_trust_region_state(self, state, label, *, candidate_center):
        return update_trust_region(
            state,
            label,
            candidate_center=candidate_center,
            config=TrustRegionConfig(),
        )


class GaggimateEndToEndTests(unittest.TestCase):
    def test_two_shots_and_preference_produce_next_cpbo_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                clock_value = [1_700_000_100]

                def clock() -> int:
                    clock_value[0] += 1
                    return clock_value[0]

                shots = SQLiteShotRepository(store)
                recommendations = SQLiteRecommendationRepository(store)
                uploads = SQLiteUploadQueueRepository(store)
                service = EspressoRLService(
                    shots,
                    recommendations,
                    upload_queue=uploads,
                    clock=clock,
                )
                cpbo_repository = SQLitePreferentialOptimizationRepository(store)
                optimizer = ConsecutivePreferenceOptimizationService(
                    cpbo_repository,
                    DeterministicEngine(),
                    _recipe_space,
                    random_seed=7,
                    trace_feature_extractor=lambda sequence: (("duration",), (1.0,)),
                    clock=clock,
                )
                bridge = CPBORuntimeBridge(
                    optimizer,
                    shots,
                    service.persist_generated_recommendation,
                    strict_context_from_shot,
                    service.enqueue_comparison_upload,
                    comparison_mode=ComparisonMode.BEST_INCUMBENT,
                )
                transport = FakeTransport()

                class Publisher:
                    def publish_recommendation(self, recommendation: Recommendation) -> None:
                        client.publish_recommendation(recommendation)

                    def clear_recommendation(self, machine_id: str) -> None:
                        client.clear_recommendation(machine_id)

                    def publish_status(self, machine_id, bean_context_id, grinder_context_id, **kwargs) -> None:
                        client.publish_status(machine_id, {"optimizer_mode": kwargs.get("mode")})

                def optimize_after_shot(shot) -> PostShotOptimizationResult:
                    outcome = bridge.handle_shot(shot)
                    return PostShotOptimizationResult(
                        recommendation=outcome.recommendation,
                        preference_request=outcome.preference_request,
                    )

                coordinator = AutoTuningRuntimeCoordinator(
                    service,
                    Publisher(),
                    post_shot_recommendation=optimize_after_shot,
                )

                def on_preference(event) -> None:
                    client.publish_recommendation(bridge.handle_preference(event))

                client = GaggimateMQTTClient(
                    Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1"),
                    on_shot=coordinator.handle_shot,
                    on_correction=lambda event: None,
                    on_upload_maintenance=lambda event: None,
                    on_decision=lambda event: None,
                    on_apply=lambda event: None,
                    on_machine_state=coordinator.handle_machine_state,
                    on_preference=on_preference,
                )
                client._client = transport  # type: ignore[assignment]

                candidate = _shot_payload(
                    "shot_2",
                    grind=3.0,
                    community_upload_enabled=True,
                )
                candidate.pop("dose_in_g")
                candidate["dose_observed"] = False
                candidate["dose_target_confirmed"] = True
                historical_candidate = service.ingest_shot_profile(
                    client.translate_shot_payload(candidate, "AA_BB")
                )
                self.assertTrue(historical_candidate.stored)

                baseline = _shot_payload("shot_1", grind=2.0, community_upload_enabled=True)
                baseline.pop("dose_in_g")
                baseline["dose_observed"] = False
                baseline["dose_target_confirmed"] = True
                _send(client, transport, "gaggimate/AA_BB/shot/profile", baseline)
                baseline_ack = json.loads(transport.published[-1][1])
                self.assertEqual(baseline_ack["outcome"], "accepted")
                first = recommendations.get_current(
                    "install_1", "gaggimate:AA_BB", "bean_1", clock(),
                    grinder_context_id="grinder_1", profile_id="profile_1",
                )
                self.assertIsNotNone(first)
                _send(client, transport, "gaggimate/AA_BB/shot/profile", baseline)
                replay_ack = json.loads(transport.published[-1][1])
                self.assertEqual(replay_ack["outcome"], "already_processed")
                self.assertEqual(
                    len(cpbo_repository.list_shots(first.optimization_run_id)),  # type: ignore[union-attr]
                    1,
                )

                _send(client, transport, "gaggimate/AA_BB/shot/profile", candidate)
                candidate_ack = json.loads(transport.published[-1][1])
                self.assertEqual(candidate_ack["outcome"], "already_processed")
                self.assertEqual(
                    candidate_ack["preference_request"]["new_shot_id"],
                    "shot_2",
                )
                self.assertEqual(
                    candidate_ack["preference_request"]["anchor_shot_id"],
                    "shot_1",
                )
                self.assertEqual(
                    recommendations.get(first.recommendation_id).status,  # type: ignore[union-attr]
                    RecommendationStatus.SUPERSEDED,
                )
                self.assertTrue(
                    any(
                        topic.endswith("/rl/recommendation") and payload == "" and retained
                        for topic, payload, _, retained in transport.published
                    )
                )
                recommendation_publications_before_idle = sum(
                    topic.endswith("/rl/recommendation") and bool(payload)
                    for topic, payload, _, _ in transport.published
                )
                _send(
                    client,
                    transport,
                    "gaggimate/AA_BB/machine/state",
                    {
                        "event_type": "machine_state",
                        "schema_version": 1,
                        "install_id": "install_1",
                        "machine_id": "gaggimate:AA_BB",
                        "timestamp": clock(),
                        "state": "idle",
                        "bean_context_id": "bean_1",
                        "grinder_context_id": "grinder_1",
                        "profile_id": "profile_1",
                        "taste_goal": {
                            "schema_version": 1,
                            "mode": "balanced",
                            "targets": {},
                        },
                    },
                )
                recommendation_publications_after_idle = sum(
                    topic.endswith("/rl/recommendation") and bool(payload)
                    for topic, payload, _, _ in transport.published
                )
                self.assertEqual(
                    recommendation_publications_after_idle,
                    recommendation_publications_before_idle,
                )
                clears_before_pending_replay = sum(
                    topic.endswith("/rl/recommendation") and payload == ""
                    for topic, payload, _, _ in transport.published
                )
                _send(client, transport, "gaggimate/AA_BB/shot/profile", candidate)
                pending_replay_ack = json.loads(transport.published[-1][1])
                self.assertEqual(pending_replay_ack["outcome"], "already_processed")
                self.assertEqual(
                    pending_replay_ack["preference_request"],
                    candidate_ack["preference_request"],
                )
                clears_after_pending_replay = sum(
                    topic.endswith("/rl/recommendation") and payload == ""
                    for topic, payload, _, _ in transport.published
                )
                self.assertGreater(
                    clears_after_pending_replay,
                    clears_before_pending_replay,
                )
                _send(
                    client,
                    transport,
                    "gaggimate/AA_BB/rl/preference",
                    {
                        "event_type": "preference_feedback",
                        "schema_version": 1,
                        "optimization_run_id": first.optimization_run_id,  # type: ignore[union-attr]
                        "new_shot_id": "shot_2",
                        "anchor_shot_id": "shot_1",
                        "label": "new_better",
                        "comparison_mode": "best_incumbent",
                        "taste_goal": {"schema_version": 1, "mode": "balanced", "targets": {}},
                        "install_id": "install_1",
                        "machine_id": "gaggimate:AA_BB",
                        "timestamp": clock(),
                        "source": "webui",
                    },
                )

                comparisons = cpbo_repository.list_comparisons(first.optimization_run_id)  # type: ignore[union-attr]
                self.assertEqual(len(comparisons), 1)
                self.assertEqual(comparisons[0].label.value, "new_better")
                from espresso_rl.domain.models import UploadQueueStatus

                self.assertEqual(uploads.count_by_status().get(UploadQueueStatus.PENDING), 5)
                recommendation_payloads = [
                    json.loads(payload)
                    for topic, payload, _, _ in transport.published
                    if topic.endswith("/rl/recommendation") and payload
                ]
                self.assertEqual(len(recommendation_payloads), 2)
                self.assertEqual(recommendation_payloads[-1]["comparison_anchor_shot_id"], "shot_2")
                self.assertFalse(any("human_rating" in payload for payload in recommendation_payloads))

                clears_before_replay = sum(
                    topic.endswith("/rl/recommendation") and payload == ""
                    for topic, payload, _, _ in transport.published
                )
                _send(client, transport, "gaggimate/AA_BB/shot/profile", candidate)
                clears_after_replay = sum(
                    topic.endswith("/rl/recommendation") and payload == ""
                    for topic, payload, _, _ in transport.published
                )
                self.assertEqual(clears_after_replay, clears_before_replay)


def _recipe_space(recipe: Recipe, _recipe_domain: object) -> RecipeSpace:
    return RecipeSpace(
        RecipeParameter("grind_size", 0.0, 10.0, 1.0, "step"),
        RecipeParameter("dose_g", 14.0, 22.0, 0.1, "g"),
        RecipeParameter("target_output_g", 20.0, 60.0, 0.1, "g"),
        recipe.grinder_step_direction,
    )


def _shot_payload(shot_id: str, *, grind: float, **overrides: object) -> dict:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload.update(
        {
            "shot_id": shot_id,
            "install_id": "install_1",
            "machine_id": "gaggimate:AA_BB",
            "bean_context_id": "bean_1",
            "grinder_context_id": "grinder_1",
            "profile_id": "profile_1",
            "relative_grind_steps_from_reference": grind,
            **overrides,
        }
    )
    return payload


def _send(client: GaggimateMQTTClient, transport: FakeTransport, topic: str, payload: dict) -> None:
    client._on_message(transport, None, FakeMessage(topic, payload))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
