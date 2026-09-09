import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import test_cpbo_runtime_bridge as runtime
import test_gaggimate_adapter as mqtt
from espresso_rl.adapters.delivery_receipts import DatabaseDeliveryReceipts
from espresso_rl.adapters.sqlite_repositories import SQLiteStore, SQLiteUploadQueueRepository
from espresso_rl.application.community_handoff import CommunityHandoffService
from espresso_rl.application.cpbo_runtime import _physical_status
from espresso_rl.application.upload_payloads import shot_upload_payload
from espresso_rl.domain.cpbo import PhysicalShotStatus
from espresso_rl.domain.models import UploadQueueStatus


class DeliveryReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteStore(Path(self.directory.name) / "test.db")
        self.addCleanup(self.store.close)
        self.fixture = mqtt.GaggimateAdapterTests()
        self.fixture.setUp()
        self.client = self.fixture.client
        self.broker = mqtt.FakeMQTT()
        self.client._client = self.broker
        self.client._delivery_receipts = DatabaseDeliveryReceipts(self.store)
        self.topic = "gaggimate/AA_BB/rl/preference"

    def deliver(self, payload):
        raw = json.dumps(payload)
        self.client._on_message(self.broker, None, SimpleNamespace(topic=self.topic, payload=raw.encode()))
        return hashlib.sha256((self.topic + "\n" + raw).encode()).hexdigest()

    def preference(self):
        return dict(event_type="preference_feedback", schema_version=1, optimization_run_id="run",
                    new_shot_id="new", anchor_shot_id="anchor", label="abstain", install_id="install",
                    machine_id="gaggimate:AA_BB", timestamp=200, comparison_mode="best_incumbent",
                    taste_goal={"schema_version": 1, "mode": "balanced", "targets": {}}, source="webui")

    def test_offline_retry_then_durable_receipt_and_restart_replay(self):
        handle = Mock(side_effect=RuntimeError("model unavailable"))
        self.client._on_preference = handle
        identity = self.deliver(self.preference())
        self.assertEqual(self.broker.published, [])
        self.assertIsNone(self.client._delivery_receipts.get(identity))
        handle.side_effect = None
        self.deliver(self.preference())
        receipt = json.loads(self.broker.published[-1][1])
        self.assertEqual(receipt["delivery_id"], identity)
        self.assertEqual(receipt["outcome"], "accepted")
        self.client._delivery_receipts = DatabaseDeliveryReceipts(self.store)
        self.deliver(self.preference())
        self.assertEqual(handle.call_count, 2)
        self.assertEqual(len(self.broker.published), 2)

    def test_invalid_label_is_rejected_without_calling_application(self):
        self.client._on_preference = Mock()
        identity = self.deliver({**self.preference(), "label": "guess"})
        self.client._on_preference.assert_not_called()
        self.assertEqual(self.client._delivery_receipts.get(identity), "permanent_rejection")
        self.assertEqual(json.loads(self.broker.published[-1][1])["outcome"], "permanent_rejection")

    def test_receipt_journal_failure_does_not_acknowledge(self):
        self.client._on_preference = Mock()
        self.client._delivery_receipts = Mock(get=Mock(return_value=None), record=Mock(side_effect=OSError("disk full")))
        self.deliver(self.preference())
        self.assertEqual(self.broker.published, [])

    def test_cloud_backlog_is_only_acknowledged_after_container_queue_commit(self):
        queue = SQLiteUploadQueueRepository(self.store)
        service = CommunityHandoffService(queue, enabled=True, install_id="container-install", clock=lambda: 200)
        self.client._on_community_handoff = service.accept
        self.topic = "gaggimate/AA_BB/rl/community/handoff"
        payload = shot_upload_payload(runtime._shot("old-shot", grind=5))
        payload.update(machine_id="gaggimate:AA_BB", timestamp=1720000000, profile_temperature_c=93, final_phase_temperature_c=93)
        identity = self.deliver(payload)
        self.assertEqual(self.client._delivery_receipts.get(identity), "accepted")
        rows = queue.list_by_status(UploadQueueStatus.PENDING)
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0].payload_json)["install_id"], "container-install")
        self.deliver(payload)
        self.assertEqual(len(queue.list_by_status(UploadQueueStatus.PENDING)), 1)

    def test_legacy_handoff_cannot_replace_a_container_record(self):
        from espresso_rl.application.upload_payloads import make_upload_item
        queue = SQLiteUploadQueueRepository(self.store)
        payload = shot_upload_payload(runtime._shot("old-shot", grind=5))
        payload.update(machine_id="gaggimate:AA_BB", timestamp=1720000000,
                       profile_temperature_c=93, final_phase_temperature_c=93)
        current = {**payload, "dose_g": 19}
        queue.enqueue(make_upload_item("shot", "old-shot", current, 201))
        CommunityHandoffService(queue, enabled=True, install_id="container", clock=lambda: 202).accept(payload)
        self.assertEqual(json.loads(queue.list_ready(203)[0].payload_json), current)

    def test_local_correction_wins_over_device_backlog_even_without_prior_upload(self):
        queue = SQLiteUploadQueueRepository(self.store)
        current = runtime._shot("old-shot", grind=8)
        payload = shot_upload_payload(runtime._shot("old-shot", grind=5))
        payload.update(machine_id="gaggimate:AA_BB", timestamp=1720000000,
                       profile_temperature_c=93, final_phase_temperature_c=93)
        CommunityHandoffService(queue, enabled=True, install_id="container", clock=lambda: 202,
                                shots=Mock(get=Mock(return_value=current))).accept(payload)
        self.assertEqual(json.loads(queue.list_ready(203)[0].payload_json), shot_upload_payload(current))

    def test_queue_failure_retains_saved_shot_for_retry(self):
        import test_application_service as application
        from espresso_rl.application.services import EspressoRLService
        from espresso_rl.adapters.sqlite_repositories import SQLiteShotRepository, SQLiteRecommendationRepository
        shots = SQLiteShotRepository(self.store)
        queue = SQLiteUploadQueueRepository(self.store)
        failing_queue = Mock(wraps=queue)
        failing_queue.enqueue.side_effect = OSError("disk full")
        service = EspressoRLService(shots, SQLiteRecommendationRepository(self.store),
                                   upload_queue=failing_queue, clock=lambda: 200)
        event = application._shot_event(community_upload_enabled=True)
        with self.assertRaises(OSError):
            service.ingest_shot_profile(event)
        self.assertIsNotNone(shots.get(event.shot_id))
        self.assertEqual(queue.list_ready(200), [])
        failing_queue.enqueue.side_effect = None
        result = service.ingest_shot_profile(event)
        self.assertTrue(result.replayed)
        self.assertEqual(len(queue.list_ready(200)), 1)

    def test_persisted_preference_recovers_after_model_failure_and_restart(self):
        fixture = runtime.CPBORuntimeBridgeTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        bridge = fixture.bridge([6, 7])
        baseline = runtime._shot("baseline", grind=5)
        candidate = runtime._shot("new", grind=6)
        fixture.shots.rows.update(baseline=baseline, new=candidate)
        first = bridge.handle_shot(baseline).recommendation
        bridge.handle_shot(candidate)
        bridge._optimizer.suggest_next = Mock(side_effect=ValueError("model fit failed"))
        self.client._on_preference = bridge.handle_preference
        payload = {**self.preference(), "label": "new_better", "anchor_shot_id": "baseline",
                   "optimization_run_id": first.optimization_run_id}
        identity = self.deliver(payload)
        self.assertIsNone(self.client._delivery_receipts.get(identity))
        self.assertEqual(len(fixture.repository.list_comparisons(first.optimization_run_id)), 1)
        # Reconstruct the application service against persisted optimizer state.
        recovered = fixture.bridge([7])
        self.client._on_preference = recovered.handle_preference
        self.deliver(payload)
        self.assertEqual(self.client._delivery_receipts.get(identity), "accepted")
        self.assertEqual(len(fixture.repository.list_comparisons(first.optimization_run_id)), 1)
        self.assertEqual(fixture.recommendations[-1].projected_relative_step_from_reference, 7)

    def test_disabled_container_keeps_handoff_unacknowledged(self):
        self.client._on_community_handoff = CommunityHandoffService(
            SQLiteUploadQueueRepository(self.store), enabled=False, install_id="container", clock=lambda: 200).accept
        self.topic = "gaggimate/AA_BB/rl/community/handoff"
        payload = shot_upload_payload(runtime._shot("old-shot", grind=5))
        payload.update(machine_id="gaggimate:AA_BB", timestamp=1720000000, profile_temperature_c=93, final_phase_temperature_c=93)
        self.deliver(payload)
        self.assertEqual(self.broker.published, [])
