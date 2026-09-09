"""Cross-repository wire test using the real C++ encoder/decoder.

Build the sibling firmware's scripts/test_artifact_recovery.py first.
Pure core tests do not require this optional firmware toolchain.
"""
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import test_gaggimate_adapter as fixtures
from espresso_rl.domain.cpbo import ComparisonMode, PendingPreferenceRequest, PreferenceAnchorSummary
from espresso_rl.domain.taste_goal import TasteGoal

BINARY = Path(__file__).resolve().parents[2] / "gaggiuino-gaggimate/.pio/host-artifact" / (
    "test.exe" if os.name == "nt" else "test"
)


@unittest.skipUnless(BINARY.exists(), "build the firmware host codec test first")
class FirmwareContractTests(unittest.TestCase):
    def test_confirmed_and_unknown_recipes_cross_the_wire_without_assumptions(self):
        from espresso_rl.adapters.sqlite_repositories import SQLiteStore, SQLiteShotRepository, SQLiteRecommendationRepository
        from espresso_rl.application.services import EspressoRLService
        from espresso_rl.application.cpbo_runtime import _known_recipe
        fixture = fixtures.GaggimateAdapterTests()
        fixture.setUp()
        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed), tempfile.TemporaryDirectory() as folder:
                command = "--export-confirmed-recipe" if confirmed else "--export-unknown-recipe"
                payload = json.loads(subprocess.check_output([str(BINARY), command], text=True, timeout=30))
                event = fixture.client.translate_shot_payload(payload, "AA_BB")
                with SQLiteStore(Path(folder) / "recipe.db") as store:
                    shots = SQLiteShotRepository(store)
                    result = EspressoRLService(shots, SQLiteRecommendationRepository(store), clock=lambda: 1720000100).ingest_shot_profile(event)
                    self.assertTrue(result.stored)
                    shot = shots.get(event.shot_id)
                    self.assertEqual(shot.grind_observed, confirmed)
                    self.assertEqual(shot.dose_target_confirmed, confirmed)
                    self.assertEqual(_known_recipe(shot) is not None, confirmed)
                    if confirmed:
                        self.assertEqual(shot.current_absolute_step, 14)
                        self.assertEqual(shot.relative_grind_steps_from_reference, 4)
                        self.assertEqual(shot.dose_target_g, 19)

    def test_cpp_shot_to_python_ingest_to_cpp_receipt(self):
        payload = json.loads(subprocess.check_output([str(BINARY), "--export-shot"], text=True, timeout=30))
        self.assertEqual(payload["delivery"], {"record_revision": 7, "reprocess": False})
        fixture = fixtures.GaggimateAdapterTests()
        fixture.setUp()
        mqtt = fixtures.FakeMQTT()
        fixture.client._client = mqtt
        ingested = []
        request = PendingPreferenceRequest(
            install_id=fixture.client._config.install_id, machine_id="gaggimate:AA_BB", optimization_run_id="run",
            new_shot_id=payload["shot_id"], anchor_shot_id="anchor",
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
            taste_goal=TasteGoal.custom({"sweet": "high"}),
            anchor=PreferenceAnchorSummary(timestamp=1720000000,
                relative_grind_steps_from_reference=5, dose_g=18, target_yield_g=36,
                beverage_out_g=35.5, profile_label="Morning espresso"),
        )

        def ingest(event):
            ingested.append(event)
            return SimpleNamespace(shot=object(), replayed=False, dropped_reason=None, preference_request=request)

        fixture.client._on_shot = ingest
        fixture.client._handle_shot_message(payload, "AA_BB")
        self.assertEqual(len(ingested), 1)
        receipt = mqtt.published[-1][1]
        self.assertEqual(json.loads(receipt)["outcome"], "accepted")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            path.write_text(receipt, encoding="utf-8")
            result = subprocess.check_output([str(BINARY), "--verify-receipt", str(path)], text=True, timeout=30)
            self.assertIn("PASS", result)
            # Firmware measures UTF-8 bytes; Python validates character counts.
            request = replace(request, anchor=replace(request.anchor, profile_label="\u00e9" * 90 + " espresso"))
            fixture.client._handle_shot_message(payload, "AA_BB")
            path.write_text(mqtt.published[-1][1], encoding="utf-8")
            result = subprocess.check_output([str(BINARY), "--verify-receipt", str(path)], text=True, timeout=30)
            self.assertIn("PASS", result)
