from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.local_model_store import LocalModelArtifactStore
from espresso_rl.application.checkpoint_loading import CheckpointLoadError, load_verified_dreamer_checkpoint
from espresso_rl.domain.model_checkpoint import DreamerCheckpointCompatibility
from espresso_rl.domain.optimization import DEFAULT_OPTIMIZER_MODE, OPTIMIZER_MODE_DREAMER_V3_SHADOW
from espresso_rl.optimizers.runtime import RuntimeOptimizer


class MemoryModelArtifactStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        payload = self.payloads[reference]
        if len(payload) > max_bytes:
            raise ValueError("artifact exceeds limit")
        return payload


class CheckpointLoadingTests(unittest.TestCase):
    def test_loads_authenticated_checkpoint_as_non_executable_typed_data(self) -> None:
        bundle = checkpoint_bundle()

        checkpoint = load_bundle(bundle)

        self.assertFalse(checkpoint.inference_ready)
        self.assertEqual(checkpoint.component_names, ("actor", "critic", "world_model"))
        self.assertEqual(checkpoint.tensor("actor.weight").shape, (1,))
        self.assertEqual(bytes(checkpoint.tensor_bytes("actor.weight")), struct.pack("<f", 1.0))
        self.assertEqual(checkpoint.artifact_sha256, bundle["artifact_sha256"])
        self.assertEqual(checkpoint.evaluation_report_sha256, "7" * 64)

    def test_rejects_configured_artifact_hash_mismatch(self) -> None:
        bundle = checkpoint_bundle()

        with self.assertRaisesRegex(CheckpointLoadError, "configured digest"):
            load_bundle(bundle, expected_artifact_sha256="0" * 64)

    def test_rejects_tensor_payload_tampering_even_with_updated_outer_hash(self) -> None:
        bundle = checkpoint_bundle()
        artifact = bytearray(bundle["artifact"])
        artifact[-1] ^= 1
        replace_artifact(bundle, bytes(artifact))

        with self.assertRaisesRegex(CheckpointLoadError, "tensor .* SHA-256"):
            load_bundle(bundle)

    def test_rejects_missing_checkpoint_tensor(self) -> None:
        bundle = checkpoint_bundle()
        header, data = split_safetensors(bundle["artifact"])
        del header["actor.weight"]
        replace_artifact(bundle, encode_safetensors(header, data))

        with self.assertRaisesRegex(CheckpointLoadError, "tensor names"):
            load_bundle(bundle)

    def test_rejects_metadata_that_disagrees_with_manifest(self) -> None:
        bundle = checkpoint_bundle()
        header, data = split_safetensors(bundle["artifact"])
        header["__metadata__"]["control_spec_sha256"] = "9" * 64
        replace_artifact(bundle, encode_safetensors(header, data))

        with self.assertRaisesRegex(CheckpointLoadError, "metadata control_spec_sha256"):
            load_bundle(bundle)

    def test_rejects_pickle_format_before_tensor_loading(self) -> None:
        bundle = checkpoint_bundle()
        bundle["manifest"]["model_artifact"]["format"] = "torch_pickle"
        refresh_manifest(bundle)

        with self.assertRaisesRegex(CheckpointLoadError, "safetensors"):
            load_bundle(bundle)

    def test_rejects_incompatible_checkpoint_schema(self) -> None:
        bundle = checkpoint_bundle()
        bundle["manifest"]["model_artifact"]["checkpoint_schema_version"] = 99
        refresh_manifest(bundle)

        with self.assertRaisesRegex(CheckpointLoadError, "schema_version"):
            load_bundle(bundle)

    def test_rejects_runtime_feature_layout_mismatch(self) -> None:
        bundle = checkpoint_bundle()

        with self.assertRaisesRegex(CheckpointLoadError, "feature layout"):
            load_bundle(
                bundle,
                compatibility=DreamerCheckpointCompatibility(feature_layout_sha256="9" * 64),
            )

    def test_rejects_manifest_duplicate_fields(self) -> None:
        bundle = checkpoint_bundle()
        manifest_text = bundle["manifest_payload"].decode("utf-8")
        duplicate = manifest_text.replace(
            '"format":"espresso_rl_model_manifest_v1"',
            '"format":"espresso_rl_model_manifest_v1","format":"espresso_rl_model_manifest_v1"',
            1,
        ).encode("utf-8")
        store = MemoryModelArtifactStore({"model": bundle["artifact"], "manifest": duplicate})

        with self.assertRaisesRegex(CheckpointLoadError, "duplicate field"):
            load_verified_dreamer_checkpoint(
                store,
                artifact_reference="model",
                manifest_reference="manifest",
                expected_artifact_sha256=bundle["artifact_sha256"],
            )

    def test_verified_preview_does_not_enable_dreamer_or_remove_bo_fallback(self) -> None:
        bundle = checkpoint_bundle()
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "dreamer_v3.safetensors"
            manifest_path = Path(temporary_directory) / "dreamer_v3_manifest.json"
            artifact_path.write_bytes(bundle["artifact"])
            manifest_path.write_bytes(bundle["manifest_payload"])
            checkpoint = load_verified_dreamer_checkpoint(
                LocalModelArtifactStore(),
                artifact_reference=str(artifact_path),
                manifest_reference=str(manifest_path),
                expected_artifact_sha256=bundle["artifact_sha256"],
            )
            optimizer = RuntimeOptimizer(
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW,
                model_artifact_path=str(artifact_path),
                model_artifact_sha256=bundle["artifact_sha256"],
                model_manifest_path=str(manifest_path),
                verified_checkpoint=checkpoint,
            )

        self.assertFalse(checkpoint.inference_ready)
        self.assertEqual(optimizer.status().effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertFalse(optimizer.status().dreamer_v3_available)
        self.assertTrue(optimizer.status().checkpoint_verified)
        self.assertFalse(optimizer.status().checkpoint_inference_ready)
        self.assertIn("not enabled", optimizer.status().checkpoint_unavailable_reason or "")

    def test_local_store_enforces_size_limit_and_reads_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.safetensors"
            path.write_bytes(b"safe tensor bytes")
            store = LocalModelArtifactStore()

            self.assertEqual(store.read_bytes(str(path), max_bytes=100), b"safe tensor bytes")
            with self.assertRaisesRegex(ValueError, "size limit"):
                store.read_bytes(str(path), max_bytes=2)


def load_bundle(
    bundle: dict,
    *,
    expected_artifact_sha256: str | None = None,
    compatibility: DreamerCheckpointCompatibility | None = None,
):
    store = MemoryModelArtifactStore(
        {
            "model": bundle["artifact"],
            "manifest": bundle["manifest_payload"],
        }
    )
    return load_verified_dreamer_checkpoint(
        store,
        artifact_reference="model",
        manifest_reference="manifest",
        expected_artifact_sha256=expected_artifact_sha256 or bundle["artifact_sha256"],
        compatibility=compatibility,
    )


def checkpoint_bundle() -> dict:
    tensor_payloads = {
        "actor.weight": struct.pack("<f", 1.0),
        "critic.bias": struct.pack("<f", 2.0),
        "world_model.scale": struct.pack("<f", 3.0),
    }
    tensor_entries = {}
    components = {}
    header_entries = {}
    chunks = []
    offset = 0
    for name, payload in sorted(tensor_payloads.items()):
        component = name.split(".", 1)[0]
        tensor_entries[name] = {
            "component": component,
            "dtype": "F32",
            "shape": [1],
            "element_count": 1,
            "sha256": sha256(payload),
        }
        components[component] = {"tensor_count": 1, "element_count": 1}
        header_entries[name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + len(payload)],
        }
        chunks.append(payload)
        offset += len(payload)

    tensor_manifest = {
        "format": "espresso_rl_checkpoint_tensor_manifest_v1",
        "schema_version": 1,
        "tensor_count": 3,
        "component_count": 3,
        "component_names": ["actor", "critic", "world_model"],
        "components": components,
        "tensors": tensor_entries,
    }
    tensor_manifest_sha256 = sha256_json(tensor_manifest)
    metadata = {
        "format": "espresso_rl_dreamer_v3_checkpoint_safetensors_v1",
        "schema_version": "1",
        "model_family": "dreamer_v3",
        "artifact_stage": "world_model_train_preview",
        "inference_ready": "false",
        "dataset_sha256": "1" * 64,
        "training_config_sha256": "3" * 64,
        "dreamer_tensor_contract_sha256": "4" * 64,
        "feature_layout_sha256": "5" * 64,
        "control_spec_sha256": "6" * 64,
        "tensor_manifest_sha256": tensor_manifest_sha256,
        "evaluation_report_sha256": "7" * 64,
        "world_model_smoke_sha256": "",
        "world_model_train_preview_sha256": "8" * 64,
        "row_count": "4",
        "created_at": "1800000000",
    }
    artifact = encode_safetensors({"__metadata__": metadata, **header_entries}, b"".join(chunks))
    artifact_sha256 = sha256(artifact)
    manifest = {
        "format": "espresso_rl_model_manifest_v1",
        "schema_version": 1,
        "model_family": "dreamer_v3",
        "model_artifact": {
            "format": "safetensors",
            "sha256": artifact_sha256,
            "checkpoint_format": "espresso_rl_dreamer_v3_checkpoint_safetensors_v1",
            "checkpoint_schema_version": 1,
            "tensor_manifest_sha256": tensor_manifest_sha256,
            "evaluation_report_sha256": "7" * 64,
            "tensor_manifest": tensor_manifest,
            "tensor_count": 3,
            "component_count": 3,
            "component_names": ["actor", "critic", "world_model"],
            "dreamer_tensor_contract_sha256": "4" * 64,
            "feature_layout_sha256": "5" * 64,
            "control_spec_sha256": "6" * 64,
        },
        "dataset": {
            "format": "espresso_rl_training_dataset_v1",
            "sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
        },
        "trainer": {
            "git_sha": "trainerabc",
            "training_config_sha256": "3" * 64,
            "artifact_stage": "world_model_train_preview",
        },
        "schemas": {
            "state_schema_version": 1,
            "action_schema_version": 1,
            "reward_schema_version": 1,
        },
        "runtime_compatibility": {
            "optimizer_mode": "dreamer_v3_shadow",
            "espresso_rl_runtime_schema_version": 1,
            "inference_ready": False,
        },
    }
    bundle = {
        "artifact": artifact,
        "artifact_sha256": artifact_sha256,
        "manifest": manifest,
    }
    refresh_manifest(bundle)
    return bundle


def replace_artifact(bundle: dict, artifact: bytes) -> None:
    bundle["artifact"] = artifact
    bundle["artifact_sha256"] = sha256(artifact)
    bundle["manifest"]["model_artifact"]["sha256"] = bundle["artifact_sha256"]
    refresh_manifest(bundle)


def refresh_manifest(bundle: dict) -> None:
    bundle["manifest_payload"] = canonical_json(bundle["manifest"]).encode("utf-8")


def split_safetensors(payload: bytes) -> tuple[dict, bytes]:
    header_length = int.from_bytes(payload[:8], "little")
    data_start = 8 + header_length
    return json.loads(payload[8:data_start]), payload[data_start:]


def encode_safetensors(header: dict, data: bytes) -> bytes:
    header_payload = canonical_json(header).encode("utf-8")
    return len(header_payload).to_bytes(8, "little") + header_payload + data


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value) -> str:
    return sha256(canonical_json(value).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
