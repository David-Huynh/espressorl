# Model Artifacts And Training Exports

EspressoRL can export validated community data and build auditable DreamerV3
artifact candidates. Runtime model use remains gated by strict validation and
release metadata.

## Runtime Model Configuration

Docker builds can embed a release-default model SHA with:

```text
ESPRESSORL_RELEASE_MODEL_ARTIFACT_SHA256
```

If that default SHA is present and no explicit path is configured, EspressoRL
looks for:

```text
/data/espresso_rl/models/dreamer_v3.safetensors
/data/espresso_rl/models/dreamer_v3_manifest.json
```

Custom model trainers can configure:

```json
{
  "optimizer_model_artifact_path": "/data/espresso_rl/models/dreamer_v3.safetensors",
  "optimizer_model_artifact_sha256": "MODEL_FILE_SHA256",
  "optimizer_model_manifest_path": "/data/espresso_rl/models/dreamer_v3_manifest.json",
  "optimizer_model_artifact_max_bytes": 536870912
}
```

The equivalent environment variables are:

```text
ESPRESSORL_OPTIMIZER_MODEL_ARTIFACT_PATH
ESPRESSORL_OPTIMIZER_MODEL_ARTIFACT_SHA256
ESPRESSORL_OPTIMIZER_MODEL_MANIFEST_PATH
```

Explicit config overrides the release default.

## Manifest Requirements

The model manifest is UTF-8 JSON. It must identify the model family, artifact
format, artifact SHA-256, dataset SHA-256, dataset manifest SHA-256, trainer git
SHA, training config SHA-256, schema versions, and runtime compatibility.

Runtime inference artifacts must use `safetensors`. Pickle-style files such as
full `torch.save(model)` outputs are intentionally rejected.

Minimal manifest shape:

```json
{
  "format": "espresso_rl_model_manifest_v1",
  "schema_version": 1,
  "model_family": "dreamer_v3",
  "model_artifact": {
    "format": "safetensors",
    "sha256": "MODEL_FILE_SHA256"
  },
  "dataset": {
    "format": "espresso_rl_training_dataset_v1",
    "sha256": "TRAINING_ROWS_JSONL_SHA256",
    "manifest_sha256": "TRAINING_EXPORT_MANIFEST_SHA256"
  },
  "trainer": {
    "git_sha": "TRAINER_REPO_COMMIT",
    "training_config_sha256": "TRAINING_CONFIG_JSON_SHA256"
  },
  "schemas": {
    "state_schema_version": 1,
    "action_schema_version": 1,
    "reward_schema_version": 1
  },
  "runtime_compatibility": {
    "optimizer_mode": "dreamer_v3_shadow",
    "espresso_rl_runtime_schema_version": 1,
    "inference_ready": false
  }
}
```

## Legacy Training Export Format

The old Dreamer community export is no longer exposed by the admin dashboard.
Its isolated implementation remains temporarily while Dreamer is removed. The
legacy format was intentionally plain:

- `training_rows.jsonl`: canonical UTF-8 JSON Lines, one
  `espresso_rl_training_transition_v1` object per line.
- `training_rows.csv`: spreadsheet-friendly review summary without profile
  arrays.
- `manifest.json`: row counts, file hashes, dataset hash, schema version,
  canonical row format, source git SHA, and zero-trust flags.
- `README.txt`: plain-language format notes.

No legacy export artifact uses pickle, SQLite dumps, parquet, macros, absolute
grinder settings, or executable/opaque formats. A future offline-model dataset
will use a new versioned artifact that joins immutable physical shots to
oriented comparisons instead of carrying this scalar reward schema forward.

## Dreamer Episode And Tensor Conversion

`espresso_rl.dreamer.dataset.load_dreamer_episodes_from_jsonl` converts
canonical rows into `espresso_rl_dreamer_episode_v4` episodes. Dreamer uses
fixed-cadence telemetry at 250 ms intervals, while the older fixed 5x100
profile remains a duration-normalized summary for BO, profile scoring, and
compatibility.

Dreamer episodes include:

- pressure and pressure target
- pump flow and flow target
- beverage mass flow
- weight
- temperature and temperature target
- pump target mode
- valve state
- static context such as relative grind, dose, planned yield, and taste
  objective

Unknown values carry masks instead of being fabricated. Unknown grind or dose
does not discard the rest of the trajectory.

## Artifact Builder

Create default training configs:

```bash
uv run espresso-rl-build-dreamer-artifacts \
  --write-default-config training_config.json
```

Build a release-candidate artifact:

```bash
uv run espresso-rl-build-dreamer-artifacts \
  --dataset-jsonl training_rows.jsonl \
  --dataset-manifest manifest.json \
  --training-config training_config.json \
  --output-dir model_out \
  --trainer-git-sha TRAINER_REPO_COMMIT \
  --artifact-stage world_model_release_candidate
```

The builder validates the dataset hash, revalidates every JSONL transition,
rejects absolute grinder fields, builds deterministic Dreamer tensors, writes
schema hashes into the audit report, and produces fixed safe filenames:

```text
dreamer_v3.safetensors
dreamer_v3_manifest.json
training_config.json
audit_report.json
checksums.txt
```

The generated checkpoint remains `inference_ready=false` until explicitly
released.

## Release Bundle

After reviewing candidate audit/evaluation outputs, create the runtime release
bundle:

```bash
uv run espresso-rl-release-dreamer-model \
  --candidate-artifact model_out/dreamer_v3.safetensors \
  --candidate-manifest model_out/dreamer_v3_manifest.json \
  --candidate-artifact-sha256 CANDIDATE_ARTIFACT_SHA256 \
  --candidate-manifest-sha256 CANDIDATE_MANIFEST_SHA256 \
  --released-by RELEASE_IDENTITY \
  --release-version RELEASE_TAG \
  --output-dir release_out
```

The release command accepts only release-candidate inputs, verifies exact
hashes, revalidates through the strict checkpoint loader, preserves every tensor
byte, and writes:

```text
dreamer_v3.safetensors
dreamer_v3_manifest.json
release_record.json
checksums.txt
```

Configure the released artifact SHA-256, not the candidate SHA-256, as the
runtime model SHA.

## Runtime Gates

The runtime only serves `optimizer_mode=dreamer_v3` when:

- the manifest declares release-ready runtime compatibility
- the model artifact SHA-256 matches configuration
- tensor and schema contracts verify
- deterministic inference parity succeeds
- safety validation accepts the candidate recommendation

If any requirement is missing, BO remains the fallback for shot-level
recommendations.

Live adaptive Dreamer profile control is separate from shot-level
recommendations. It requires the explicit Dreamer-controlled profile, verified
model readiness, bounded command contracts, live telemetry validation, and
machine capability checks. It must not silently switch to BO mid-shot.
