# EspressoRL Standalone Container

EspressoRL is a local espresso dial-in service for Gaggimate. It communicates
through MQTT, stores shot history in Postgres, and publishes conservative BO
recommendations back to Gaggimate.

```text
Gaggimate -> MQTT broker -> EspressoRL -> MQTT broker -> Gaggimate
                         -> local Postgres
                         -> optional Supabase upload
```

The active runtime path follows the ports-and-adapters architecture described in
`EspressoRL_DESIGN.md`:

```text
Gaggimate MQTT adapter -> canonical events -> application service
application service -> repository/optimizer ports -> domain logic
Postgres repositories <- repository ports
Conservative BO optimizer -> domain models/safety only
```

The core packages are:

- `espresso_rl.domain`: canonical events, shot/recommendation models, profile
  resampling, grind normalization, reward, safety, and follow-through logic.
- `espresso_rl.application`: use cases for shot ingestion, feedback,
  recommendation decisions, recommendation apply acknowledgements, reward
  recomputation, and next recommendation generation.
- `espresso_rl.ports`: repository and optimizer interfaces.
- `espresso_rl.optimizers`: machine-agnostic conservative BO implementation.
- `espresso_rl.adapters`: Postgres/SQLite persistence, signed upload,
  Supabase queue access, and Gaggimate MQTT translation.

DreamerV3 modules remain present but are not wired into the active
recommendation path until they can pass through the same safety,
recommendation-memory, and follow-through gates.

## Quick Start

Copy the example config and edit it:

```bash
mkdir -p data
cp data/options.example.json data/options.json
```

Important fields in `data/options.json`:

```json
{
  "mqtt_host": "192.168.1.85",
  "mqtt_port": 1883,
  "mqtt_user": "mqtt",
  "mqtt_password": "replace-me",

  "machine_id": "gaggimate:YOUR_GAGGIMATE_TOPIC_ID",
  "bean_context_id": "current_bean",
  "grinder_context_id": "primary_grinder",
  "grinder_step_size_um": 12.5,
  "initial_grind_steps": null,
  "initial_grind_um": 525.0,
  "initial_dose_g": 18.0,
  "initial_target_yield_g": 36.0,
  "optimizer_mode": "bayesian_optimization",
  "optimizer_model_artifact_path": "",
  "optimizer_model_artifact_sha256": "",
  "optimizer_model_manifest_path": "",
  "default_optimizer_model_artifact_sha256": "",
  "optimizer_model_artifact_max_bytes": 536870912,

  "storage_backend": "postgres",
  "postgres_dsn": "postgresql://espresso_rl:espresso_rl@postgres:5432/espresso_rl",

  "deployment_role": "public",
  "training_mode": false
}
```

If you know grinder steps, set `initial_grind_steps` and leave
`initial_grind_um` as `0.0`. If you know microns but not the step number, set
`initial_grind_steps` to `null` and provide `initial_grind_um`; EspressoRL will
derive internal steps from `grinder_step_size_um`.

Local optimizer state is scoped by `install_id`, `machine_id`,
`bean_context_id`, and `grinder_context_id`. Use a different
`grinder_context_id` when the same bean is dialed on a different grinder; local
shots and active recommendations will not be mixed across those grinder
contexts.

Gaggimate can override `optimizer_mode` at runtime by publishing a retained
`gaggimate/<machine>/rl/settings` payload. `bayesian_optimization` is always
available. DreamerV3 is only advertised to the Gaggimate UI when model artifact
metadata is configured, the local artifact file matches its SHA-256, and a
plain JSON model manifest verifies the dataset/trainer/schema provenance; until
active Dreamer inference is safety-gated, BO remains the effective
recommendation path.

Official Docker builds can embed a release-default model SHA with the
`ESPRESSORL_RELEASE_MODEL_ARTIFACT_SHA256` build arg, normally supplied by the
GitHub repository variable with the same name. If that default SHA is present
and no explicit path is configured, EspressoRL looks for
`/data/espresso_rl/models/dreamer_v3.safetensors` and
`/data/espresso_rl/models/dreamer_v3_manifest.json`. A trainer using their own
model can set `optimizer_model_artifact_path`,
`optimizer_model_artifact_sha256`, and `optimizer_model_manifest_path` in
`data/options.json` or via `ESPRESSORL_OPTIMIZER_MODEL_ARTIFACT_PATH`,
`ESPRESSORL_OPTIMIZER_MODEL_ARTIFACT_SHA256`, and
`ESPRESSORL_OPTIMIZER_MODEL_MANIFEST_PATH`; those explicit values override the
release default. Oversized artifacts are refused by
`optimizer_model_artifact_max_bytes`.

The model manifest is regular UTF-8 JSON. It must identify the model family,
model artifact format, model artifact SHA-256, training dataset SHA-256,
training dataset manifest SHA-256, trainer git SHA, training config SHA-256,
state/action/reward schema versions, and EspressoRL runtime schema
compatibility. Runtime inference artifacts must use `safetensors`; pickle-style
formats such as full `torch.save(model)` files are intentionally not accepted
by the manifest verifier. Example:

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
    "espresso_rl_runtime_schema_version": 1
  }
}
```

Dreamer action schema version `1` is also intentionally narrow and
machine-agnostic. Model output may propose only canonical relative recipe
actions:

```json
{
  "format": "espresso_rl_dreamer_action_v1",
  "schema_version": 1,
  "grind_delta_steps_from_current": -3,
  "next_dose_g": 18.5,
  "target_yield_g": 40.0,
  "target_ratio": 2.16,
  "confidence": 0.62,
  "reason": "DreamerV3 candidate."
}
```

Absolute grinder settings, profile edits, machine topics, and adapter payloads
are not part of the Dreamer action schema. The action space can use the same
full safety envelope available to BO, currently up to 5 relative grind steps,
1.0 g dose change, and 8.0 g target-yield change from the current recipe, while
still respecting global dose/yield/ratio bounds. In this pre-shot recipe schema,
`target_yield_g` is the initial planned stop target; future dynamic profiling
should represent in-shot yield changes as bounded per-step controls such as
`yield_stop_target_g` or `stop`.

Start the local service and Postgres:

```bash
docker compose up -d --build --remove-orphans
```

Follow logs:

```bash
docker compose logs -f espresso-rl
```

Expected startup lines:

```text
Using Postgres storage backend
Subscribed to gaggimate/+/shot/profile...
```

## MQTT Setup

EspressoRL and Gaggimate must point to the same MQTT broker.

The default Compose stack does not start an MQTT broker. Set `mqtt_host` to
your existing broker's LAN IP address or DNS name. If you deliberately run a
broker in another Compose project, use that broker's reachable host name from
the EspressoRL container.

Gaggimate should publish and subscribe on these topics:

```text
gaggimate/{topic_id}/shot/profile
gaggimate/{topic_id}/machine/state
gaggimate/{topic_id}/rl/rating
gaggimate/{topic_id}/rl/shot/correction
gaggimate/{topic_id}/rl/recommendation/decision
gaggimate/{topic_id}/rl/recommendation/apply
gaggimate/{topic_id}/rl/recommendation
gaggimate/{topic_id}/rl/status
```

If Gaggimate publishes to:

```text
gaggimate/AA_BB_CC_DD_EE_FF/shot/profile
```

then set:

```json
"machine_id": "gaggimate:AA_BB_CC_DD_EE_FF"
```

## Data-Collection MVP

You can start accumulating local BO data when:

- EspressoRL is running and connected to the same MQTT broker as Gaggimate.
- Gaggimate MQTT publishing is enabled.
- Gaggimate publishes `gaggimate/{topic_id}/shot/profile` at brew end.
- Gaggimate subscribes to `gaggimate/{topic_id}/rl/recommendation`.
- `data/options.json` has the current grinder step size, initial grind, dose,
  target yield, bean context, and grinder context.
- `storage_backend=postgres` points at a reachable Postgres DSN.

Recommendation flow:

```text
Gaggimate publishes shot/profile
  -> EspressoRL stores shot and waits for rating or explicit Skip
  -> Gaggimate publishes shot feedback with optional taste tags
  -> EspressoRL updates reward and generates a bounded BO recommendation
  -> EspressoRL publishes gaggimate/{topic_id}/rl/recommendation
  -> Gaggimate stores the pending recommendation
  -> Gaggimate UI shows the recommendation after feedback
  -> Gaggimate publishes decision and apply acknowledgement after Use
  -> next shot data determines actual follow-through
```

The recommendation payload includes grind delta, next dose target, target yield,
ratio, mode, confidence, reason, and IDs. Grind is recommendation-only because
Gaggimate cannot automate a grinder setting. When the user chooses Use, firmware
saves the selected-profile target yield when possible. It saves the recommended
grind-by-weight dose target only when Gaggimate grind-by-weight targeting is
enabled; otherwise it prompts the user to grind that dose manually.

Human rating is the primary reward signal. Optional taste tags also guide the
bounded BO step: under-extraction tags bias toward a slightly finer/longer
recipe, over-extraction tags bias coarser/shorter, and positive tags favor
holding near the observed recipe. Explicit Skip completes feedback with lower
confidence so the shot can still contribute profile evidence.

Use publishes both an accepted recommendation decision and
`gaggimate/{topic_id}/rl/recommendation/apply`. The apply acknowledgement records
which fields were applied by the machine and which fields still require manual
action, but it never counts as follow-through by itself. EspressoRL only marks a
recommendation followed after comparing the next actual shot data against the
recommendation. Choosing Later sends no decision, so the retained pending
recommendation can be shown again on the next wake/reconnect. Choosing Ignore
sends an ignored decision so the optimizer will not count it as followed.

Manual shot corrections are sent as
`gaggimate/{topic_id}/rl/shot/correction`. They can exclude the latest shot from
local optimization, mark bad puck prep/channeling, or mark grind/dose/yield as
not followed while preserving the shot record. EspressoRL updates the stored
shot in place, recomputes reward confidence, and requeues the corrected espresso
snapshot for community upload when upload is enabled.

EspressoRL also publishes retained `gaggimate/{topic_id}/rl/status` payloads.
Gaggimate can use that status to show service connectivity, last stored shot,
last recommendation, recommendation apply status, current BO mode, local shot
count, queued upload count, rejected upload count, uploaded snapshot count, and
community upload state. Utility flushes and other non-espresso shots are kept
out of the community upload queue so validation failures there do not mask real
espresso data collection.

If `upload_queue_rejected_count` is non-zero, use the Auto Tuning page's
`Retry Valid` action. EspressoRL locally preflights rejected payloads before
retrying them, resets only valid rejected rows back to pending, and leaves
invalid rows rejected. This is meant for recovery after ingestion/schema fixes;
it will not blindly retry malformed utility flushes or impossible espresso
payloads.

## Supabase Upload

Community upload is optional. Local recommendations and local Postgres data
collection work without Supabase.

`community_upload_enabled` is the backend deployment gate. It must be true for
the container to register upload credentials and run the upload worker, but it
does not override the user's Gaggimate setting. Gaggimate publishes the user's
anonymous community upload opt-in over MQTT, and EspressoRL only queues upload
snapshots when both the backend gate and that per-machine opt-in are enabled.
If EspressoRL has not received an explicit per-machine opt-in, uploads remain
disabled.

Public deployment settings:

```json
{
  "deployment_role": "public",
  "community_upload_enabled": true,
  "supabase_registration_url": "https://PROJECT_REF.supabase.co/functions/v1/espresso-rl-register",
  "supabase_ingest_url": "https://PROJECT_REF.supabase.co/functions/v1/espresso-rl-ingest",
  "upload_secret": "",
  "upload_token_id": ""
}
```

If upload is enabled without a configured `upload_secret`, EspressoRL uses the
registration URL to request anonymous upload credentials and stores them under
`/data/espresso_rl/community_upload_credentials.json`. The secret is not logged.
Before any queued payload is signed or retried, EspressoRL validates its schema
and verifies that the stored SHA-256 `payload_hash` still matches the queued
JSON. The signed upload adapter also refuses to sign payloads whose `install_id`
does not match the configured upload credential. Community upload validation is
allowlisted by `schema_version`; unknown fields, unsafe IDs/metadata strings,
non-finite numbers, invalid JSON shapes, and out-of-range physical values are
rejected before network upload or trusted admin storage.

The Supabase scaffold lives in `supabase/`:

- `supabase/migrations/202605290001_espresso_rl_raw_queue.sql`
- `supabase/migrations/202606250001_espressorl_grinder_catalog.sql`
- `supabase/migrations/202606270001_espressorl_grinder_catalog_seed.sql`
- `supabase/migrations/202606290001_espressorl_prior_rule_catalog.sql`
- `supabase/functions/espresso-rl-register/index.ts`
- `supabase/functions/espresso-rl-ingest/index.ts`
- `supabase/functions/espresso-rl-grinder-search/index.ts`
- `supabase/functions/espresso-rl-prior-rule-search/index.ts`

The grinder catalog is reviewed metadata, not client-owned runtime state. Add
new grinders with a PR that updates the catalog seed/migrations and includes
source quality for the click or marker scale. Unknown step sizes or ranges
should stay `NULL`; stepless grinders can use a practical dial-marker unit when
the marker pitch is sourced or measured. One physical grinder should remain one
catalog row: piecewise ranges and compound controls such as macro plus micro
rings belong in `metadata_json` so bean/grinder history is not fragmented.

The prior-rule catalog follows the same reviewed PR workflow. It contains only
bounded declarative condition/direction records; it cannot contain executable
expressions or exact recipe deltas. Gaggimate searches pack names live and
stores only rules the user selected.

Deploy it with:

```bash
supabase db push
supabase functions deploy espresso-rl-register --no-verify-jwt
supabase functions deploy espresso-rl-ingest --no-verify-jwt
supabase functions deploy espresso-rl-grinder-search --no-verify-jwt
supabase functions deploy espresso-rl-prior-rule-search --no-verify-jwt
```

## Local Data Dashboard

Public machine-connected containers can expose a small local dashboard for
testing and cleanup. This dashboard manages only the local public-container
database; it does not mirror Supabase, generate community priors, or use the
service-role key.

```json
{
  "deployment_role": "public",
  "local_dashboard_enabled": true,
  "local_dashboard_port": 8081,
  "local_dashboard_token": "AT_LEAST_32_RANDOM_CHARS"
}
```

The local dashboard shows which shots are included in local BO, which shots are
excluded/utility/rejected-upload records, and context-level BO/rated counts. It
can:

- dry-run or purge local shots that are already useless for BO
- exclude a selected shot from BO without deleting history
- delete a selected local shot
- reset a bean context so old shots no longer train local BO
- retry or clean local rejected upload queue rows

The dashboard requires a bearer token and never renders upload secrets, Supabase
keys, raw payload JSON, raw profiles, or raw request headers. Deleting a shot is
scoped to the configured `install_id` and `machine_id`. A Supabase rejection does
not automatically delete local optimizer evidence; mark the shot excluded or
delete it explicitly when you know it is bad data.

## Admin Mirror And Validation

Admin mode is a separate deployment for mirroring the Supabase raw queue into an
admin Postgres database. Do not run admin mode in the same container that is
attached to your espresso machine for daily use.

Admin settings:

```json
{
  "deployment_role": "admin",
  "admin_collector_enabled": true,
  "admin_dashboard_enabled": true,
  "admin_dashboard_port": 8080,
  "training_export_dir": "/data/espresso_rl/exports",
  "training_export_max_rows": 50000,
  "supabase_rest_url": "https://PROJECT_REF.supabase.co/rest/v1",
  "supabase_service_role_key": "SERVICE_ROLE_KEY",
  "admin_dashboard_token": "AT_LEAST_32_RANDOM_CHARS"
}
```

For Docker/TrueNAS, prefer setting `ESPRESSORL_ADMIN_DASHBOARD_TOKEN` as a
secret/environment value instead of putting it in a visible app form. The
dashboard never renders tokens, upload secrets, Supabase service-role keys, raw
HMAC material, raw request headers, or raw payloads. The browser keeps the
admin token only in page memory and sends it as a bearer token for API calls.

The admin mirror claims rows with `espressorl_claim_raw_uploads`, which uses a
short lease and `FOR UPDATE SKIP LOCKED`. Multiple admin mirrors can poll at the
same time without claiming the same row. Mirrored rows are retained in Supabase
for a short audit/debug window and later removed by
`espressorl_purge_raw_upload_queue`.

After mirroring, the admin worker validates local `community_raw_uploads` rows
before they can enter trusted warehouse tables. Validation rejects spoofed
payload install IDs, event-type mismatches, payload-hash mismatches, malformed
payloads, impossible espresso values, invalid taste tags, unsafe profile arrays,
and non-espresso utility shots. The admin Supabase adapter rejects malformed raw
queue rows instead of coercing them into objects. Accepted shot uploads are
stored in `community_validated_shots` as sanitized allowlisted payloads with a
capped low trust weight and are copied into `training_dataset` only when their
trust weight is non-zero.
Recommendation uploads are stored for audit and follow-through analysis, but
they are not training rows by themselves.

The admin worker can also generate released community priors into
`community_priors`. These are weak prior summaries, not trusted commands. The
generator rereads and revalidates each training row, requires multiple
independent installs or sufficient within-install diversity for a context, caps
repeated per-install contribution inside narrow context/action/profile buckets,
applies diminishing per-install influence within each context key, uses robust
median/trimmed aggregation, and caps confidence at a low value so real local
shots can override the prior quickly. High-volume installs are allowed and
valuable when they span many bean contexts, recipe ranges, and profile shapes;
repetitive data from one narrow context remains stored for diagnostics and
experiments, but does not dominate released public priors.
Released priors carry a bounded low-confidence point cloud, not only a single
median recipe, so early optimizers can infer a rough direction from many weak
validated examples while still letting local shots dominate.

Manual dashboard actions are locked per job type. If `Run validation now`,
`Generate priors now`, `Purge queue now`, or another manual action is clicked
while that same job is already running, the service returns the current status
instead of starting a second run. Manual actions support `dry_run=true` where
practical: validation and prior generation report proposed counts without
mutating trusted training state or released priors; mirror dry-run does not
claim Supabase rows because claiming takes a remote lease; purge dry-run does
not call the Supabase purge RPC because that RPC deletes rows. Each manual
action writes an `admin_action_log` entry containing action type, requester
label, dry-run flag, status, row counts,
warning count, and a short sanitized error summary. Rejection summaries exposed
by the dashboard are category-only, such as `invalid_schema`,
`invalid_signature`, `rate_limited`, `impossible_flow`, `duplicate_shot_id`, and
`payload_too_large`.
Local and admin dashboards require bearer tokens, reject oversized request
bodies, set no-store/security headers, and avoid rendering raw payload JSON,
profiles, request headers, Supabase keys, upload secrets, or HMAC material.

The admin dashboard can export the validated training dataset for external
model training. Exports are written under `training_export_dir` and are capped
by `training_export_max_rows`. The export format is intentionally boring and
auditable:

- `training_rows.jsonl`: canonical UTF-8 JSON Lines, one
  `espresso_rl_training_transition_v1` object per line. This is the
  authoritative trainer input.
- `training_rows.csv`: a spreadsheet-friendly summary for review. It omits the
  profile arrays and escapes formula-looking string cells.
- `manifest.json`: row counts, file hashes, dataset SHA-256, schema version,
  canonical row format, source git SHA, and zero-trust flags.
- `README.txt`: plain-language format notes.

Each JSONL transition has `source`, `context`, `action`, `recommendation`,
`observation`, and `reward` sections. It is not a raw upload dump. The exporter
revalidates each row, strips adapter payload shape, excludes absolute grinder
display fields, and canonicalizes grind as relative steps plus relative microns
from the grinder context reference. Not-followed or ignored recommendations may
remain visible for analysis, but their recommendation attribution weight is
zero so they are not treated as successful optimizer actions. Sequence trainers
should group rows by `install_id`, `machine_id`, `bean_context_id`, and
`grinder_context_id`.

The Dreamer helper `espresso_rl.dreamer.dataset.load_dreamer_episodes_from_jsonl`
converts those canonical rows into `espresso_rl_dreamer_episode_v2` shot
episodes for recurrent training. Dreamer uses the additional named
`fixed_cadence_sequence`, resampled onto exact 250 ms intervals from the first
real telemetry sample. Shots remain variable length and are padded only during
batching. Each step contains pressure, pump flow in `ml/s`, beverage mass flow
in `g/s`, weight, boiler temperature, profile targets, explicit pump target
mode, and valve state. The fixed 5x100 profile remains a duration-normalized
summary for BO, profile scoring, and compatibility; it is not Dreamer's
recurrent clock. Gaggimate MQTT `flow` is never compared directly with its
`target_flow`.
Relative grind, dose, the initial planned yield target, grinder calibration,
and the default taste objective stay in `static_context`. Historical
pressure/pump-flow/temperature setpoints are stored separately as observed profile
targets with a matching active mask, while `dynamic_action` remains null unless
a future capability-gated control path supplies safe per-step actions such as
`yield_stop_target_g` or `stop`. Pressure and flow targets are represented as
profile targets/control modes, not as proof that pressure and flow are
independently writable on a given machine. Dreamer episode construction requires
sampled actual temperature, sampled resolved target temperature, explicit
per-step pump target mode, and valve state. It also requires sampled beverage
flow separately from pump flow. Scalar profile/final temperatures remain
metadata and are never expanded into fabricated live telemetry.
`espresso_rl.dreamer.dataset.build_dreamer_episode_batch` then turns validated
episodes into deterministic tensors for offline training: observations,
observed profile targets plus active masks, dynamic action tensors plus
presence masks, constraints, static context, terminal features, rewards,
continuations, and step padding masks. The batch includes feature-name metadata
so external trainers can audit the numeric layout instead of relying on
implicit column order. `elapsed_seconds` advances by 0.25 on every valid step,
and `step_duration_seconds` is 0.25 for valid steps and zero for padding.
The batch also includes a Dreamer control spec, a `decision_step_mask`, and a
`control_action_mask`. Observations use the 250 ms recurrent clock, while actor
decisions default to 1000 ms and must never be faster than the supported control
cadence. Decision steps are fixed and equally spaced; the batcher forward-fills
the latest decision as the held action between decision ticks. BO and other
shot-level optimizers do not emit adaptive in-shot profile controls.

No export artifact uses pickle, model binaries, SQLite dumps, parquet, macros,
absolute grinder settings, or another executable/opaque format. Trainers should
publish the model file SHA-256 separately and configure it with
`optimizer_model_artifact_sha256`.

The offline artifact-contract builder consumes the canonical export and creates
the expected DreamerV3 release filenames:

```bash
uv run espresso-rl-build-dreamer-artifacts \
  --write-default-config training_config.json

uv run espresso-rl-build-dreamer-artifacts \
  --write-default-config training_config.json \
  --artifact-stage world_model_smoke

uv run espresso-rl-build-dreamer-artifacts \
  --write-default-config training_config.json \
  --artifact-stage world_model_train_preview

uv run espresso-rl-build-dreamer-artifacts \
  --dataset-jsonl training_rows.jsonl \
  --dataset-manifest manifest.json \
  --training-config training_config.json \
  --output-dir model_out \
  --trainer-git-sha TRAINER_REPO_COMMIT
```

It validates the dataset hash, revalidates every JSONL transition, rejects
absolute grinder fields, builds `espresso_rl_dreamer_episode_v2` episodes,
constructs deterministic Dreamer tensors under the configured control spec,
writes tensor feature/cadence hashes into the audit report, optionally runs a
deterministic CPU-only `world_model_smoke` step through the reference-aligned
categorical RSSM world model, and writes only fixed safe filenames.
For a larger offline-loop preview, `world_model_train_preview` deterministically
splits episodes into train/validation sets, runs a bounded CPU-only
fixed-cadence recurrent world-model training loop, and records train/validation
loss curves, dyn/rep KL losses, best epoch, split hash, model-size fields, and
hyperparameters in `audit_report.json`. It also runs an audit-only DreamerV3
imagination preview from posterior RSSM starts through the prior, using masked
actor heads for static recipe deltas and dynamic controls plus a symlog/two-hot
critic and lambda-return targets. The preview stage now performs a bounded
deterministic actor/critic training loop in latent imagination and records
actor loss, critic loss, entropy, imagined return, and dynamic-control mask
metrics in the audit report. After training, it writes a deterministic offline
evaluation report covering world-model validation loss, reward/continuation
calibration, critic value error, actor entropy, imagined-return stability, and
dynamic-action mask conformance.

It produces:

- `dreamer_v3.safetensors`
- `dreamer_v3_manifest.json`
- `training_config.json`
- `audit_report.json`
- `checksums.txt`

This command is intentionally an artifact pipeline skeleton. The generated
`.safetensors` file is a valid checkpoint container but remains
`inference_ready=false`, so the runtime verifier will not expose it as an active
DreamerV3 model. The
`world_model_smoke` stage proves that the exported tensors can run through the
same reference-aligned categorical RSSM path used by the preview trainer and
records initial/final losses in `audit_report.json`; `world_model_train_preview`
extends that to deterministic train/validation curves plus actor/critic
training curves from latent imagination and an offline evaluation report.
Neither stage produces a useful runtime model artifact. The train-preview stage
serializes its deterministic RSSM, trained actor-head, and trained critic-head
tensors with explicit tensor names, shapes, component metadata, feature-layout
hash, control-spec hash, tensor-manifest hash, and evaluation-report hash;
runtime compatibility should only be set after inference is safe. The command
has a configurable `--max-dataset-bytes` resource
guard, defaulting to 8 GiB, because this skeleton validates JSONL in-process.
That guard is not a training policy; real large-scale Dreamer training should
use streaming or sharded dataset loading so the corpus can grow beyond one
in-memory JSONL file.

The runtime checkpoint loader reads model bundle members through the
`ModelArtifactStore` port. The local filesystem implementation is an adapter;
checkpoint validation remains in the application/domain layers. Loading is
strict and non-executable: it accepts only the checkpoint safetensors contract,
rejects duplicate/unknown JSON fields, verifies the configured artifact hash,
manifest and schema versions, safetensors metadata, tensor names/shapes/offsets,
per-tensor hashes, component totals, and optional runtime-owned feature-layout,
control-spec, and tensor-contract hashes. It never calls `torch.load` or any
pickle loader.

A successfully loaded preview is reported as `checkpoint_verified=true` and
`checkpoint_inference_ready=false`. That distinction is intentional: verified
bytes may be safe for shadow evaluation without being a release-approved active
policy. `dreamer_v3_shadow` remains the default Dreamer-compatible mode while
Bayesian Optimization serves recommendations.

Checkpoint contract v2 also authenticates the exact world-model, actor, and
critic reconstruction configuration. The trainer records a deterministic
inference-probe hash from the original modules, serializes the checkpoint,
reloads it through the same strict runtime loader, and requires the reloaded
modules to reproduce both that probe and the held-out validation inference
hash. Materialized modules are CPU-only, evaluation mode, and have gradients
disabled. Runtime status exposes checkpoint parity separately from inference
readiness; parity does not enable recommendations or machine control.
Earlier v1 preview checkpoints do not contain authenticated reconstruction
metadata and must be regenerated; they are rejected rather than inferred.

Parity-verified checkpoints can run context-conditioned shadow evaluation after
a rated local shot. Only canonical shots with fixed-cadence telemetry and exact
bean/grinder context are accepted. Dreamer emits a static recipe proposal in
relative grind steps plus dose and yield; the existing Dreamer safety validator
checks it without clamping unsafe output. The proposal, its safety result, and
the context-matched BO comparison are stored in `dreamer_shadow_evaluations`
through the `ShadowEvaluationRepository` port, with SQLite and Postgres
adapters. The next shot resolves only the pending record for the same install,
machine, bean, and grinder context. Reward deltas are aggregated only when the
actual next recipe matched the corresponding proposal, avoiding unsupported
counterfactual attribution.

Shadow proposals are never inserted into the recommendation repository and are
never published or applied. BO remains the sole active recommendation path.
Status output contains aggregate shadow counts and safety/outcome metrics, not
actionable shadow proposals.

The Dreamer candidate builder is shared by shadow evaluation and the active
runtime path. It consumes only canonical `OptimizationContext` shots, rejects
mixed or future context replay, supports unknown observed grind fields when the
current recipe is known, emits relative-grind `dreamer_candidate`
recommendations, and reuses the normal Dreamer action and recommendation safety
validators.

Active Dreamer is a code/config switchover, not a per-user shadow promotion.
The runtime only serves `optimizer_mode=dreamer_v3` when a release-approved
manifest declares `runtime_compatibility.inference_ready=true`, the model
artifact SHA-256 matches, checkpoint tensor parity succeeds, and a Dreamer
optimizer implementation is wired in. If any requirement is missing or the
candidate fails safety validation, BO remains the fallback. Shadow evaluation is
for admin/beta evidence and defaults, not a normal-user prerequisite.
That BO fallback applies only to shot-level Dreamer recipe recommendations.
Future Dreamer automatic/adaptive profile control must use an explicit
Dreamer-controlled profile and fail safe if the model is unavailable or unsafe;
it must not silently switch to BO in the middle of live machine control. Runtime
status exposes only aggregate active-Dreamer success and BO-fallback counts with
sanitized reasons, not raw model proposals.

## Warm-Started BO Priors

Runtime recommendation generation can consume canonical `PriorPoint` values
from local history and released community priors.
Community prior JSON is treated as hostile at read time: the provider requires
the expected context key and zero-trust metadata, revalidates numeric fields,
caps confidence, enforces minimum observation noise, and emits only weak
canonical prior points. The optimizer uses priors only while local data is
sparse, starts after the first real shot receives feedback, applies normal
trust-region and safety bounds, and stops using external priors once enough
local shots exist. Service startup clears stale retained MQTT recommendations
when no active recommendation exists instead of publishing a no-op baseline.
Previous bags of the same normalized bean on the same grinder are converted into
up to 64 local `local_bean_history` prior points after the new bag has at least
one valid local observation. These points are ranked and down-weighted by rank,
so they can provide an initial directional shape without replacing the current
bag's real shots. `OptimizationContext.prior_points` carries empirical history
while `OptimizationContext.prior_signals` carries optional user/community rule
directions. Rules state only semantic direction (`finer`, `coarser`,
`increase`, or `decrease`); BO selects magnitude from its trust region and
local evidence. Semantic grind direction is converted through the active
grinder's step direction, so dial numbering is never embedded in a rule. Both
streams are optimizer-agnostic and contain no adapter-specific data.

Per-machine prior modes are `no_priors`, `community_only`, and
`rules_and_community`. Rules are bounded to 16, strictly allowlisted at the
firmware and backend boundaries, confidence-capped, and ignored after the
fifth local shot. Local observations remain dominant and all recommendations
still pass normal safety validation.

Run local verification with:

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src tests
```

## License

EspressoRL is licensed under the GNU Affero General Public License v3.0 or
later. See `LICENSE` for the full license text.
