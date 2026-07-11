# EspressoRL Operations

This page contains the operational details that do not belong in the README
primer.

## Local Data Collection

You can collect local optimizer data when:

- EspressoRL is running and connected to the same MQTT broker as Gaggimate.
- Gaggimate publishes `gaggimate/{topic_id}/shot/profile` at brew end.
- Gaggimate publishes oriented preference feedback on
  `gaggimate/{topic_id}/rl/preference`.
- Gaggimate subscribes to `gaggimate/{topic_id}/rl/recommendation`.
- `data/options.json` has the current machine, bean, grinder, dose, and target
  yield context.
- `storage_backend=postgres` points at a reachable Postgres DSN.

EspressoRL keeps utility flushes and non-espresso records out of optimization
and community upload. Manual shot corrections are sent on:

```text
gaggimate/{topic_id}/rl/shot/correction
```

Corrections can exclude a shot from local optimization, mark bad puck prep or
channeling, or mark grind/dose/yield as not followed while preserving the shot
record.

## Local Dashboard

Public machine-connected containers can expose a local dashboard for testing
and cleanup:

```json
{
  "deployment_role": "public",
  "local_dashboard_enabled": true,
  "local_dashboard_port": 8081,
  "local_dashboard_token": "AT_LEAST_32_RANDOM_CHARS"
}
```

The dashboard manages only the local public-container database. It can inspect
local BO inclusion state, retry or clean local rejected uploads, exclude a shot
from BO, delete a selected local shot, and reset local context data.

The dashboard requires a bearer token and must not render upload secrets,
Supabase keys, raw request headers, raw payload JSON, or raw profiles.

## Supabase Upload

Community upload is optional and best-effort. Transient network/server failures
retry in the background. Schema or credential rejections discard only the queued
upload snapshot and retain local optimizer evidence.

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

If upload is enabled without a configured `upload_secret`, EspressoRL requests
anonymous upload credentials from the registration function and stores them
under:

```text
/data/espresso_rl/community_upload_credentials.json
```

Before upload, EspressoRL validates the queued payload schema, verifies its
stored SHA-256 hash, and signs the canonical JSON body. The signed upload
adapter refuses to sign payloads whose `install_id` does not match the upload
credential.

Deploy Supabase resources with:

```bash
supabase db push
supabase functions deploy espresso-rl-register --no-verify-jwt
supabase functions deploy espresso-rl-ingest --no-verify-jwt
supabase functions deploy espresso-rl-grinder-search --no-verify-jwt
supabase functions deploy espresso-rl-prior-rule-search --no-verify-jwt
```

The public upload queue stores raw signed payloads only. It accepts physical
`shot_record`, recommendation lifecycle `recommendation_record`, and
optimizer-neutral `comparison_record` payloads. A comparison identifies the
new and anchor physical shots, preserves orientation, and records exactly one
of `new_better`, `anchor_better`, or `tie`. Numeric ratings and scalar rewards
are rejected by this community contract.

Gaggimate queues physical shots and later preference comparisons as independent
retryable records. A missing or delayed comparison never withholds a useful
physical trajectory. Admin validation promotes accepted records into separate
trusted warehouse tables, and dataset builders join them by shot ID.

## Grinder And Prior Catalogs

The grinder catalog is reviewed metadata, not client-owned runtime state. Add
new grinders with a PR that updates the seed/migrations and includes source
quality for click or marker scale data.

Unknown step sizes or ranges should stay `NULL`. Stepless grinders can use a
practical dial-marker unit when marker pitch is sourced or measured. One
physical grinder should remain one catalog row; piecewise ranges and compound
controls belong in metadata so bean/grinder history is not fragmented.

The prior-rule catalog follows the same reviewed PR workflow. It contains only
bounded declarative condition/direction records. It cannot contain executable
expressions or exact recipe deltas.

## Admin Mirror

Admin mode is a separate deployment for mirroring the Supabase raw queue into
an admin Postgres database. Do not run admin mode in the same container that is
attached to your espresso machine for daily use.

Admin settings:

```json
{
  "deployment_role": "admin",
  "admin_collector_enabled": true,
  "admin_dashboard_enabled": true,
  "admin_dashboard_port": 8080,
  "supabase_rest_url": "https://PROJECT_REF.supabase.co/rest/v1",
  "supabase_service_role_key": "SERVICE_ROLE_KEY",
  "admin_dashboard_token": "AT_LEAST_32_RANDOM_CHARS"
}
```

For Docker or TrueNAS, prefer setting `ESPRESSORL_ADMIN_DASHBOARD_TOKEN` as an
environment secret instead of putting it in a visible app form.

The admin mirror claims rows with a short lease and `FOR UPDATE SKIP LOCKED`.
Multiple admin mirrors can poll at the same time without claiming the same row.
Mirrored rows are retained in Supabase for a short audit/debug window and later
removed by the purge RPC.

Admin validation rejects spoofed install IDs, event-type mismatches,
payload-hash mismatches, malformed payloads, impossible espresso values,
unsafe profile arrays, scalar taste scores, reversed comparison orientation,
and non-espresso utility shots. Accepted shot uploads are stored as sanitized
allowlisted physical observations with capped low trust weight.

Recommendation uploads are stored for audit and follow-through analysis, but
they are not subjective outcomes by themselves.

Pairwise feedback is stored in `community_comparisons`. The table is not tied
to CPBO: online preference optimizers and future offline learners can join each
comparison to the two immutable shot recipes and telemetry trajectories.

The pre-release `training_dataset` copy table and scalar `community_priors`
table are removed on admin Postgres startup. The old scalar/Dreamer community
export action is disabled. A future offline dataset builder will join sanitized
shots and comparisons into a separately versioned artifact; no numeric-rating
compatibility data is carried forward.

## Local Verification

```bash
uv run python -m unittest discover -s tests
uv run python -m compileall -q src tests
```
