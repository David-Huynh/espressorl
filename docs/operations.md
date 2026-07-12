# EspressoRL Operations

## Public Runtime

A public deployment connects to one machine's MQTT broker and owns its local
optimization state. It receives canonicalized Gaggimate shot and preference
events, stores them in SQLite or Postgres, and publishes CPBO recommendations.

Required MQTT topics:

```text
gaggimate/{topic_id}/shot/profile
gaggimate/{topic_id}/machine/state
gaggimate/{topic_id}/rl/preference
gaggimate/{topic_id}/rl/recommendation
gaggimate/{topic_id}/rl/status
gaggimate/{topic_id}/rl/shot/ack
```

Shot profiles and acknowledgements use QoS 1. The acknowledgement is an
adapter protocol emitted only after the canonical application use case returns:

- `accepted`: the physical shot was stored and processing completed.
- `already_processed`: an exact immutable replay was handled idempotently.
- `transient_failure`: storage or runtime processing was unavailable; retry.
- `permanent_rejection`: the payload or application classification is terminal;
  do not retry.

Every receipt carries the original `shot_id`, topic-bound `machine_id`, a
bounded reason code, retryability, and timestamp. Shot `machine_id` must match
the MQTT topic. Exception text and infrastructure details are never published.
The receipt is non-retained, so a lost receipt causes a safe idempotent replay
rather than accepting stale broker state.

Corrections may exclude bad puck preparation, utility brews, or unobserved
recipe controls. Failed and aborted physical shots remain operational records
but never become preference observations.

## Local Dashboard

```json
{
  "deployment_role": "public",
  "local_dashboard_enabled": true,
  "local_dashboard_port": 8081,
  "local_dashboard_token": "AT_LEAST_32_RANDOM_CHARS"
}
```

The dashboard can inspect local physical shots, CPBO eligibility, upload queue
state, and context resets. It must not render credentials, raw signed payloads,
request headers, or complete telemetry blobs.

## Community Upload

Community upload is optional and independent from recommendation generation:

```json
{
  "community_upload_enabled": true,
  "supabase_registration_url": "https://PROJECT_REF.supabase.co/functions/v1/espresso-rl-register",
  "supabase_ingest_url": "https://PROJECT_REF.supabase.co/functions/v1/espresso-rl-ingest"
}
```

If credentials are absent, the public service requests anonymous HMAC upload
credentials and stores them locally. Before every upload it validates the
allowlisted schema, recomputes the payload SHA-256, verifies credential
ownership, and signs canonical JSON.

The ingress queue accepts three independent records:

- `shot_record`: physical recipe, realized outcome, and telemetry
- `recommendation_record`: recommendation lifecycle and follow-through audit
- `comparison_record`: oriented new/anchor shot IDs and one JND label

Numeric ratings and scalar rewards are rejected. A delayed comparison never
withholds a useful physical trajectory. Transient failures retry; permanent
schema or credential failures discard only the upload snapshot and preserve
local data.

Community upload has its own signed HTTP queue. Local MQTT retries never enqueue
another community copy; the two delivery lifecycles are independent.

Deploy current Supabase resources with:

```bash
supabase db push
supabase functions deploy espresso-rl-register --no-verify-jwt
supabase functions deploy espresso-rl-ingest --no-verify-jwt
supabase functions deploy espresso-rl-grinder-search --no-verify-jwt
```

The grinder catalog is reviewed metadata. Public clients have bounded,
rate-limited read access through the Edge Function and no direct table write
access. Unknown grinder geometry remains `NULL` rather than being fabricated.

## Admin Mirror

Admin mode runs separately from the machine-connected public service. Its
Postgres warehouse is the durable community-data store; Supabase is only a
small, short-lived ingress queue.

```json
{
  "deployment_role": "admin",
  "storage_backend": "postgres",
  "postgres_dsn": "postgresql://USER:PASSWORD@HOST:5432/espresso_rl_admin",
  "admin_collector_enabled": true,
  "admin_dashboard_enabled": true,
  "admin_dashboard_port": 8080,
  "supabase_rest_url": "https://PROJECT_REF.supabase.co/rest/v1",
  "supabase_service_role_key": "SERVICE_ROLE_KEY",
  "admin_dashboard_token": "AT_LEAST_32_RANDOM_CHARS",
  "admin_source_purge_enabled": true,
  "admin_source_mirrored_retention_days": 1,
  "admin_source_rejected_retention_days": 30,
  "admin_source_failed_retention_days": 90
}
```

Prefer environment secrets for the Postgres DSN, service-role key, and
dashboard token.

Every collector cycle:

1. claims queued rows with a bounded lease and `FOR UPDATE SKIP LOCKED`
2. writes the raw row into admin Postgres
3. marks the Supabase row `mirrored` only after the local write succeeds
4. validates and promotes allowlisted data into normalized warehouse tables

Once per hour, the collector invokes source retention. Only terminal
`mirrored` rows older than one day are routinely removed. Active `mirroring`
leases and queued rows are never purged. Rejected and mirror-failed rows remain
longer for diagnosis.

The normalized tables `community_validated_shots` and
`community_comparisons` are not removed by source retention. They are the
long-term records used for exports. The local `community_raw_uploads` staging
table can be purged separately after validation.

Validation rejects spoofed ownership, event/hash mismatches, unknown fields,
non-finite or impossible trajectories, reversed comparison orientation,
scalar taste fields, and non-espresso utility records. Recommendation records
are audit data, not subjective outcomes.

## Offline Export

```bash
uv run espresso-rl-export-offline-dataset \
  --postgres-dsn "$ESPRESSORL_POSTGRES_DSN" \
  --output-dir offline_dataset
```

The exporter reads only normalized admin warehouse tables. It joins each
trusted comparison to both shots, validates shared context and orientation,
then writes deterministic JSONL and a SHA-256 manifest. See
[offline-dataset.md](offline-dataset.md).

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
```
