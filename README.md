# EspressoRL Home Assistant Add-on

This add-on hosts the local EspressoRL service. The active runtime path follows
the ports-and-adapters layout described in `EspressoRL_DESIGN.md`:

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
- `espresso_rl.adapters`: Postgres/SQLite persistence, signed upload, and
  Gaggimate MQTT translation.

The old rushed active agent/replay-buffer/MQTT path has been removed. DreamerV3
modules remain present but are not wired into the active recommendation path
until they can pass through the same safety, recommendation-memory, and
follow-through gates.

Implemented backend behavior includes canonical shot, feedback, decision,
apply-acknowledgement, and machine-state events; recommendation memory;
wake/idle recommendation display; stale recommendation expiry; actual-shot
follow-through inference; low-confidence profile-only rewards; retained add-on
status reporting; a Postgres runtime storage backend; and an opt-in signed
upload queue for a Supabase Edge Function or compatible ingestion endpoint.

## Data-collection MVP

You can start accumulating local BO data when:

- the add-on is running and connected to the same MQTT broker as Gaggimate
- Gaggimate has Home Assistant over MQTT enabled
- the nested EspressoRL Auto Tuning setting is enabled in the Gaggimate
  Home Assistant/MQTT settings
- Gaggimate publishes `gaggimate/{mac}/shot/profile` at brew end
- Gaggimate subscribes to `gaggimate/{mac}/rl/recommendation`
- the add-on options set the current grinder step size, initial grind, dose,
  and target yield
- `storage_backend=postgres` points at a reachable Postgres DSN

Recommendation flow:

```text
Gaggimate publishes shot/profile
  -> add-on stores shot and generates bounded BO recommendation
  -> add-on publishes gaggimate/{mac}/rl/recommendation
  -> Gaggimate stores the pending recommendation
  -> LVGL and WebUI show the recommendation/rating prompts
  -> Gaggimate publishes decision and apply acknowledgement after Use
  -> next shot data determines actual follow-through
```

The recommendation payload includes grind delta, next dose target, target yield,
ratio, mode, confidence, reason, and IDs. Grind is recommendation-only because
Gaggimate cannot automate a grinder setting. When the user chooses Use, firmware
saves the selected-profile target yield when possible. It saves the recommended
grind-by-weight dose target only when Gaggimate grind-by-weight targeting is
enabled; otherwise it prompts the user to grind that dose manually.

Use publishes both an accepted recommendation decision and
`gaggimate/{mac}/rl/recommendation/apply`. The apply acknowledgement records
which fields were applied by the machine and which fields still require manual
action, but it never counts as follow-through by itself. EspressoRL only marks a
recommendation followed after comparing the next actual shot data against the
recommendation. Choosing Later sends no decision, so the retained pending
recommendation can be shown again on the next wake/reconnect. Choosing Ignore
sends an ignored decision so the optimizer will not count it as followed.

The add-on also publishes retained `gaggimate/{mac}/rl/status` payloads. The
Gaggimate WebUI settings page shows whether the add-on has been seen, the last
stored shot, the last recommendation, the recommendation apply status, the
current BO mode, local shot count, queued upload count, and community upload
state.

If EspressoRL Auto Tuning is disabled, Gaggimate still uses the Home
Assistant/MQTT plugin normally, but it does not publish EspressoRL shot profiles,
listen for BO recommendations, or send ratings.

Optional Supabase upload is controlled by:

- `community_upload_enabled`
- `supabase_ingest_url`
- `upload_secret`
- `upload_token_id`
- `addon_role`

If upload is enabled without an ingest URL or secret, records are queued locally
but not sent. Local recommendations and Postgres data collection continue either
way.

The Supabase ingestion scaffold lives in `supabase/`:

- `supabase/migrations/202605290001_espresso_rl_raw_queue.sql`
- `supabase/functions/espresso-rl-ingest/index.ts`

The Edge Function verifies the signed upload headers produced by the add-on,
checks timestamp and payload hash, applies basic rate limits and espresso sanity
bounds, and inserts only into `raw_upload_queue`. It does not write validated
shots, priors, training data, trust scores, or model tables.

Set `addon_role=admin` for an admin/training deployment. Admin mode may read the
community-fed Supabase queue, but it never pushes its own local records back
into Supabase, which prevents doubled data. SQLite remains available as a
development fallback adapter, but Postgres is the intended public-user and admin
runtime backend.

## Admin community mirror

An admin deployment can mirror the community-fed Supabase raw queue into local
Postgres:

```text
Supabase raw_upload_queue
  -> service-role admin collector
  -> local Postgres community_raw_uploads
  -> later validation/trust/training jobs
```

Enable it with:

- `addon_role=admin`
- `admin_collector_enabled=true`
- `supabase_rest_url`
- `supabase_service_role_key`
- `postgres_dsn`

The collector claims rows by calling the `espressorl_claim_raw_uploads` RPC,
which uses a short lease and `FOR UPDATE SKIP LOCKED`. Multiple admin mirrors can
poll at the same time without claiming the same row. The collector upserts
claimed rows into local Postgres, then marks only rows it owns as `mirrored` or
`mirror_failed`.

The Supabase table is a small raw queue, not the long-term warehouse. Rows are
not deleted immediately after mirroring; `espressorl_purge_raw_upload_queue`
clears old mirrored, rejected, and failed rows after retention windows. The
local admin Postgres database is where mirrored raw data waits for later
validation, trust scoring, and training jobs.

The expected Supabase raw queue shape includes `install_id`, `upload_id`,
`payload_hash`, `event_type`, `payload_json`, `received_at`, `status`,
`mirror_claimed_by`, `mirror_claim_expires_at`, `mirror_completed_at`, and
`mirror_error`. Public add-ons should only create raw queue rows through the
signed ingestion path; admin mode only mirrors and never reuploads.

Run local verification with:

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src tests
```
