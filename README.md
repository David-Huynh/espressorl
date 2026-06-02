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
  "grinder_step_size_um": 12.5,
  "initial_grind_steps": null,
  "initial_grind_um": 525.0,
  "initial_dose_g": 18.0,
  "initial_target_yield_g": 36.0,

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

Start the local service and Postgres:

```bash
docker compose up -d --build
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

If the broker is outside Docker, set `mqtt_host` to the broker IP address.
If the broker runs in the same Compose project, set `mqtt_host` to that service
name.

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
  and target yield.
- `storage_backend=postgres` points at a reachable Postgres DSN.

Recommendation flow:

```text
Gaggimate publishes shot/profile
  -> EspressoRL stores shot and generates bounded BO recommendation
  -> EspressoRL publishes gaggimate/{topic_id}/rl/recommendation
  -> Gaggimate stores the pending recommendation
  -> Gaggimate UI shows recommendation/rating prompts
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

The Supabase scaffold lives in `supabase/`:

- `supabase/migrations/202605290001_espresso_rl_raw_queue.sql`
- `supabase/functions/espresso-rl-register/index.ts`
- `supabase/functions/espresso-rl-ingest/index.ts`

Deploy it with:

```bash
supabase db push
supabase functions deploy espresso-rl-register --no-verify-jwt
supabase functions deploy espresso-rl-ingest --no-verify-jwt
```

## Admin Mirror And Validation

Admin mode is a separate deployment for mirroring the Supabase raw queue into an
admin Postgres database. Do not run admin mode in the same container that is
attached to your espresso machine for daily use.

Admin settings:

```json
{
  "deployment_role": "admin",
  "admin_collector_enabled": true,
  "supabase_rest_url": "https://PROJECT_REF.supabase.co/rest/v1",
  "supabase_service_role_key": "SERVICE_ROLE_KEY"
}
```

The admin mirror claims rows with `espressorl_claim_raw_uploads`, which uses a
short lease and `FOR UPDATE SKIP LOCKED`. Multiple admin mirrors can poll at the
same time without claiming the same row. Mirrored rows are retained in Supabase
for a short audit/debug window and later removed by
`espressorl_purge_raw_upload_queue`.

After mirroring, the admin worker validates local `community_raw_uploads` rows
before they can enter trusted warehouse tables. Validation rejects spoofed
payload install IDs, event-type mismatches, malformed payloads, impossible
espresso values, invalid taste tags, unsafe profile arrays, and non-espresso
utility shots. Accepted shot uploads are stored in `community_validated_shots`
with a capped low trust weight and are copied into `training_dataset` only when
their trust weight is non-zero. Recommendation uploads are stored for audit and
follow-through analysis, but they are not training rows by themselves.

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

## Warm-Started BO Priors

Runtime recommendation generation can consume canonical `PriorPoint` values
from local history, lightweight rule priors, and released community priors.
Community prior JSON is treated as hostile at read time: the provider requires
the expected context key and zero-trust metadata, revalidates numeric fields,
caps confidence, enforces minimum observation noise, and emits only weak
canonical prior points. The optimizer uses priors only while local data is
sparse, keeps the first shot as `zero_observe`, applies normal trust-region and
safety bounds, and stops using external priors once enough local shots exist.

Run local verification with:

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src tests
```

## License

EspressoRL is licensed under the GNU Affero General Public License v3.0 or
later. See `LICENSE` for the full license text.
