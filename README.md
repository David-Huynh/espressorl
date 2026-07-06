# EspressoRL

## Summary

EspressoRL is a local espresso dial-in service for Gaggimate. It listens to
Gaggimate shot and machine-state events over MQTT, stores local shot history,
and publishes bounded optimizer recommendations back to Gaggimate.

```text
Gaggimate -> MQTT broker -> EspressoRL -> MQTT broker -> Gaggimate
                         -> local Postgres
                         -> optional Supabase upload
```

The default active optimizer is conservative Bayesian optimization. DreamerV3
support is present behind model and safety gates, but BO remains the available
fallback path unless a verified release-ready Dreamer model is configured.

## Idea / Purpose

Espresso dial-in is a sequence of small experiments. EspressoRL keeps those
experiments organized by bean, grinder, and profile context, then uses the
actual shot result plus human feedback to recommend the next bounded change.

The core idea is:

- use local data first
- treat community data as optional low-trust prior evidence
- keep optimizer logic machine-agnostic
- require safety checks before any recommendation is published or applied
- infer follow-through from the next actual shot, not from button clicks

## Overview

EspressoRL is structured around canonical EspressoRL events and ports. External
systems such as MQTT, Gaggimate payloads, Postgres, SQLite, Supabase, and local
dashboards live in adapters.

Repository layout:

- `src/espresso_rl/domain`: canonical events, models, safety, reward,
  profile processing, follow-through, and validation logic.
- `src/espresso_rl/application`: use cases for shot ingestion, feedback,
  recommendation decisions, upload handling, admin validation, exports, and
  Dreamer artifact checks.
- `src/espresso_rl/ports`: interfaces used by the application core.
- `src/espresso_rl/adapters`: MQTT, SQLite, Postgres, Supabase, dashboards,
  and filesystem implementations.
- `src/espresso_rl/optimizers`: machine-agnostic optimizer implementations.
- `supabase`: public upload, registration, grinder search, and prior-rule
  catalog migrations/functions.
- `tests`: unit, adapter, contract, and integration-style tests using fakes.

## Installing

Copy the example config and edit it:

```bash
mkdir -p data
cp data/options.example.json data/options.json
```

Set at least these fields in `data/options.json`:

```json
{
  "mqtt_host": "192.168.1.85",
  "mqtt_port": 1883,
  "mqtt_user": "mqtt",
  "mqtt_password": "replace-me",
  "install_id": "local_install",
  "machine_id": "gaggimate:YOUR_GAGGIMATE_TOPIC_ID",
  "storage_backend": "postgres",
  "postgres_dsn": "postgresql://espresso_rl:espresso_rl@postgres:5432/espresso_rl",
  "optimizer_mode": "bayesian_optimization"
}
```

Start EspressoRL and Postgres:

```bash
docker compose up -d --build --remove-orphans
```

Follow logs:

```bash
docker compose logs -f espresso-rl
```

Expected startup output includes:

```text
Using Postgres storage backend
Subscribed to gaggimate/+/shot/profile...
```

The Compose stack does not include an MQTT broker. EspressoRL and Gaggimate must
both point at the same reachable broker.

## Usage

If Gaggimate publishes shots on:

```text
gaggimate/AA_BB_CC_DD_EE_FF/shot/profile
```

then set:

```json
{
  "machine_id": "gaggimate:AA_BB_CC_DD_EE_FF"
}
```

Gaggimate should publish:

```text
gaggimate/{topic_id}/shot/profile
gaggimate/{topic_id}/machine/state
gaggimate/{topic_id}/rl/rating
gaggimate/{topic_id}/rl/shot/correction
gaggimate/{topic_id}/rl/recommendation/decision
gaggimate/{topic_id}/rl/recommendation/apply
```

EspressoRL publishes:

```text
gaggimate/{topic_id}/rl/recommendation
gaggimate/{topic_id}/rl/status
gaggimate/{topic_id}/rl/dreamer/live_target
gaggimate/{topic_id}/rl/dreamer/fail_safe
```

Typical recommendation loop:

```text
1. Gaggimate publishes a completed shot profile.
2. EspressoRL stores the shot and waits for rating or skip feedback.
3. Gaggimate sends rating/taste feedback.
4. EspressoRL recomputes reward and generates a bounded recommendation.
5. EspressoRL publishes the recommendation to Gaggimate.
6. Gaggimate records accept/ignore/apply acknowledgement.
7. The next actual shot determines whether the recommendation was followed.
```

Apply acknowledgement is not treated as proof that a recommendation was
followed. Follow-through is inferred from the next actual shot data.

Optimizer state is scoped by:

```text
install_id
machine_id
bean_context_id
grinder_context_id
profile_id when available
```

Use separate grinder contexts for different grinders. Bean/grinder/profile
contexts keep local shot history and active recommendations isolated, while old
bags of the same normalized bean can still provide low-weight prior evidence
after the new context has at least one rated local shot.

## Optional Features

Community upload is optional. Local BO and local history work without Supabase.
Public containers only queue anonymous upload snapshots when backend upload is
enabled, Gaggimate has sent the user opt-in, and payloads pass local schema and
hash preflight. Supabase rejects or network failures do not delete local
optimizer evidence.

DreamerV3 artifacts must be non-executable, hash-verified, schema-compatible,
and explicitly release-ready before runtime use. `safetensors` is the accepted
runtime artifact format; pickle-style model files are intentionally rejected.

More detail:

- [docs/operations.md](docs/operations.md): Supabase, admin mirror, local
  dashboard, catalogs, and queue cleanup.
- [docs/model-artifacts.md](docs/model-artifacts.md): training exports,
  model manifests, release bundles, and Dreamer runtime gates.

## Development

Install dependencies with `uv`, then run tests:

```bash
uv sync
uv run python -m unittest discover -s tests
```

Useful focused checks:

```bash
uv run python -m compileall -q src tests
uv run python -m unittest tests.test_application_service
uv run python -m unittest tests.test_sqlite_and_boundaries
```

On this Windows setup, `uv` may be available at:

```powershell
C:\Users\David\.local\bin\uv.exe
```

## License

EspressoRL is licensed under the GNU Affero General Public License v3.0 or
later. See [LICENSE](LICENSE).
