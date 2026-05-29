# EspressoRL Home Assistant Add-on

This add-on hosts the local EspressoRL service. The active runtime path follows
the ports-and-adapters layout described in `EspressoRL_DESIGN.md`:

```text
Gaggimate MQTT adapter -> canonical events -> application service
application service -> repository/optimizer ports -> domain logic
SQLite repositories <- repository ports
Conservative BO optimizer -> domain models/safety only
```

The core packages are:

- `espresso_rl.domain`: canonical events, shot/recommendation models, profile
  resampling, grind normalization, reward, safety, and follow-through logic.
- `espresso_rl.application`: use cases for shot ingestion, feedback,
  recommendation decisions, reward recomputation, and next recommendation
  generation.
- `espresso_rl.ports`: repository and optimizer interfaces.
- `espresso_rl.optimizers`: machine-agnostic conservative BO implementation.
- `espresso_rl.adapters`: SQLite persistence and Gaggimate MQTT translation.

The old rushed active agent/replay-buffer/MQTT path has been removed. DreamerV3
modules remain present but are not wired into the active recommendation path
until they can pass through the same safety, recommendation-memory, and
follow-through gates.

Implemented backend behavior includes canonical shot, feedback, decision, and
machine-state events; recommendation memory; wake/idle recommendation display;
stale recommendation expiry; actual-shot follow-through inference; low-confidence
profile-only rewards; and an opt-in signed upload queue for a Supabase Edge
Function or compatible ingestion endpoint.

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

Recommendation flow:

```text
Gaggimate publishes shot/profile
  -> add-on stores shot and generates bounded BO recommendation
  -> add-on publishes gaggimate/{mac}/rl/recommendation
  -> Gaggimate stores the pending recommendation
  -> LVGL and WebUI show the recommendation/rating prompts
```

The recommendation payload includes grind delta, next dose target, target yield,
ratio, mode, confidence, reason, and IDs. Grind is recommendation-only because
Gaggimate cannot automate a grinder setting. When the user chooses Use, firmware
saves the recommended target grind dose and selected-profile target yield, then
publishes an accepted recommendation decision. It does not auto-apply on MQTT
receipt. Choosing Later sends no decision, so the retained pending
recommendation can be shown again on the next wake/reconnect. Choosing Ignore
sends an ignored decision so the optimizer will not count it as followed.

If EspressoRL Auto Tuning is disabled, Gaggimate still uses the Home
Assistant/MQTT plugin normally, but it does not publish EspressoRL shot profiles,
listen for BO recommendations, or send ratings.

Optional Supabase upload is controlled by:

- `community_upload_enabled`
- `supabase_ingest_url`
- `upload_secret`
- `upload_token_id`

If upload is enabled without an ingest URL or secret, records are queued locally
but not sent. Local recommendations and SQLite data collection continue either
way.

Run local verification with:

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src tests
```
