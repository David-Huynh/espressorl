# EspressoRL

## Summary

EspressoRL is a machine-agnostic espresso dial-in service. The Gaggimate
adapter receives completed shots over MQTT, stores physical shot history, and
returns one bounded recipe recommendation at a time.

The optimizer is Consecutive Preferential Bayesian Optimization (CPBO). Its
only subjective observation is an oriented comparison:

- new shot is better
- anchor shot is better
- no noticeable difference

Numeric ratings, derived taste rewards, and fabricated physics observations
are not part of the optimizer or community-data contracts.

## Purpose

EspressoRL keeps recipe experiments isolated by bean, grinder, profile,
basket, machine, user, and taste-goal context. It distinguishes recipe settings from
physical shots, retains repeated shots at the same recipe, and never records a
machine failure as a taste tie.

The code follows ports and adapters:

- `domain`: canonical espresso and CPBO models
- `application`: shot, recommendation, preference, upload, and admin use cases
- `ports`: interfaces required by the core
- `adapters`: MQTT, SQLite, Postgres, Supabase, dashboard, and file adapters
- `optimizers`: machine-independent CPBO inference and acquisition
- `supabase`: signed public ingress, grinder search, and queue migrations

## Installing

Create `data/options.json` from the example and set the MQTT and database
connection:

```bash
mkdir -p data
cp data/options.example.json data/options.json
```

Minimum public-container configuration:

```json
{
  "mqtt_host": "192.168.1.85",
  "machine_id": "gaggimate:YOUR_TOPIC_ID",
  "optimizer_mode": "cpbo",
  "storage_backend": "postgres",
  "postgres_dsn": "postgresql://espresso_rl:espresso_rl@postgres:5432/espresso_rl"
}
```

Start the service:

```bash
docker compose up -d --build --remove-orphans
docker compose logs -f espresso-rl
```

The Compose stack does not include an MQTT broker. Gaggimate and EspressoRL
must use the same reachable broker.

## Usage

Gaggimate publishes:

```text
gaggimate/{topic_id}/shot/profile
gaggimate/{topic_id}/machine/state
gaggimate/{topic_id}/rl/preference
gaggimate/{topic_id}/rl/shot/correction
gaggimate/{topic_id}/rl/recommendation/decision
gaggimate/{topic_id}/rl/recommendation/apply
```

EspressoRL publishes retained state on:

```text
gaggimate/{topic_id}/rl/recommendation
gaggimate/{topic_id}/rl/status
```

The first valid shot establishes a baseline. CPBO then proposes exactly one
quantized recipe and identifies its comparison anchor. After the candidate is
pulled, the user supplies one of the three preference outcomes and CPBO emits
the next candidate. `best_incumbent` and `global_previous` comparison modes are
configured under `cpbo.comparison_mode`.

Taste goals are selected on Gaggimate as balanced or categorical custom
targets. Changing the goal creates or resumes a separate run for the same
bean/grinder/profile context. CPBO still consumes only pairwise preferences;
the goal is retained as context for comparison integrity and future offline
conditional models.

Apply acknowledgement records only what the adapter accepted. Actual next-shot
data determine whether grind, dose, and output were followed.

## Community Data

Community upload is optional and best-effort. Shot trajectories and pairwise
comparisons are separate signed records. Local optimization does not wait for
Supabase, and upload failure does not delete local evidence.

Supabase is a short-lived ingress queue. An admin deployment leases rows,
mirrors them into its Postgres warehouse, validates them, and routinely removes
completed source rows. The admin Postgres database is the durable source for
offline exports.

Export the optimizer-neutral preference dataset with:

```bash
uv run espresso-rl-export-offline-dataset \
  --postgres-dsn "$ESPRESSORL_POSTGRES_DSN" \
  --output-dir offline_dataset
```

The export is deterministic UTF-8 JSONL plus a SHA-256 manifest. It contains no
executable or opaque model files.

## Documentation

- [CPBO mathematics and persistence](docs/cpbo.md)
- [Operations and admin retention](docs/operations.md)
- [Offline dataset format](docs/offline-dataset.md)

## Development

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
```

On this Windows installation, `uv` is available at
`C:\Users\David\.local\bin\uv.exe`.

## License

EspressoRL is licensed under the GNU Affero General Public License v3.0 or
later. See [LICENSE](LICENSE).
