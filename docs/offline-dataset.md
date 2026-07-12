# Offline Preference Dataset

The admin Postgres warehouse is EspressoRL's durable community-data store.
The offline exporter joins every trusted oriented comparison to its two
immutable physical shot records. It is intentionally independent of CPBO so a
future offline learner can consume the same evidence.

## Files

```text
preference_examples.jsonl
manifest.json
README.txt
```

`preference_examples.jsonl` is canonical UTF-8 JSON Lines. Each row contains:

- comparison identity, orientation, label, and comparison mode
- shared machine, bean, grinder, profile, basket, water, and user context
- new-shot recipe, realized outcome, quality metadata, and trajectories
- anchor-shot recipe, realized outcome, quality metadata, and trajectories
- source trust weights

The label is exactly `new_better`, `anchor_better`, or `tie`. The exporter
rejects numeric ratings, scalar rewards, reversed joins, mixed contexts,
future timestamps, duplicate comparison identities, and non-finite values.

## Export

```bash
uv run espresso-rl-export-offline-dataset \
  --postgres-dsn "$ESPRESSORL_POSTGRES_DSN" \
  --output-dir offline_dataset
```

The admin dashboard also exposes authenticated manifest and JSONL download
endpoints. The manifest records row count, format version, exporter version,
generation time, byte length, and SHA-256 digest.

JSONL and JSON are data-only formats. The export contains no pickle, Python
object graph, macros, database dump, model checkpoint, or executable payload.
