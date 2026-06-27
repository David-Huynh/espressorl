## Grinder Catalog Addition

Use this template for PRs that add or update rows in
`supabase/migrations/*_grinder_catalog*.sql`.

### Grinder

- Manufacturer:
- Model:
- Aliases:
- Adjustment model: `single_axis`, `piecewise_single_axis`, or `compound_dual_axis`
- Adjustment unit: `click`, `setting`, `dial_marker`, `macro_micro_index`, or `inner_micro_notch`

### Step Scale

- Microns per click/marker/setting:
- Minimum displayed setting:
- Maximum displayed setting:
- Direction as numbers increase: `higher_is_coarser` or `higher_is_finer`
- For piecewise grinders, list each segment and whether any segment is nonlinear:
- For compound grinders, list each axis and how to combine it into one effective relative grind coordinate:

### Evidence

- Source URL(s):
- Source quality: `official`, `manufacturer_reported`, `user_measured`, or `unverified`
- Notes about variable step sizes, burr-zero offsets, aftermarket dials, or measurement method:

### Safety

- [ ] This only changes catalog metadata.
- [ ] Optimizer training still uses normalized relative grind from the active grinder context.
- [ ] Unknown values are left `NULL` instead of guessed.
- [ ] One physical grinder is not split into multiple catalog rows just because it has range tiers or micro-adjustments.
- [ ] `normalized_alias` values are unique.
