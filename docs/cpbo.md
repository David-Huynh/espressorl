# Consecutive Preferential Bayesian Optimization

EspressoRL uses Consecutive Preferential Bayesian Optimization (CPBO) as its
default recipe optimizer. It learns from one comparison after each valid
candidate shot:

- new_better
- anchor_better
- tie

Numeric taste ratings are not converted into CPBO observations. Machine
failures and aborted shots are stored as physical attempts but never become
preference labels.

The implementation follows the fixed-anchor maximum-value entropy-search design
from [Consecutive Preferential Bayesian Optimization](https://arxiv.org/abs/2511.05163)
and the q=1 local-search behavior from
[Local Preferential Bayesian Optimization / TuRPBO](https://arxiv.org/abs/2606.02351).
Physics features are soft kernel coordinates, not synthetic objective values.

## Architecture

The implementation follows the repository's ports-and-adapters boundary:

- domain/cpbo.py contains immutable recipe, physical-shot, comparison, run,
  state, and diagnostic models.
- ports/preference_optimization.py defines optimizer and repository ports.
- application/preference_optimization.py owns the optimization use cases.
- application/cpbo_runtime.py translates canonical ShotRecord instances into
  the CPBO application API.
- optimizers/cpbo*.py contain machine-independent inference and acquisition.
- SQLite, Postgres, and Gaggimate MQTT remain adapters.

No optimizer module imports MQTT, Gaggimate, Supabase, UI, or database drivers.

## Recipe Space

The latent utility is:

\[
f(x) \sim \mathcal{GP}(0, k(x,x')), \qquad
x = [\text{grind},\text{dose},\text{target output}]
\]

Brew ratio is always derived:

\[
\text{ratio}(x)=\frac{\text{target output}}{\text{dose}}
\]

Each run freezes its physical bounds, resolution, units, normalization, grind
direction, ratio constraints, and effective configuration fingerprint. Stored
recipes are never rewritten after configuration changes. A changed effective
configuration archives the old active run and starts a new run for that
context.

Internally, all controls are normalized to [0,1]^3. Grind is represented as
fineness, where a larger value always means finer. Candidate controls are
converted back to physical units and quantized before identity, deduplication,
or display.

The commanded recipe dose is stored separately as `dose_target_g` from the
measured physical dose in `dose_in_g`. A grind-by-weight measurement sets
`dose_observed`. Without that measurement, CPBO uses the commanded dose only
after the user confirms it, represented by `dose_target_confirmed`; it never
relabels a manual confirmation as a measurement. An unconfirmed or explicitly
not-followed dose remains useful as a masked physical shot for future offline
learning, but it cannot define a complete CPBO recipe point or preference.

Hardware-scale brew-by-weight records the measured output at control cutoff in
`beverage_out_g`. Predictive stopping is stored separately as
`predicted_final_beverage_out_g` with its delay, flow-rate, and lead diagnostics.
The prediction never overwrites the observation, and neither local ingestion
nor community validation rejects a shot merely because measured cutoff output
differs from the configured or predicted target.

Run context includes install, machine, bean, grinder, profile identity, basket,
water, user identifiers, and the selected taste goal when available. A stable
profile ID is authoritative when present. A profile hash is a fallback only
when no stable ID is available, and adapters using that fallback must provide a
stable configuration hash rather than an observed shot-trajectory hash.
Missing identifiers are not fabricated. Materially different contexts are not
mixed.

### Taste goal context

A taste goal is either `balanced` or a versioned `custom` set of categorical
targets. Custom targets use `low`, `medium`, or `high` for the shared flavor
vocabulary (for example `sweet`, `nutty_cocoa`, `fruity`, `bitter`, or
`astringent_harsh`). Balanced means that no explicit attribute target is set;
it is not a numeric target or an inferred reward.

The goal is part of the run fingerprint. The same bean, grinder, and profile
can therefore maintain separate resumable CPBO runs for different goals. A
comparison, its two physical shots, and the generated recommendation all carry
the immutable goal snapshot. Feedback from one goal cannot enter another
goal's preference likelihood.

CPBO does not append goal values to recipe `x`, assign target weights, or
convert them into pseudo-observations. The user evaluates which shot is closer
to the selected goal, and CPBO learns only from the resulting oriented
comparison inside that goal-scoped run. The goal snapshot is retained so a
future offline conditional model can learn relationships between requested
flavor profiles, trajectories, and preferences.

## Preference Likelihood

Orientation is always:

\[
d=f(x_{new})-f(x_{anchor})
\]

With fixed per-run perceptual noise sigma_pref and learned nonnegative JND
threshold gamma:

\[
s=\sqrt{2}\sigma_{pref}
\]

\[
P(new\ better)=\Phi((d-\gamma)/s)
\]

\[
P(tie)=\Phi((\gamma-d)/s)-\Phi((-\gamma-d)/s)
\]

\[
P(anchor\ better)=\Phi((-d-\gamma)/s)
\]

Gamma uses a softplus parameterization. The total GP output scale is fixed to
one and sigma_pref is fixed within a run, avoiding an unidentified joint scale.
At gamma=0, tie probability is zero up to floating-point roundoff and the model
reduces to binary probit preference learning.

The GP uses float64, a zero mean, ApproximateGP,
CholeskyVariationalDistribution, full-batch Adam, Monte Carlo expected log
likelihood, KL regularization, early stopping, retained best parameters,
configurable jitter, capped inducing points, and safe JSON checkpoints.

## Additive Kernel

The covariance is a fixed-scale convex mixture:

\[
k = w_{raw}k_{Matern\ 5/2}
  + w_{phys}k_{Matern\ 3/2}
  + w_{trace}k_{uncertain\ RBF}
\]

The active weights are a softmax and sum to one. The raw kernel uses ARD over
the three normalized controls. Recipe physics includes robustly standardized
proxies for dose, output, ratio, bed depth, particle size, resistance, flow,
and expected duration. Missing basket or grinder calibration uses documented
monotone fallbacks. These values are never wins, losses, or ratings.

Telemetry is reduced to a fixed trace summary. Independent exact GPs predict
each standardized trace feature from recipe controls. Candidate and observed
recipes both use surrogate predictions in the trace kernel. The trace component
stays disabled, with exactly zero weight, until enough complete finite traces
exist and validation passes.

## CPBO-MES

Every iteration proposes exactly one recipe. The acquisition is:

\[
I(f_*;R_{next}\mid D,x,a)
=H[R_{next}\mid D,x,a]
-E_{f_*}[H[R_{next}\mid f_*,D,x,a]]
\]

The unconditional entropy analytically integrates GP difference uncertainty,
candidate-anchor covariance, and perceptual noise across all three outcomes.
The maximum distribution is approximated over a quantized candidate set made
from scrambled Sobol points, evaluated recipes, local points, and valid
boundaries.

Production uses the paper-style Gumbel maximum approximation. A seeded direct
posterior-max strategy remains available only as a diagnostic reference. For
each maximum representative, the local bivariate posterior is conditioned on
both f(candidate) <= maximum and f(anchor) <= maximum. Sampling uses vectorized
rejection and a finite Gibbs fallback.

The CPBO path never calls EI, UCB, Thompson sampling, EUBO, qEUBO, or a scalar
rating acquisition.

## Offline Model Integration

Pairwise comparisons are canonical supervision that can be reused by future
offline model-based RL or preference-learning methods. Community persistence is
therefore algorithm-neutral: physical recipes and trajectories are shot
records, while `new_better`, `anchor_better`, and `tie` are oriented rows in
`community_comparisons`. Both operands and the comparison must have the same
versioned taste-goal snapshot.

An offline dynamics model can train self-supervised on physical trajectories,
actions, and context. A separate preference or terminal-utility model can join
the candidate and anchor trajectory embeddings through the comparison table.
The labels must not be converted to fabricated scalar rewards, and optimizer
implementation details must not become persistence columns.

## Anchor Modes

global_previous compares against the immediately previous valid physical shot
and searches the full feasible domain. A loss still makes the new valid shot
the next anchor.

best_incumbent compares against the directly established incumbent and searches
a q=1 lengthscale-shaped trust region. Only new_better replaces the incumbent.
A tie remains a JND observation but counts as a trust-region failure.

| Setting | Value |
| --- | ---: |
| Initial length | 0.8 |
| Minimum length | 0.5^7 |
| Maximum length | 1.6 |
| Successes before expansion | 3 |
| Non-wins before contraction | 4 |

When the region falls below its minimum, one full-domain CPBO-MES proposal is
made against the unchanged incumbent. An untested restart candidate is never
declared best.

## Persistence Migration

Startup creates these non-destructive SQLite/Postgres tables:

- cpbo_runs
- cpbo_recipes
- cpbo_shots
- cpbo_comparisons
- cpbo_states
- cpbo_suggestions

Recommendation storage also adds:

- optimization_run_id
- comparison_anchor_shot_id
- comparison_mode
- preference_feedback_required
- taste_goal_json
- taste_goal_fingerprint

Pre-release numeric BO records and scalar feedback columns are removed during
schema migration. CPBO data are separate physical shots and oriented
comparisons, not synthetic scalar observations. Reset All deletes CPBO rows
for that install and machine through the repository port.

Pre-goal local records migrate explicitly to the balanced goal. Active legacy
CPBO run lookup keys are rewritten lazily after successful context validation;
historical recipe and shot values are not mutated.

Community persistence is separate from local optimizer checkpoints:

- `community_validated_shots` stores sanitized physical observations.
- `community_recommendations` stores proposal lifecycle records.
- `community_comparisons` stores algorithm-neutral oriented feedback.

There is no optimizer-owned community table. Offline dataset builders join
physical shot trajectories to comparison rows and version their own artifacts.

The signed Supabase queue uses the matching `shot_record`,
`recommendation_record`, and `comparison_record` event types. Community shot
records do not contain numeric taste ratings or derived scalar rewards.

## Configuration

CPBO is selected with:

~~~json
{
  "optimizer_mode": "cpbo",
  "cpbo": {
    "profile_name": "application",
    "comparison_mode": "best_incumbent",
    "random_seed": 17,
    "recipe_domain": {
      "grind_radius_steps": 10.0,
      "dose_min_g": 6.0,
      "dose_max_g": 30.0,
      "target_output_min_g": 5.0,
      "target_output_max_g": 250.0
    }
  }
}
~~~

Application defaults use sigma_pref=0.20, initial_gamma=0.20, 300 GP fit steps,
192 Sobol candidates, 10 maximum bins, 256 truncated samples per bin, an 80/20
raw-to-physics initial kernel mixture, and trace activation after eight valid
telemetry shots. These values prioritize bounded container latency and are
application assumptions, not claimed espresso psychophysics.

The application trust region starts at normalized length 0.1 so the wider
physical recipe domain still begins with local exploration. The
paper-fidelity profile retains the reference 0.8 initial length.

The paper_fidelity profile defaults to global-previous mode, 2,000 GP steps,
25,000 Gumbel samples, 20 bins, and 1,000 truncated samples per bin. It is
intended for reproduction and offline analysis, not low-latency operation.
The runtime comparison policy is independent and may override that profile
default.

Gaggimate publishes the selected `cpbo_profile_name` and
`cpbo_comparison_mode` in its `optimizer_settings` event. Both fields are
strict enums. The display defaults to `application` and `best_incumbent`.
Changing either field clears an outstanding, unconsumed recommendation and
forces a model refit, but it does not change the bean/grinder context or delete
physical shots and pairwise comparisons. When the physical recipe domain is
unchanged, the active run retains compatible evidence across configuration
changes, including comparisons collected with the other anchor policy. Each
stored comparison retains the policy under which its anchor was selected.
Switching away from a bean/grinder context and later returning to the same
context resumes its active run and current recommendation. A different stable
recipe-profile identity or taste goal is a different optimization context and
retains its own independent run. Normal shot-to-shot trajectory variation under
the same profile ID does not create a new run.

The recipe domain is CPBO's configurable physical search space, not a set of
machine-safety limits. The default relative grind domain is the baseline plus
or minus 10 grinder steps, dose spans 6-30 g, and target output spans 5-250 g.
Stepped resolution defaults to 1.0 and stepless resolution to 0.1. Brew ratio
is derived from target output divided by dose and is not an independent search
coordinate or feasibility bound. All nested fields are strictly allowlisted;
unknown configuration keys fail startup.

Advanced domains remain inside a broad data-integrity envelope: grind radius
0.1-1,000 steps, dose 0.1-100 g, and target output 0.1-1,000 g. These limits
reject malformed or abusive inputs at trust boundaries; they are not the
optimizer's active search region or machine-safety claims. Upload adapters
validate any supplied ratio against output divided by dose instead of imposing
a separate ratio range.

Each run snapshots and fingerprints its recipe domain. When the domain changes,
an active run with physical-shot evidence is migrated in place. Physical
recipes, shot IDs, comparisons, and the incumbent are preserved; observation
coordinates are recomputed against the new domain and may fall outside
`[0, 1]`. Model checkpoints and an unbrewed suggestion are invalidated, the
trust-region state is rebuilt from comparison history, and a new bounded
suggestion is fitted immediately. Resolved suggestion rows remain available as
audit history. If a shot is still awaiting preference feedback, migration waits
for that answer instead of discarding the comparison.

An active run with no physical shots may be retired and initialized again from
the next canonical machine recipe. Canonical events and offline datasets retain
physical units; normalized coordinates are derived only at the CPBO model
boundary.

### Historical recipe corrections

The active recipe domain constrains executable recommendations, not historical
observations. A shot-history correction is accepted when its physical values
pass the broad integrity envelope, even when the corrected recipe is outside
the run's frozen search space. The canonical shot and any enabled community
replacement upload are persisted before CPBO reprocessing.

CPBO stores the corrected physical values as an `ObservedRecipe` and creates a
quantized observation `RecipePoint` without clipping it to the run's bounds.
Its normalized coordinates may therefore be below zero or above one. Every
physically valid shot and its pairwise comparison remains active GP evidence,
including observations outside the frozen search space.

The distinction is enforced at acquisition time. Candidate generation, MES
maximum support, and emitted recommendations stay inside the frozen normalized
domain. An out-of-space observation may remain the comparison anchor or
incumbent, but a reconstructed trust-region center is projected to the nearest
domain boundary; the stored observation itself is never projected or changed.
An unanswered comparison remains pending after either shot is corrected.

Rebuilding after a recipe correction clears model checkpoints and recomputes
the previous shot, incumbent, and trust-region state from the complete valid
history. A stale unbrewed suggestion is replaced with a newly fitted bounded
recommendation. No additional comparison is requested for comparisons already
recorded.

Explicitly excluding a historical shot is different from deleting it. The
canonical shot remains stored for audit, community processing, and offline
training, while its CPBO physical-shot projection is marked `excluded`.
Comparisons that depend on the excluded shot are removed, the remaining valid
comparison chain and optimizer state are rebuilt transactionally, and any
unanswered suggestion is superseded. The next eligible physical shot can then
become the pending candidate against the rebuilt previous shot or incumbent.
Repeated exclusion events are idempotent and do not emit duplicate
recommendations.

When an accepted or idempotently replayed shot becomes the pending candidate,
the application result carries a typed `PendingPreferenceRequest`. It contains
the run, candidate, anchor, comparison mode, and snapshotted taste goal. The
Gaggimate MQTT adapter may include that canonical request in its shot-delivery
acknowledgement, allowing a replayed shot to open the required comparison
without inventing recommendation attribution. MQTT JSON and firmware prompt
state do not enter CPBO or the optimizer.

Changing the configured recipe domain uses the same observation rule for every
shot in the active run: physical values are unchanged, recipe IDs and
normalized coordinates are regenerated for the new domain, and all existing
comparisons remain optimizer evidence. Acquisition and emitted recommendations
remain bounded to the new search space.

## Operational Loop

1. A first valid baseline shot initializes previous and incumbent state.
2. CPBO emits one quantized candidate and its anchor.
3. The candidate is pulled and stored as a distinct physical shot.
4. Invalid shots clear the candidate without creating a comparison.
5. A valid shot asks which result is closer to the run's snapshotted taste goal
   using exactly one three-outcome preference.
6. The oriented comparison updates the GP and, in best mode, trust-region state.
7. CPBO emits the next candidate.

The API shape is:

~~~python
request = optimizer.initialize(
    context,
    baseline_recipe,
    comparison_mode=ComparisonMode.BEST_INCUMBENT,
)
baseline = optimizer.record_shot(
    request.optimization_run_id,
    request.recipe,
    status="valid",
)

suggestion = optimizer.suggest_next(request.optimization_run_id)
candidate = optimizer.record_shot(
    request.optimization_run_id,
    suggestion.recipe,
    status="valid",
)

# Choose exactly one: NEW_BETTER, ANCHOR_BETTER, or TIE.
label = PreferenceLabel.NEW_BETTER
state = optimizer.record_preference(
    request.optimization_run_id,
    candidate.shot_id,
    suggestion.anchor_shot_id,
    label,
)
~~~

## Verification

~~~powershell
C:\Users\David\.local\bin\uv.exe run python -m unittest discover -s tests
C:\Users\David\.local\bin\uv.exe run python -m compileall -q src tests
~~~

The paper-fidelity acquisition can be expensive because every candidate and
maximum bin requires truncated bivariate samples. The application profile
chunks candidates but still scales with candidate count, maximum bins, and
samples per bin. The first implementation also uses independent exact trace
GPs, so very large telemetry datasets will eventually need a sparse surrogate.
