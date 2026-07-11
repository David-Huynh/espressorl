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

Run context includes install, machine, bean, grinder, profile ID, raw profile
hash, basket, water, and user identifiers when available. Missing identifiers
are not fabricated. Materially different contexts are not mixed.

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

## DreamerV3 Integration

CPBO comparisons are canonical shot feedback, so a future DreamerV3 taste
model should consume the same oriented `new_better`, `anchor_better`, and `tie`
records. The labels must not be converted to `+1`, `0`, and `-1` rewards.

Dreamer's RSSM world model remains self-supervised on physical trajectories,
actions, and context. Preference feedback belongs in a separate terminal
utility head over the candidate and anchor trajectory embeddings, conditioned
on recipe context and the selected taste objective. That head should use the
same three-outcome JND likelihood and learn a nonnegative perceptual threshold.
The actor can then optimize predicted terminal utility during imagined
rollouts without teaching the dynamics model that a tie is zero reward.

For one-shot recipe recommendations, unavailable or rejected Dreamer inference
hands control to stateful CPBO. For Dreamer's live adaptive profile, a control
safety failure falls back to the configured static machine profile; CPBO does
not issue mid-shot control actions. Numeric historical ratings remain separate
legacy supervision and are not silently transformed into pairwise preferences.

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

Existing numeric BO records remain readable. CPBO data are separate physical
shots and oriented comparisons, not synthetic scalar observations. Reset All
deletes CPBO rows for that install and machine through the repository port.

## Configuration

CPBO is selected with:

~~~json
{
  "optimizer_mode": "cpbo",
  "cpbo": {
    "profile_name": "application",
    "comparison_mode": "best_incumbent",
    "random_seed": 17
  }
}
~~~

Application defaults use sigma_pref=0.20, initial_gamma=0.20, 300 GP fit steps,
192 Sobol candidates, 10 maximum bins, 256 truncated samples per bin, an 80/20
raw-to-physics initial kernel mixture, and trace activation after eight valid
telemetry shots. These values prioritize bounded container latency and are
application assumptions, not claimed espresso psychophysics.

profile_name set to paper_fidelity uses global-previous mode, 2,000 GP steps,
25,000 Gumbel samples, 20 bins, and 1,000 truncated samples per bin. It is
intended for reproduction and offline analysis, not low-latency operation.

The default relative grind domain is the baseline plus or minus 10 grinder
steps. Stepped resolution defaults to 1.0 and stepless resolution to 0.1.
Dose/output limits come from global machine safety bounds. All nested fields
are strictly allowlisted; unknown configuration keys fail startup.

## Operational Loop

1. A first valid baseline shot initializes previous and incumbent state.
2. CPBO emits one quantized candidate and its anchor.
3. The candidate is pulled and stored as a distinct physical shot.
4. Invalid shots clear the candidate without creating a comparison.
5. A valid shot asks for exactly one three-outcome preference.
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
