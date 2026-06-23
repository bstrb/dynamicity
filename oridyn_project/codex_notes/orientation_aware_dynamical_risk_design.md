# Physics-Grounded Orientation-Aware Dynamical-Risk Score For cSerialED

Status: research/design note, not a production correction model.

This note proposes a next-generation OriDyn diagnostic score for orientation-dependent dynamical electron diffraction risk in cSerialED. It is intentionally not an extension of the current toy enhancement-feed score. The toy score was useful because it exposed orientation-correlated behavior, but a physically grounded model should start from coupled beam amplitudes, excitation error, reciprocal-space coupling by Fourier potential components, and zone-axis beam manifolds.

## Scientific Aim

For each indexed frame and each observed reflection `g`, estimate whether orientation-dependent dynamical coupling is likely to enhance or deplete the measured intensity of `g`, and estimate the magnitude of that risk.

The score should be useful for:

- Diagnosing orientation-dependent intensity shifts.
- Comparing high-risk and low-risk observations within the same signed HKL.
- Testing whether dynamical-risk metrics predict partialator unmerged intensity shifts.
- Eventually informing optional filtering, weighting, or correction experiments, but not before validation.

## Physical Background

In electron diffraction, the incident beam and diffracted beams are coupled through the crystal electrostatic potential. In a Bloch-wave or Darwin-Howie-Whelan picture, the amplitude of beam `g` is not independent. It evolves with thickness due to coupling from other beams `h` through the Fourier potential component `U_(g-h)`.

A schematic coupled-amplitude equation is:

```text
d psi_g(z) / dz ~= i * pi * lambda * sum_h U_(g-h) psi_h(z) exp(2 pi i s_g z)
```

where:

- `psi_g(z)` is the complex amplitude of beam `g` at thickness `z`.
- `lambda` is the electron wavelength.
- `U_(g-h)` is the Fourier component of the crystal potential connecting beams `h -> g`.
- `s_g` is the excitation error of beam `g`.
- The exact prefactors depend on the chosen electron diffraction convention.

The important point for a diagnostic score is not the exact prefactor, but the structure:

```text
beam h transfers amplitude to beam g through U_(g-h), modulated by excitation and thickness.
```

The current toy feed score approximates only one part of this: build-up paths into a target. A better OriDyn diagnostic should represent both incoming and outgoing coupling and distinguish signed predicted shift from risk magnitude.

## Reciprocal-Space Beam-Coupling Graph

For each indexed frame `f`, define a directed graph:

```text
G_f = (V_f, E_f)
```

### Nodes

Each node is a reciprocal-lattice beam `n = (h,k,l)` near enough to excitation to participate in dynamical coupling.

Given frame orientation `R_f` and reciprocal basis `B`, the laboratory reciprocal vector is:

```text
G_f(n) = R_f B n
```

Candidate node set:

```text
V_f = { n : d_min <= d(n) <= d_max and |s_f(n)| <= s_node_max }
```

where `s_f(n)` is the excitation error in that frame. `s_node_max` may be wider than the plotted/observed excitation cutoff because off-Ewald beams can still contribute weakly in a path expansion.

Observed reflections are a subset:

```text
O_f subset V_f
```

The graph may include unobserved predicted beams, because unobserved beams can still mediate dynamical transfer.

### Edges

An edge `a -> b` represents possible amplitude transfer from beam `a` to beam `b` through:

```text
Delta = b - a
U_Delta = Fourier potential component for Delta
```

The physically meaningful object is a complex amplitude coupling. A diagnostic graph score will usually use a real nonnegative proxy for coupling capacity:

```text
C_f(a -> b) ~= E_f(a) E_f(b) |U_(b-a)|^2 L(b-a) Z_f(a,b) T_f(a,b)
```

where:

- `E_f(a)` and `E_f(b)` are excitation weights for source and destination beams.
- `|U_(b-a)|^2` is the approximate scattering strength for the transfer vector.
- `L(b-a)` is a low-order or small-angle preference.
- `Z_f(a,b)` is a zone-manifold consistency factor.
- `T_f(a,b)` is an optional thickness/order penalty or path-length regularizer.

This is not yet a full Bloch-wave solution. It is a graph-based probability proxy inspired by the Born/path expansion of coupled amplitudes.

## Excitation Weight

A simple excitation weight can be:

```text
E_f(n) = exp(-0.5 * (s_f(n) / s0)^2)
```

or a Lorentzian-like alternative:

```text
E_f(n) = 1 / (1 + (s_f(n) / s0)^2)
```

The Gaussian version is familiar from the existing toy score and is numerically stable. The Lorentzian version may be more forgiving for thickness-integrated excitation.

The sign of `s_f(n)` may also matter. A first graph-risk model can use `|s|`; a more physical signed-shift model may need phase-like terms or separate treatment of beams above and below the Ewald surface.

## Scattering Strength Term

The edge strength should include an estimate of:

```text
|U_Delta|^2
```

Possible levels of approximation:

1. Orientation/cell-only proxy:

```text
|U_Delta|^2 proxy ~= exp(-( |G(Delta)| / G0 )^2)
```

This captures small-angle Coulomb scattering preference but ignores structure.

2. Merged-intensity proxy:

```text
|U_Delta|^2 proxy ~= I_merged(Delta)
```

or:

```text
|U_Delta| proxy ~= sqrt(max(I_merged(Delta), 0))
```

This uses empirical scattering strength but mixes dynamical effects into the proxy.

3. Model/Fcalc proxy:

```text
|U_Delta|^2 proxy ~= |F_calc(Delta)|^2 * electron_form_factor_scale(Delta)
```

This is more physical if a model is available and if electron scattering factors are used.

4. Atomic electron scattering factor approximation:

```text
U_Delta ~= sum_j f_j^e(|G_Delta|) exp(2 pi i Delta dot x_j)
```

This is closest to the intended physics but requires coordinates, atom types, occupancies, B factors, and electron scattering factors.

For early diagnostics, the recommended path is:

- Start with cell/orientation-only `L(Delta)`.
- Add a second mode using merged intensities or `Fcalc`.
- Compare whether the physics-informed term improves validation against unmerged diagnostics.

## Small-Angle / Low-Order Preference

Electron scattering is strongly forward-peaked. A practical low-order factor can be:

```text
L(Delta) = exp(-( |G(Delta)| / G0 )^2)
```

or:

```text
L(Delta) = 1 / (|G(Delta)|^2 + g_epsilon^2)^p
```

The exponential is safer numerically. The Coulomb-like inverse-power form is more physical but needs careful regularization near the origin and should never include `Delta = 000`.

Important distinction:

- `L(Delta)` is a transfer-vector property.
- `E_f(a)` and `E_f(b)` are frame/orientation properties.
- `|U_Delta|^2` is a structure/scattering property.

These should stay separated in the implementation so the model can be audited.

## Zone-Axis And Laue-Zone Manifold Factors

For a frame with incident beam direction approximately along a crystal direction `[u v w]`, Laue-zone membership can be approximated by the integer:

```text
zone(n) = h u + k v + l w
```

For arbitrary indexed frames, it may be better to classify zones by the component of `G_f(n)` along the beam direction:

```text
zeta_f(n) = dot(G_f(n), beam_direction)
```

and then cluster or bin beams into ZOLZ/FOLZ/SOLZ-like manifolds.

The edge zone factor can encode the idea that strongly coupled beams often live in the same low-order manifold:

```text
Z_f(a,b) =
    1.0                         if zone(a) == zone(b)
    alpha_cross_zone            if one beam is ZOLZ and the other is FOLZ/SOLZ
    alpha_far_zone              otherwise
```

Alternative continuous form:

```text
Z_f(a,b) = exp(-0.5 * ((zeta_f(a) - zeta_f(b)) / zeta0)^2)
```

The zone factor should be diagnostic, not dogmatic. Some dynamical transfer can occur across zones, especially near zone axes or via multi-step paths. The useful output is not only the total risk, but the fraction of risk carried by ZOLZ and same-zone paths.

## Optional Thickness / Order Penalty

A one-edge graph captures first-order pairwise coupling. A Born/path expansion can include paths:

```text
a0 -> a1 -> ... -> am
```

with path contribution roughly:

```text
PathWeight ~= product_i C_f(a_i -> a_(i+1)) * P_m
```

where an order penalty might be:

```text
P_m = tau^m / m!
```

or:

```text
P_m = exp(-m / m0)
```

Here `tau` is a proxy for thickness or effective interaction strength. If thickness is unknown, path order should be limited and reported as a sensitivity parameter rather than interpreted literally.

The first implementation should probably compute only one-step incoming/outgoing graph flux, plus optionally two-step path summaries as a separate diagnostic. This avoids silently recreating a complicated toy path score.

## Per-Observation Scores

For an observed target reflection `g in O_f`, define incoming flux:

```text
incoming_flux_f(g) = sum_{a in V_f, a != g} C_f(a -> g)
```

Define outgoing flux:

```text
outgoing_flux_f(g) = sum_{b in V_f, b != g} C_f(g -> b)
```

Then define signed and magnitude scores:

```text
signed_flux_balance_f(g) = incoming_flux_f(g) - outgoing_flux_f(g)
```

```text
abs_flux_risk_f(g) = incoming_flux_f(g) + outgoing_flux_f(g)
```

Interpretation:

- `incoming_flux > outgoing_flux` is an enhancement-like prediction.
- `outgoing_flux > incoming_flux` is a depletion-like prediction.
- `abs_flux_risk` is the magnitude of dynamical coupling risk, regardless of sign.

This distinction is essential. A reflection can be highly dynamical but have near-zero signed balance because incoming and outgoing couplings are both strong.

## Zone Fractions

For a target `g`, define same-zone incoming/outgoing subsets:

```text
same_zone_in_f(g) = sum_{a: zone(a) == zone(g)} C_f(a -> g)
same_zone_out_f(g) = sum_{b: zone(b) == zone(g)} C_f(g -> b)
```

Then:

```text
same_zone_flux_fraction_f(g) =
    (same_zone_in_f(g) + same_zone_out_f(g)) / max(abs_flux_risk_f(g), epsilon)
```

Define ZOLZ-associated flux:

```text
zolz_flux_f(g) =
    sum_{a: zone(a) == 0 or zone(g) == 0} C_f(a -> g)
  + sum_{b: zone(b) == 0 or zone(g) == 0} C_f(g -> b)
```

and:

```text
ZOLZ_flux_fraction_f(g) = zolz_flux_f(g) / max(abs_flux_risk_f(g), epsilon)
```

Possible output columns:

- `dyn_incoming_flux`
- `dyn_outgoing_flux`
- `dyn_signed_flux_balance`
- `dyn_abs_flux_risk`
- `dyn_zolz_flux_fraction`
- `dyn_same_zone_flux_fraction`
- `dyn_n_nodes`
- `dyn_n_edges_in`
- `dyn_n_edges_out`

## Physical Amplitudes Versus Probability Proxies

The graph score above is not a complex amplitude simulation unless phases and propagation through thickness are explicitly modeled.

Physical amplitude model:

- Uses complex `U_Delta`.
- Tracks phases.
- Evolves amplitudes through thickness.
- Can predict interference, enhancement, depletion, and Pendellosung-like behavior.
- Requires thickness, orientation, structure factors, and numerical propagation.

Probability/capacity proxy:

- Uses nonnegative edge weights.
- Can estimate where coupling is plausible.
- Can separate incoming and outgoing coupling tendency.
- Cannot predict coherent cancellation or exact intensity transfer.
- Is easier to validate statistically across many observations.

The first OriDyn dynamical-risk graph should be described as a proxy unless and until complex propagation is implemented.

## Enhancement Versus Depletion

Enhancement and depletion should not be collapsed too early.

For an observed reflection `g`:

- Enhancement-like signal: high `incoming_flux(g)` and positive `signed_flux_balance(g)`.
- Depletion-like signal: high `outgoing_flux(g)` and negative `signed_flux_balance(g)`.
- General dynamical risk: high `abs_flux_risk(g)`.

A correction experiment would need to decide whether to:

- Downweight all high `abs_flux_risk` observations.
- Split high positive balance from high negative balance.
- Test signed intensity shifts within each signed HKL before applying any correction.

The correction direction must be learned or validated. It should not be assumed from the graph score alone.

## What Can Be Computed From Orientation And Unit Cell Only

With only stream orientation/cell and electron wavelength, we can compute:

- Candidate HKLs within `d_min/d_max`.
- Laboratory reciprocal vectors `G_f(n)`.
- Excitation error `s_f(n)`.
- Excitation weights `E_f(n)`.
- Small-angle transfer proxy `L(b-a)`.
- Laue-zone or zone-manifold labels.
- Incoming/outgoing graph topology.
- Orientation-only flux proxies.

This is enough for a first diagnostic score:

```text
C_f(a -> b) = E_f(a) E_f(b) L(b-a) Z_f(a,b)
```

It is not enough to claim structure-specific scattering strength.

## What Needs Structure Factors, Intensities, Or Fcalc

To approximate `|U_Delta|^2`, we need one of:

- Merged intensities by signed or symmetry-merged HKL.
- Model `Fcalc`.
- Atomic model plus electron scattering factors.
- External scattering-factor table.

Possible added edge model:

```text
C_f(a -> b) = E_f(a) E_f(b) |F_proxy(b-a)|^2 L(b-a) Z_f(a,b)
```

Risks:

- Merged intensities may already contain dynamical bias.
- Symmetry merging may erase signed-HKL orientation specificity.
- `Fcalc` quality varies with refinement state.
- Weak `Fcalc` can still receive dynamical intensity via multi-beam paths, so zeroing weak `U_Delta` too aggressively may hide real effects.

Recommendation: keep orientation-only and structure-weighted outputs side by side during validation.

## Minimal Implementation Plan

### Phase 1: Frame-Level Graph Diagnostic

Create a standalone diagnostic tool, not a correction tool.

Inputs:

- CrystFEL stream.
- Existing `reflection_scores.csv` or `reflection_scores_with_enh_feed.csv`.
- `d_min`, `d_max`, excitation cutoff, wavelength.
- Optional merged intensity or Fcalc table.

Per frame:

1. Parse orientation and cell.
2. Generate candidate `V_f`.
3. Compute `G_f(n)`, `d(n)`, `s_f(n)`, `E_f(n)`, and zone labels.
4. Build sparse incoming/outgoing edge lists only for beams with non-negligible `E_f`.
5. For each observed `g`, compute incoming/outgoing flux summaries.
6. Append graph diagnostic columns to observation rows by exact `source_filename + event + signed h,k,l`.

Keep output observation-level. Do not aggregate away orientation.

### Phase 2: Validation Against Existing Diagnostics

Use the partialator unmerged high-vs-low orientation diagnostic already developed:

1. Join graph scores to unmerged observations by exact observation key.
2. For each signed HKL, compare high-score and low-score tails.
3. For `signed_flux_balance`, test whether positive-balance tails have higher `I_unmerged * partialator_weight` than negative/low-balance tails.
4. For `abs_flux_risk`, test whether high-risk tails show larger scatter, larger partialator residuals, or worse merging consistency.
5. Stratify by resolution and local strength, as the current enhancement-feed diagnostics showed strong resolution/strength dependence.

Key validation summaries:

- Spearman correlation with `graph_crowding_norm`.
- Spearman correlation with `frame_axis_risk_norm`.
- High-vs-low tail shifts within signed HKL.
- Weak/middle/strong local-strength behavior within resolution bins.
- Whether ZOLZ and same-zone fractions explain which HKLs are affected.

### Phase 3: Sensitivity Sweep

Run small sweeps over:

- Excitation width `s0`.
- Node inclusion cutoff.
- Low-order decay `G0`.
- Zone cross-coupling penalties.
- Orientation-only versus structure-weighted `U` proxy.

Compare diagnostics, not refinements, first.

### Phase 4: Only Then Consider Corrections

If validation is consistent:

- Test filtering or weighting high `abs_flux_risk` observations.
- Test signed down/up correction only where signed balance predicts observed shifts.
- Keep correction proof-of-concept separate from score generation.
- Always compare against random matched controls at signed-HKL level.

## Assumptions

- The stream orientation matrices are accurate enough to compute excitation errors.
- Candidate beam generation covers the relevant low-order manifold.
- A nonnegative graph-capacity proxy can reveal statistical dynamical risk even without complex phases.
- Signed-HKL matching is required for orientation-specific diagnostics.
- Symmetry canonicalization can obscure orientation-specific effects and should be avoided during validation unless explicitly modeled.

## Limitations

- A graph-capacity score does not model coherent phase interference.
- Thickness is usually unknown or variable in cSerialED.
- Excitation error alone may not capture rocking-curve integration or mosaicity.
- Merged intensity proxies can be contaminated by the effect being studied.
- Zone classification may be ambiguous for arbitrary orientations.
- Multi-step paths can explode combinatorially and must be sparse or regularized.
- A positive signed balance is not automatically a safe correction direction.

## Recommended First Score Family

Start with three variants, all observation-level:

```text
C0_f(a -> b) = E_f(a) E_f(b) L(b-a)
```

```text
Czone_f(a -> b) = E_f(a) E_f(b) L(b-a) Z_f(a,b)
```

```text
Cstr_f(a -> b) = E_f(a) E_f(b) L(b-a) Z_f(a,b) |F_proxy(b-a)|^2
```

For each variant, output:

```text
incoming_flux
outgoing_flux
signed_flux_balance
abs_flux_risk
ZOLZ_flux_fraction
same_zone_flux_fraction
```

This keeps the model interpretable. If `Czone` improves validation over `C0`, zone manifolds matter. If `Cstr` improves validation over `Czone`, structure-specific coupling matters.

## Practical Guardrails

- Do not call the score an intensity correction.
- Do not mix signed predicted shift with risk magnitude.
- Do not hide the difference between physical complex amplitudes and nonnegative proxies.
- Do not tune on final refinement metrics first; validate observation-level behavior first.
- Always include random signed-HKL-matched controls.
- Always report resolution and local-strength stratification.
- Keep ZOLZ/same-zone fractions because they help diagnose whether the model is learning physics or just density/crowding.

## Short Summary

The next OriDyn dynamical-risk model should be a framewise reciprocal-space beam-coupling graph. Nodes are near-excited beams. Edges are possible amplitude-transfer channels through `U_(b-a)`. For each observed reflection, compute incoming and outgoing coupling separately, then derive signed balance and absolute risk. The first implementation can use nonnegative probability proxies, but the design should preserve a clear path to structure-weighted and eventually complex-amplitude models.

