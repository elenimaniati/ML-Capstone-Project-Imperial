# Model Card: Adaptive GP-UCB for Black-Box Optimisation

## Overview

**Name:** Adaptive GP-UCB (capstone BBO pipeline)
**Type:** Bayesian Optimisation — Gaussian Process surrogate + Upper Confidence Bound acquisition, with per-function hyperparameter validation and automated sanity checks.
**Version:** v4 (Week 10). Prior milestones: v1 (Week 1, naive baseline), v2 (Week 2–3, kernel/bound fix), v3 (Week 6, alpha added as a 4th tuning axis).

---

## Intended Use

**Suitable for:** low-dimensional (2–8 variable), continuous, expensive-to-evaluate black-box functions under a tight, serial query budget (one query per function per round) — the exact shape of this capstone's 8 functions. Appropriate wherever gradients are unavailable, the functional form is unknown, and each evaluation is too costly.

**Not suitable for:**
- High-dimensional problems (roughly >10 dimensions) without modification — random candidate sampling loses coverage density as dimensionality grows, a limitation already visible between this project's 2D and 8D functions.
- Discrete or combinatorial search spaces
- Batch or parallel query settings — the design is explicitly serial (one query, refit, repeat)
- Fully automated, unsupervised deployment

---

## Details: Strategy Evolution Across 10 Rounds

- **Weeks 1–2:** Standard GP (RBF kernel) + UCB. A manual override was used for Function 1 after its data (mostly near-zero, one meaningful outlier) made the automated pipeline untrustworthy. Week 2 diagnosed a serious, previously invisible failure: the kernel's `length_scale` was collapsing to a degenerate near-zero value for several functions, producing a flat, meaningless posterior. Fixed by bounding `length_scale_bounds`.
- **Week 3:** Replaced the single default kernel assumption with a systematic sweep (RBF, Matern ν=1.5, Matern ν=2.5) selected per function by log marginal likelihood (LML), rather than assumed.
- **Weeks 4–5:** Compared `beta=1.0` against the default `1.96` to quantify the exploration/exploitation trade-off directly rather than trust the default blindly. Introduced an alternating GP+UCB / RBF-interpolation-plus-distance hybrid strategy for Function 1, inspired by ensemble Bayesian optimisation literature (Wu et al., NeurIPS 2020 BBO Challenge).
- **Week 6:** Added `alpha` (the assumed noise level) as a fourth tuning axis, after finding Function 4's default value was producing an unsupported extrapolation. A later genuine duplicate measurement for Function 6 (identical input, different output) gave a direct empirical noise estimate that closely matched the independently LML-selected value — real cross-validation of the method.
- **Weeks 7–9:** Continued applying the validated pipeline; caught and corrected several repeated-query failures (the same point being resubmitted because both exploration settings converged on it). This recurring problem motivated building a duplicate/near-duplicate detector into the pipeline in Week 9.
- **Week 10:** Diagnosed a severe miscalibration for Function 5 (a prediction missed by 7.4 standard deviations) traced to its output spanning four orders of magnitude; fixed with a log-transform, validated by checking (and correctly rejecting) the same fix for three other functions where no calibration problem existed. Separately, tested a promising-looking log-based transform for Function 1 that initially appeared to fix its persistent kernel collapse, but was found — through further checking — to invert the correct ranking of outcomes, and was rejected despite the appealing headline result.

---

## Performance

**Metrics used:**
- **Current best observed value per function** (primary measure of progress).
- **Log marginal likelihood (LML)**, for comparing kernel/bound/alpha/transform choices.
- **Leave-one-out calibration** (fraction of held-out predictions falling outside 2 standard deviations; target ≈5%) — used to catch overconfident models before trusting their suggestions.
- **Duplicate-query** — introduced once the repeated-query problem was identified.

**Results across the 8 functions (best value as of Week 10):**

| Fn | Best value found | Notes |
|---|---|---|
| 1 | ≈0 (true best −0.0036) | Persistent kernel collapse; unresolved despite multiple diagnostic attempts |
| 2 | 0.611 | Found early; not improved on since |
| 3 | −0.0007 | Substantial improvement over early rounds |
| 4 | 0.368 | Turned positive after an all-negative history |
| 5 | 5431 | Strong sustained growth; required a log-transform fix mid-project |
| 6 | −0.114 | Large improvement following confirmed noise-level correction |
| 7 | 3.08 | Steady improvement, recurring duplicate-pick issue managed |
| 8 | 9.91 | Modest gains; output range was always narrow |

---

## Assumptions and Limitations

**Key assumptions:**
- **Stationarity** — one `length_scale`/`alpha` applies across the whole domain. Directly violated for Function 5 (heteroscedastic across its range).
- **Smoothness** — the kernels used assume continuous, smooth functions
- **Gaussian noise** — confirmed to exist for at least one function (Function 6) via a real repeated measurement, but its true distribution is unverified.
- **Candidate-based optimisation** — the acquisition function is maximised over a finite sample (a grid for 2D, 20,000 random points otherwise), not solved exactly; this is an approximation whose quality depends on candidate density, which degrades in higher dimensions.

**Known failure modes:**
- **Kernel collapse** for sparse-signal functions (Function 1) — not resolved by boundingor transforms.
- **Repeated queries** when exploration and exploitation settings agree on the same point.
- **Miscalibration under extreme output scale** — found for one function

---

## Ethical Considerations

Every methodological change in this project is supported by  a calibration check, an LML comparison, a documented reason for any per-function override, rather than an unexplained parameter choice.

---

## Reflection: Decision-Making, Strengths, Limitations, and Further Detail

**How it makes decisions:** a Gaussian Process posterior (mean and uncertainty) is combined via UCB into a single score, maximised over a candidate set. Kernel type, length_scale bound, alpha, and (where justified) an output transform are chosen by comparing LML across a grid, not fixed in advance. An automated duplicate-check and periodic calibration checks act as a sanity net on this approach.

**Strengths:** principled, calibrated uncertainty

**Limitations:** grid search cost grows multiplicatively with every added tuning axis, limiting how many further refinements can be explored exhaustively; the one genuinely unresolved problem (Function 1) shows the approach has a real ceiling when data is this sparse.

