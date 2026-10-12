# Boundary amplitudes for PEPS sampling (2026-09-30)

## Decision and numerical distinction

The roughening workflow requires boundary-MPS sampling and amplitude estimates,
without a full-network exact amplitude contraction for each sampled bitstring.
The existing χ and χ′ controlled proposal environments only. The implementation
now adds `amplitude_mode="boundary"` and `amplitude_chi` to `PepsSampler`.
The cap defaults to χ′; the downstream roughening runner selects boundary mode
by default. The library retains its exact-amplitude default for compatibility.

The accumulated conditional probability is q(s). It suffices for draws and
ordinary q averages. A separate amplitude is needed for the importance
correction |Ψ(s)|²/q(s), not for drawing a configuration. With boundary-MPS
amplitudes the correction is approximate, and probabilities cannot supply the
complex phase. The conditioned proposal boundary cannot simply be reused as
a physically scaled amplitude: its internal rescaling intentionally drops
normalization factors. This change instead projects the original private ket
and performs a bounded boundary contraction.

## Implementation and upstream choice

Adopt the installed public Quimb `contract_boundary` API: sweep `ymin` with
`max_bond=amplitude_chi`, sampler cutoff, cutoff mode through `compress_opts`,
`max_separation=0`, `equalize_norms=True`, and `strip_exponent=True`. After rows
collapse, the remaining one-dimensional boundary is contracted to a scalar.
No full-network amplitude tree or exact amplitude preflight is constructed in
this mode. Phase and the base-10 PEPS exponent survive. Approximation metadata
is carried through serial, prefix-grouped, chunked results and diagnostics.

The active task's prior dependency audit was reused: Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865,
Torch 2.6.0+cu124. No installed packages were changed.

## Validation

- New NumPy/Torch tests recover exact small-system complex amplitudes with a
  large boundary cap, including PEPS exponent 200, and forbid the exact-plan
  check. A cap-1 versus cap-16 test demonstrates actual truncation dependence.
- Sampler, new amplitude, public API, and package-layout selection:
  **297 passed, 2 skipped, 1 failed**. The failure is the pre-existing
  `test_package_version_matches_installed_distribution`: installed metadata
  reports 0.4.0 while project metadata reports 0.5.0.
- Default Pepsy smoke suite: **93 passed**. Full `ruff check src tests` passed.
- Downstream PEPS and sweep tests: **86 passed**; two diagonal/dense convention
  tests passed. The strengthened saved-observable regression also passed.
- Downstream changed Python files pass Ruff; the broader benchmark Ruff check
  retains six pre-existing issues in unrelated effective correlation/theta
  files. A full numerical repository suite was not run.

## Downstream experiment

`magnetization.runners.roughening_peps_reference` runs the canonical SU engine
and matched dense Trotter gates for a 3×3 lattice, D=2/4/6, twelve delta-theta
points, dt=0.1 through t=6, and 4096 Z samples at t=6. Data are explicitly saved
in the examples project's ignored benchmark `store/` directory.
`plots/run_exact_peps.ipynb` reads the data and reports true normalized overlap
fidelity, boundary/BP/loop norms, dense validation observables, and raw versus
boundary-amplitude-weighted sample estimates. Dense PEPS contraction is confined
to this nine-qubit validation harness; it is not part of boundary sampling.

The experiment also distinguishes raw nearest-neighbor wall count from the
selected-wall topology diagnostic. Sample standard errors exclude contraction
bias. Energy and middle-cut entropy are dense validation quantities, not
Z-shot estimates. Larger-lattice performance and GPU sampling were not measured.

All **36/36 PEPS cases and 12/12 exact references completed**. The new notebook
executed all nine code cells with no errors and retains eight figures. Its
validation copy is `/tmp/pepsy_examples_executed/run_exact_peps.ipynb`.
Root `validation.json` records saved-data checks: all shots present, zero
exact amplitude plans, norm measurements only at t=6, exact-vector norms one,
initial fidelities one, finite fidelities in [0,1], and normalized weights.

| D | Mean F at t=6 | F range | Minimum ESS / 4096 | Maximum boundary norm-squared relative error |
| --- | --- | --- | --- | --- |
| 2 | 0.007737 | 0.000858–0.025006 | 4096 | 1.11e-15 |
| 4 | 0.012300 | 0.000583–0.058615 | 4096 | 2.66e-15 |
| 6 | 0.011879 | 0.001392–0.057927 | 3765.45 | 0.09101 |

All BP solves converged. Maximum norm-squared relative errors for BP were
17.30%, 26.99%, 36.37% at D=2,4,6; loop-size-6 maxima were 15.09%, 24.85%,
30.53%. Convergence of BP messages is not accuracy of its norm approximation.
Boundary amplitude relative errors on drawn configurations stayed below
2.14e-15 for all cases. These small-system amplitudes happen to be converged,
despite finite-cap error in the proposal and double-layer measurements.

For D=6, raw/weighted imbalance RMSE against the actual PEPS was
0.011915/0.004878; magnetization-second-moment RMSE was 0.009591/0.002890.
For D=2/4, weights were uniform to numerical precision, so raw and weighted
averages agreed. Thus the additional amplitude correction helped this D=6
proposal. It does not repair the substantial SU evolution error demonstrated
by the low true fidelities. Increasing D did not give monotonic accuracy here.
