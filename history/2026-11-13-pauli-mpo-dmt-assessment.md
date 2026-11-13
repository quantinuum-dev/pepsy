# 2026-11-13 — PauliMPO DMT integration assessment

The native Pauli cores support a focused 1D DMT extension, but current SVD
compression does not protect local density matrices or energy.

- Scope: assess the energy-preserving compression in
  [White et al. (2018)](https://link.aps.org/accepted/10.1103/PhysRevB.97.035127)
  and the effort to add it to `PauliMPO`; no implementation requested.
- Branch / baseline: `develop`, `44a91cec`.
- Status: assessment only; this journal is uncommitted. Existing working-tree
  changes were preserved. No source code changed, staged, committed or pushed.

## Findings

- Confirmed native `(left, right, 4)` cores, QR canonicalization, ordinary SVD
  compression, trace/partial-trace/inner readouts and conjugation gates in
  [pauli_mpo.py](../src/pepsy/operators/pauli_mpo.py).
- Proposed work: protected left/right operator subspaces, connected-correlation
  truncation, exact reconstruction of protected blocks, explicit rank-budget
  handling and preservation diagnostics. This is a new compression policy.
- Protecting all three-contiguous-site marginals protects energy for a
  Hamiltonian whose terms fit inside those regions. Trotter error remains.
- The paper's connected-correlation formula divides by the trace. A first
  density-operator implementation should require nonzero trace; support for
  traceless Heisenberg observables needs a separately specified generalization.
- Current core-backed gate application embeds a gate MPO and multiplies full
  core chains. A scalable local gate/split driver is separate additional work.
- The earlier [lattice-DMT audit](2026-11-09-lattice-dmt-capability-audit.md)
  concerns a different paper; its real-core conversion finding was subsequently
  fixed, as recorded in the Pauli/DAOE journals.

## Read-only numerical check

Used the session environment and current checkout with Quimb 1.15.0, Autoray
0.8.4, Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1 and Symmray
0.4.1.dev15+g0374aaa3c. No dependency or implementation changes were made.

For four sites, constructed a positive density matrix from identity plus
translated Pauli terms with coefficients `IIII:1/16`, `ZZ:0.08/16`,
`X:0.06/16`, `Y:0.04/16`, `XYZ:0.05/16`. Its minimum eigenvalue was 0.0392453.
For the translated Hamiltonian `ZZ + 0.7 X + 0.2 Z`, ordinary
`compress_pauli(max_bond=2, cutoff=0, renorm=False)` changed normalized energy
from 0.408 to 0.4006215961, trace from 1 to 1.0003698353, and a three-site
marginal by up to 0.0181882 in Frobenius norm. Final bonds were `(2, 2, 2)`.
This demonstrates absent preservation constraints; it is not a DMT benchmark
or evidence that bond two suffices for DMT's stronger constraints.

No implementation tests, dynamics benchmarks or full suite were run.

## Follow-up: implementation-path investigation

The user requested a deeper comparison with Pepsy, Quimb and Autoray and a
clear implementation path. Added the
[design and feasibility note](../docs/development/notes/2026-11-13-pauli-dmt-design.md).
This note and journal are uncommitted; production source remains unchanged.

- **Refinement of the zero-trace finding above:** a suitable published variant
  already exists in [Yi-Thomas et al. 2024, Appendix D](https://arxiv.org/pdf/2310.06886).
  It truncates the raw unprotected block without forming connected
  correlations. A shared kernel with explicit density/operator policies can
  support both uses; a novel zero-trace method is unnecessary.
- Inspected installed Quimb canonicalization, gate splitting, 1D compression
  dispatch, TEBD sweeps and split-driver registration, plus Autoray dispatch.
  Reviewed the official upstream sources linked in the design note.
- Temporary six-site protected sweeps at bond cap 12 preserved three-site
  marginals to about 1e-15. Ordinary SVD changed one by 0.00217. Both raw and
  connected variants were probed; no global-error superiority is claimed.
- NumPy/Torch/JAX complex-double matrix-kernel checks, compact-factor
  reconstruction, rank-deficient raw cases, a local gate/dense oracle and
  public Quimb dimension-four MPS interoperability assertions passed.
- The temporary direct two-site formulation is a feasibility reference. It
  preserves the required partial traces but has not been checked for equality
  to the paper's compact Schmidt-space prescription. The note keeps that
  distinction explicit.
- An existing global-product gate reference produced large intermediates and
  its dense readout was interrupted. The completed local-gate probe uses a
  small dense oracle instead; there is no performance benchmark.
- Proposed order: private protected kernel and explicit report, public
  `compress_dmt`, then local gate/sweep integration under the owning namespaces.
  Thin bases, compact final QR/SVD, cached trace environments and real Pauli
  coefficients are the identified implementation improvements.
- The final temporary probe completed successfully. Local Markdown targets
  and whitespace were checked. No source tests/full suite, GPU, gradients,
  JIT, transport or long-time convergence checks were run.

## Implementation after user authorization

The user then authorized implementation with native Quimb functions and
Autoray and asked for an efficient implementation. Status: working-tree
implementation on `develop` at baseline `44a91cec`; nothing staged, committed
or pushed. Existing sampling edits and unrelated notes remain untouched.

- Added `operators/_pauli_dmt.py`: Quimb `array_split` QR/SVD, explicit
  density/connected and raw/operator variants, compact Schmidt support,
  thin-factor reconstruction, separate numerical support/residual diagnostics,
  physical-constraint checks and rank-budget failures.
- Added `PauliMPO.compress_dmt` and adjacent `apply_gate_dmt`. Existing-chain
  compression exposes the small bond centre with QR/LQ instead of merging
  larger physical pairs. Gate application forms only the local target.
- Added `optimizers.PauliDMT`: native EIGH gate preparation, second-order
  even/odd/even evolution, persistent canonical/identity environments,
  independent state snapshots and bounded latest-timestep reports.
- Added public report exports, API documentation, owning-module navigation,
  a changelog entry and 44 focused regressions. The
  [design note](../docs/development/notes/2026-11-13-pauli-dmt-design.md#implementation-validation-and-numerical-limits)
  retains the algorithm details, measurements and numerical limits.

### Validation

- New DMT tests: **44 passed** within the combined selection.
- Pauli/DMT/damping/public-API/package selection: **226 passed, 1 skipped,
  1 failed**. Failure: `test_package_version_matches_installed_distribution`,
  installed metadata 0.5.0 versus source 0.6.0, already documented before this
  task. The skip requires two logical JAX CPU devices configured before
  process startup.
- Default smoke: **111 passed, 1 failed**, the identical version mismatch.
- Full collection with `--maxfail=1`: **102 passed, 7 skipped, 1 failed** at
  `test_bp_cluster_patches.py::test_loopunion_selects_both_plaquettes_and_clips_at_edge[where0-expected0]`.
  Its NumPy input is dispatched to Torch reshape through the existing
  Quimb/Autoray namespace path. Restoring the original `HEAD` Pauli module in
  memory reproduces the same first failure and counts. A further baseline
  run restores the original Pauli module and both public facades, explicitly
  collects only the pre-existing tests, and asserts that neither new DMT
  module has been imported; it reproduces the same failure and counts.
  No shared environment or unrelated source was changed to work around it.
- `python -m ruff check src tests`, whitespace and edited Markdown local-link
  checks pass. The documented eight-site example executes. The design note
  records a limited NumPy comparison of the compact-bond versus merged-pair
  paths; no general performance optimality is claimed.

### Remaining limits

Open dense chains only. CPU NumPy/Torch/JAX forward behavior, float32/complex128
coverage and one scoped Torch gradient are tested; GPU, JIT, native symmetry
and transport convergence are unverified. Near-singular supports can change
the global approximation under roundoff/gauge changes; original physical
partial-trace constraints are checked, with dedicated nearly dependent cases.
No positivity or exact future-dynamics guarantee is introduced.

## Follow-up: larger systems and optimization review

The user requested another correctness pass on larger systems, followed by
an optimization review focused on canonical gauges, metadata and caching.
All changes remain uncommitted on the same branch/baseline; unrelated edits
and the device-local override remain untouched.

- Found long-chain precision failures hidden by tiny absolute coefficient
  residuals. Retained initial/final Schmidt support, rescaled centres and
  norms, added relative diagnostics and a linear log-scaled check of every
  local marginal before a standalone compression commits. Strong 128-site
  mixtures at cap 12 now fail explicitly when necessary; cap 24 recovers
  the source. Added extreme-scale and failure-rollback checks.
- Added large-chain analytic-mixture tests through 512 sites, larger source
  rank 48, 128-site XX dynamics against a one-particle oracle, 256-site
  commuting dynamics against an exact three-spin oracle, and truncated
  32-site density evolution with timestep/energy checks.
- Final 512-site, cap-12 NumPy double mixture result: maximum three-site
  Frobenius error `1.61e-13`, trace error `1.84e-13`.
- Replaced final two-QR/SVD refactoring by one directional QR; reused the
  initial SVD on basis retries; combined repeated constraint calculations;
  cached constant identity bases; reused shared-Hamiltonian EIGH/transfers
  within each call. Caches intentionally do not survive across parameter
  updates/autograd calls.
- Carried the canonical centre through serpentine parity sweeps, eliminating
  full return moves between timesteps. Explicit tests check isometries,
  cached/fresh results, bounded travel, continuation across calls, mutable
  Hamiltonian preparation and nonzero Torch parameter gradients.
- The warmed 64-site, cap-16 benchmark improved evolution from 717 to 562 ms
  (seven-run medians), with SVD calls 2520 → 1779. This is one CPU workload,
  not universal optimality. The [design note](../docs/development/notes/2026-11-13-pauli-dmt-design.md#larger-system-and-optimization-review)
  records the setup and a rejected overly strict rank-selection experiment.
- Rescaled Hamiltonian Hermiticity validation to prevent overflow-based
  acceptance of huge non-Hermitian inputs; reject mixed core dtypes clearly.

The full-suite retry again stopped at the same previously baseline-reproduced
BP NumPy-to-Torch reshape failure: **102 passed, 7 skipped, 1 failed**.
Before the optimization pass, the expanded unfiltered focused selection had
**255 passed, 1 skipped, 1 failed**, with only the known installed-version
metadata mismatch. Final optimization validation is recorded below.

- Revised DMT selection: **81 passed** within **263 passed, 1 skipped,
  1 deselected** across Pauli/DMT/damping/public-API/package tests. Only the
  known installed-version metadata test was deselected after its repeated
  unfiltered failure; the skip needs two logical JAX CPU devices.
- Ruff over `src tests`, whitespace and edited Markdown local-link checks
  pass. CPU NumPy/Torch/JAX and the scoped gradient checks pass; GPU,
  end-to-end JIT, native symmetry and long-time transport remain unverified.
- No files were staged, committed or published.

## Further review: gate-path numerical gap

The user requested another review and a search for open-source implementations.
Found [Jack Kemp's ITensor DMT](https://github.com/Jack-Kemp/dmt) and
[MPSToolkit.jl](https://github.com/jayren3996/MPSToolkit); compared accessible
documentation and Kemp's truncation source. MPSToolkit's implementation pages
were inaccessible and neither external codebase was executed.

A 128-site float32 product-mixture test with 127 identity gates exposed an
unresolved correctness problem: the persistent operator driver returned trace
79.95 at cap 12, 280.34 at cap 16 and 112.57 at cap 24 (initial trace 1),
without an error. At cap 12, float64 produced trace 1.00000017. The input's
nominal rank is 24, but cap 24 did not recover the gated float32 result.
Standalone compression's full-marginal check does not apply to gate updates;
the per-gate normwise check can miss amplified physical errors. Added an
expected-failure regression and updated API documentation. A correct efficient
per-gate certificate or better-conditioned formulation remains open; no
production numerical algorithm change was made in this review.

Validation for this follow-up: large-chain and optimization selections
**37 passed, 1 expected failure**, with the existing Torch QR derivative
warning. Ruff over `src tests` and `git diff --check` passed. Earlier
full-suite and package-metadata failures remain as recorded above.

## Follow-up: stability investigation and source comparison

The user asked for a deeper investigation of the gate instability and
whether the external implementations suggest a remedy. Reproduced the
128-site identity-gate case with temporary Quimb-based changes; no production
factorization was changed. A final QR/QR/SVD and/or an `8 * eps` preparatory
support cutoff improved this particular stress test but still left maximum
three-site errors of at least 0.167 in float32 at cap 12. The cutoff probe
did not protect or report discarded support, so it is not safe to adopt.

Measured the trace component relative to the bond norm decreasing to around
float32 precision as the sweep progressed, explaining why normwise checks can
miss large physical trace errors. Changing the overall operator scale did not
fix the issue. Explicitly promoting a float32 source to a float64 working
state gave errors `1.31e-7` and `1.59e-7` at caps 12 and 24 in this case;
added an all-marginal regression for both, without changing default dtype.

The GitHub connector successfully retrieved the Julia reference sources. Its
small repair SVD and support cutoff do not by themselves solve the reproduced
case in the temporary analogous probe. Its constraint checkpoint projects
onto a model-specific sector, not onto the desired local marginals. The
automatic connector threshold is a policy Pepsy deliberately does not adopt.
Neither external library was executed; Julia ITensor packages are absent.
The [design note](../docs/development/notes/2026-11-13-pauli-dmt-design.md#conditioning-and-reference-factorization-investigation)
records the data and limitations. The per-gate precision gap remains open.

Validation: all three DMT suites **83 passed, 1 expected failure**. No
dependencies were installed and no work was committed or published.

## Follow-up: double precision and cutoff policy

The user asked about double/complex-double stability and a clear
`cutoff="auto"` / Quimb `cutoff_mode` API. Added those options to standalone
compression, local gate calls and `PauliDMT`. `auto` resolves via Pepsy's
shared dtype policy, the existing default cutoff remains zero, and `auto` or
`None` mode resolves to `rel`. The six Quimb modes apply only to the residual;
absolute modes convert physical Hilbert–Schmidt cutoff units to the scaled
Pauli-centre units. Reports expose the resolved cutoff mode.

The [design note](../docs/development/notes/2026-11-13-pauli-dmt-design.md#cutoff-policy-and-double-precision-follow-up)
records Quimb 1.15 versus current upstream defaults, conversion details,
128-site real/complex gauge results and the scope of the new regressions.
The `auto` cutoff improved the measured real cases and worsened the measured
complex-gauge cases. Per-gate numerical stability is still not guaranteed,
including in double precision. The unrelated float32 expected failure
remains documented.

The combined Pauli/DMT/damping/public-API/package selection had **306 passed,
1 skipped, 1 deselected, 1 expected failure**; the known installed-version
metadata test was deselected and the skip requires two logical JAX devices.
Six additional end-to-end cutoff-mode cases, added after that collection,
passed separately as part of **43 cutoff tests**. Ruff, whitespace and edited
Markdown link checks passed. No changes were committed or published.
The full-suite retry again stopped at the baseline-reproduced BP NumPy/Torch
reshape failure after **102 passed and 7 skipped**; it is not a clean full run.

## Follow-up: auto-cutoff accuracy and gate trace guard

The user requested a deeper complex128/float64 stability and performance
review against ITensor/Julia. On the 128-site identity-gate probe, the
previous auto-versus-zero differences originated at the fully protected
boundary bonds: the positive-cutoff path reconstructed an unchanged bond.
Added an exact pass-through, including a no-SVD boundary-gate fast path.
Auto and zero now agree in the tested 128-site real and complex-gauge cases;
their physical marginal error remains around `1e-7`. Dense boundary-gate
references validate the full output and check that no SVD is called.

The 192/256-site stress tests showed that complex128 can still develop severe
trace drift. Added a relative trace-defect guard before committing a gated
split. `trace_guard="auto"` uses the declared density variant or a log-scaled
initial trace-to-norm heuristic; `True` forces the guard, `False` disables it,
and `run.trace_guard_active` exposes its choice. The guard catches the
192-site double-precision cases and the 256-site case when forced. It is a
per-split trace check, not a local-marginal or accumulated-error certificate.
A float32 operator with a physically nonzero but unresolved initial trace
still fails with the default auto classifier, retained as an expected failure;
forcing the guard detects the severe drift. Added a regression for nonfinite
gated targets and preserved standalone transactional behavior.

The design note records the independent dense/large-chain tests, source
comparison, rejected strict-basis retry, boundary and trace-guard CPU timing,
and remaining limitations. Julia/ITensor reference code was inspected, not
executed. No dependencies were installed and no work was committed or published.

Final combined Pauli/DMT/damping/public-API/package checks: **327 passed,
1 skipped, 1 deselected, 1 expected failure**. The skip needs two logical JAX
devices, the deselection is the known installed-version mismatch, and the
expected failure is the unresolved float32 auto trace-classification case.
Ruff, whitespace and edited Markdown local-link checks passed. The full suite
still stops at the baseline-reproduced BP NumPy/Torch reshape failure after
102 passes and seven skips.

## Final API review before commit

The user requested a final review, commit and push. Clarified that protected
rank zero means “not computed” on the exact QR pass-through. Standalone
compression no longer runs the gate-only trace classifier; the full
post-sweep marginal check remains in place. A focused regression confirms the
unused classifier is not called, and 14 standalone compression checks passed.
The pre-review combined selection again produced **327 passed, 1 skipped,
1 deselected, 1 expected failure**. The full-suite retry again stopped at the
baseline-reproduced BP NumPy/Torch dispatch failure after 102 passes and
seven skips. Ruff and whitespace checks passed. The known finite-precision
and auto trace-classification limits remain documented in the API guide.
