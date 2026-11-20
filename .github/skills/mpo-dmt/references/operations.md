# DMT API, verification cost and validation

Read the owning [MPO contract](../../../../docs/api/operators/mpo_dmt.md)
and [Pauli workflow](../../../../docs/api/optimizers/pauli_dmt.md) alongside
the affected code. Mathematical constraints and runtime verification are
separate controls.

## Choose the operation and verification level

| Operation | Full-region verification default | Scalar/bond checks |
| --- | --- | --- |
| Standalone `compress_dmt` / `compress_mpo_dmt` | `check_regions=True` | Active |
| `PauliDMT`, `PauliMPO.apply_gate_dmt` | `check_regions=False` | Active |
| `MpoOptimizer.run(mode="dmt")` | `dmt_check_regions=False` | Active |

Strict standalone sweeps compare original and final regions once. Do not
reintroduce repeated full-inventory scans inside that sweep. Explicit full
replay verification compares every requested region after each update,
including default contiguous triples. Relaxed full checks also provide
per-bond density-error data, so retain those requested measurements.

Density paths retain scalar trace guards, including a replay's first full
sweep. Pauli gates respect an explicit `trace_guard=False`. Raw standalone
operator compression and ordinary operator replay do not classify a trace;
persistent Pauli operator evolution retains its documented `trace_guard`
choice. Do not impose a nonzero relative-trace test on genuinely traceless
operators.

`preservation_check` controls failed finite checks: `"raise"` is the library
default, `"warn"` continues and records errors, and `"off"` skips outer
verification. MPO replay prefixes it with `dmt_`. Respect the user's selected
policy. Off does not remove protected constraints, adaptive feasibility
checks or density denominators. A failed SVD cannot produce a candidate.

`report.check_regions` records the resolved full-region setting;
`preservation_check="off"` resolves it to false. `diagnostics=True` and
`return_report=True` collect extra residuals but do not enable full-region
verification. Unmeasured density errors remain `None`, not zero.
Warning reports survive without full diagnostics. Collect
`report.preservation_errors` during a trajectory: reports describe the latest
operation, not a persistent error history.

Ordinary replay defaults to `dmt_update="canonical"`: retain its owned
workspace across matching calls; handle one-site actions at the tracked
center without a new compression sweep. `info_c` is a stable dictionary for
the live MPO, with scratch-target info kept separate. Invalidate reuse on
state/tensor/array, index/gauge, exponent, backend/dtype/device or DMT-setting
changes, and after manual canonicalization or rollback. Check object/version
metadata without hashing buffers or scanning numerical isometries. Raw NumPy
buffer writes require `sync_canonicalization()` before replay. Keep
`dmt_update="full"` for an explicit every-gate reference; this is still
protected SVD, not variational DMRG. See the
[cache record](../../../../docs/development/notes/2026-11-19-dmt-canonical-cache.md).

Ordinary MPO payloads follow the optimizer's transposed-gate convention:
use `U.T` for physical ket action `U`. Verify nonunitary target scale and
support order, not just trace-one unitary examples.

Keep canonical metadata ownership consistent: `PauliMPO`, `PauliDMT` and
`MpoOptimizer` each own a stable `info_c` dictionary; copies/snapshots never
share dictionaries. `cur_orthog=None` means unknown. Record a native center
only after an owned QR/SVD/DMT establishes it, and bind reuse to actual
cores/version metadata. Symbolic Pauli-word sorting establishes no gauge.
Native canonicalization moves only the known center distance. Explicit raw
buffer synchronization invalidates an untrusted state gauge before rebuilding;
a failed rebuild must not leave a reusable old claim. Persistent-driver
metadata mirrors its workspace even after a partial failing evolution.
See the [common contract](../../../../docs/api/optimizers/pauli_dmt.md#canonical-metadata-ownership).

## Larger contiguous regions

Build the requested inventory explicitly:

```python
regions = [tuple(range(j, j + width)) for j in range(L - width + 1)]
```

For generic full protected ranks at an interior qubit cut, the smaller-half
cover gives `p+q` up to 8/20/32 for widths 3/4/5. These are sufficient
construction budgets, not universal minimum feasible ranks. Inspect actual
reported ranks and the paper residual budget `chi-p-q`. Wider protection
at fixed chi can leave too little free rank for subsequent dynamics.

Test `truncation="optimized", max_refinement_sweeps=0` as an explicit
feasible-seed alternative before adding alternating sweeps. Do not present
a better one-cut objective or one trajectory as a global accuracy guarantee.
Keep bad chi/width combinations in results. Neither mode enforces positivity;
real Pauli coefficients maintain Hermiticity, not positive semidefiniteness.

Crossing-region planning is indexed once per workspace. Preserve its exact
string ordering and cover tie-breaking. Cache trace environments only with
correct changed-range invalidation. Never reuse projected protection vectors
or SVD support across tensor writes or gauge moves.

## Profile without changing the scientific comparison

- Warm dispatch/BLAS first and use several timings. Match backend, dtype,
  thread count, cutoff, linalg policy, protection inventory and diagnostics.
- Include genuinely occupied high-rank bonds; early product-state gates
  alone do not measure large-chi cost. Separate preparation, evolution,
  full verification and independent readout.
- Measure complete operations as well as hot helpers. A planning speedup is
  not the same as an end-to-end gain. Alternate old/new order when possible.
- Keep necessary norm scaling, canonical factorizations and protected-basis
  retries. Optimize duplicated work before dropping verification. Compare
  output arrays or physical observables against the prior implementation.

Use the public `TorchLinalgConfig` for SVD/QR choices. Native CPU SVD can
fail on finite, ill-conditioned inputs. Its supported regularized policy
provides native forward SVD with GESVD fallback; always-GESVD is a separate
timed choice. Retain dtype/device and use the shared
[numerical contracts](../../pepsy-maintainer/references/numerical-contracts.md).
Autodiff stabilization does not make the DMT state positive or cure all
finite-precision preservation loss.

## Independent evidence

Use [larger-window tests](../../../../tests/test_dmt_window_scaling.py),
[check-cost/API tests](../../../../tests/test_dmt_check_costs.py) and
[warning-policy tests](../../../../tests/test_dmt_preservation_policy.py)
beside the source owner's suites. Validate the final full-check failure
transactionally and ensure fast defaults do not contract full panels.
For accuracy use statevectors on small chains or physical local oracles;
avoid global dense density matrices on large chains.

Reproducible entry points:
[window scaling](../../../../examples/dmt_window_scaling.py),
[verification cost](../../../../examples/dmt_preservation_benchmark.py),
[Ising dynamics](../../../../examples/dmt_ising_dynamics.py),
[physical 2D dynamics](../../../../examples/dmt_2d_dynamics.py).
The [dated scaling record](../../../../docs/development/notes/2026-11-19-dmt-window-scaling.md)
contains measured scope and counterexamples, not universal performance claims.

For mapped 2D dynamics, include every physical Hamiltonian edge and construct
regions with the same mapping. Compare continuous-time exact evolution with
the uncompressed identical Trotter circuit before assessing chi convergence.
Record raw trace, Hermiticity and local/global negativity beside observable
errors. Compare against the current gated state for preservation audits;
FIT's live per-gate source can differ from its optimizer's stored snapshot.
The [finite-size 2D record](../../../../docs/development/notes/2026-11-19-dmt-2d-dynamics.md)
compares ordinary DMT and unconstrained FIT. Do not promote energy agreement,
roundoff-level preservation, or the largest tested chi to a long-time accuracy
guarantee, or describe mapped MPO evidence as native PEPO validation.

For 4×4/rectangular trajectories, use the
[scalable benchmark](../../../../examples/dmt_2d_scaling.py) and
[measured scope](../../../../docs/development/notes/2026-11-19-dmt-2d-scaling.md).
Use sparse statevectors plus local marginals for the stated depolarized pure
family; do not allocate a full 16-spin density. Global HS Hermiticity overlaps
have a cancellation floor and are not a PSD certificate. Label the exact
rank-two Ising target/full-sweep variant separately from routed replay and
separate timestep, target construction, compression, audits and readout costs.

For the [6×6 real-Pauli interval prototype](../../../../docs/development/notes/2026-11-19-dmt-2d-window.md),
reuse the canonical workspace and exactly prepare only the changed interval
before its protected splits. Real coefficient storage enforces Hermiticity,
not positivity. Retain the independent fresh scalar trace guard; cached
protection-vector checks alone missed large-chi failures. Warn-and-continue
applies to finite preservation defects, not native process faults. Keep
failed runs out of convergence references and distinguish uncontrolled Apple
Accelerate pilot timings from explicitly VECLIB-limited runs.

For White et al. comparisons reuse the
[figure convention record](../../../../docs/development/notes/2026-11-19-dmt-white-figures.md)
and extracted vector-PDF fixture. Fig. 7 has L=64, chi=16; Fig. 8 has
L=16/20/24. State time/energy conversions and endpoint alignment explicitly.
Distinguish agreement with a published energy curve from an exact dynamical
reference, and keep digitization precision separate from simulation error.
