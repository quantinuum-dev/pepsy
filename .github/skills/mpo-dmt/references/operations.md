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

Ordinary MPO payloads follow the optimizer's transposed-gate convention:
use `U.T` for physical ket action `U`. Verify nonunitary target scale and
support order, not just trace-one unitary examples.

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
string ordering and cover tie-breaking. Do not cache tensor-dependent
environments, projected vectors or SVD support across writes/gauge moves.

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

For White et al. comparisons reuse the
[figure convention record](../../../../docs/development/notes/2026-11-19-dmt-white-figures.md)
and extracted vector-PDF fixture. Fig. 7 has L=64, chi=16; Fig. 8 has
L=16/20/24. State time/energy conversions and endpoint alignment explicitly.
Distinguish agreement with a published energy curve from an exact dynamical
reference, and keep digitization precision separate from simulation error.
