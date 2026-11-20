# DMT verification policy and L=64 continuation

**Implemented:** finite preservation failures can warn and continue while
retaining their measured errors. Disabling verification reduces runtime but
does not repair the observed L=64 precision loss or nonpositive dynamics.

[Comparison figure](2026-11-19-dmt-preservation-fig7.png),
[SVG](2026-11-19-dmt-preservation-fig7.svg),
[plotted data](2026-11-19-dmt-preservation-fig7.csv),
[summary and timing samples](2026-11-19-dmt-preservation-summary.json), and
[all 21 complex128 Pauli warnings](2026-11-19-dmt-preservation-errors.jsonl)
retain the measured evidence.

## API and error records

`preservation_check="raise"` remains the library default. `"warn"` records
every finite violation in `report.preservation_errors` and warns once per
check kind per workspace. `"off"` skips outer bond/trace and full marginal
verification. MPO replay takes `dmt_preservation_check`; its
`dmt_check_regions=False` skips just full-region verification.
Protected truncation, support construction, and feasibility checks remain.
See the [API contract](../../api/operators/mpo_dmt.md#preservation-verification-policy).

The Ising benchmark CLIs default to `"warn"` and write each measured failure
to `preservation-errors.jsonl`, with its step, gate index and bond. Summaries
retain counts and a bounded example list. Reports themselves cover the latest
operation; library users must collect them during evolution for a full log.
The CLI's `--skip-region-checks` retains bond verification and independent
sampled benchmark audits. `--preservation-check off` skips library outer
verification; requested diagnostics and benchmark audits still cost time.

## Numerical results

The protocol is the [recovered White et al. Fig. 7 setup](2026-11-19-dmt-white-figures.md):
L=64, chi=16, figure time 40, figure dt=0.0625, unmerged turnaround
half gates, near-Y figure state, zero cutoff, paper truncation, strict
protected constraints. Energy is normalized by the measured trace and
converted to the published units; absolute trace drift is reported separately.
Torch CPU uses complex128 cores/gates, except the explicitly identified
real-Pauli run, which uses float64 cores and complex128 gates.

| Pauli representation / SVD | Max absolute trace drift | Max total energy drift, figure units | Lowest sampled three-site eigenvalue |
| --- | ---: | ---: | ---: |
| complex128 / Torch | 7.0234e-5 | 1.6102e-3 | -1.1847 |
| complex128 / SciPy GESVD | 7.8128e-5 | 1.6086e-3 | -0.096765 |
| float64 / Torch | 5.9900e-5 | 1.6189e-3 | -1.1717 |

The ordinary MPO complex128 continuation with full-region verification skipped
also completes: maximum trace drift 8.3171e-5, energy drift 1.6134e-3,
minimum local Hermitian-part eigenvalue -1.2513 and maximum three-site
Hermiticity defect 4.9540. Its zero recorded library violations are **not a
preservation certificate**: the full marginal check was skipped and its
per-gate trace monitor is inactive. Independent sampled trace readout flags
the drift. The expensive full-check ordinary MPO trajectory was stopped
after measuring its cost; no completed full-check MPO trajectory is claimed.

The real energy curves do follow the published DMT Fig. 7 curve closely:
RMSE against the digitized curve is 1.1593e-4 for native complex128 Pauli,
9.3281e-5 for GESVD, 9.7129e-5 for real Pauli and 9.3456e-5 for ordinary
MPO. This supports reproduction of the energy benchmark, despite the
physical-state defects below. Ordinary MPO additionally develops imaginary
energy up to 1.1524e-4 in figure units.

All three preserve Hermiticity in the sampled reduced operators, but none
resolves trace drift and nonpositivity. Driver choice changes the subsequent
compressed trajectory, so its smaller negative eigenvalue is not evidence
of greater accuracy. There is no exact L=64 trajectory in this comparison.
The complex128 native-SVD `warn` and `off` runs have **identical saved arrays**.
Verification is not the cause of their instability.

The complex128 native-SVD run records 21 above-threshold trace defects, with
maximum per-gate relative trace error 1.4020e-6. Its nine sparse gate audits
give errors near machine precision, but miss the troublesome early gates.
An additional **504-gate audit**, covering every gate in the first four
steps, measures maximum three-site Frobenius defect 1.6166e-6, absolute trace
jump 1.4020e-6 and energy jump 3.5734e-7 in raw Pauli-Hamiltonian units.
The original trace warning therefore detects actual preservation loss.

Two effects must be distinguished: finite-precision loss of protected local
data, and the absence of a positivity constraint even for an exact DMT split.
In exact arithmetic, the protected condition

$$
\operatorname{Tr}_{\bar R}{\color{purple}\widetilde\rho_n}
=\operatorname{Tr}_{\bar R}{\color{blue}\rho_n^{\mathrm{target}}}
$$

compares a split with its current gated approximate target. It neither makes
the whole operator positive nor fixes future local dynamics after discarded
correlations propagate. Large late negative local eigenvalues cannot be
explained away by energy conservation.

**Unresolved:** the particular operation causing the early trace error has
not been isolated. For a normalized pure density in raw Pauli coefficients,
the trace coefficient relative to the coefficient norm scales as
`2**(-L/2)`, giving an illustrative amplification `eps * 2**32 ~ 1e-6`
at L=64. This is a conditioning hypothesis, not an identified root cause.
No unvalidated change to the protected splitting mathematics was adopted.

A separate short **operator-variant probe** also fails to resolve precision
loss: all 504 gates in four steps give maximum three-site defect 1.0916e-6
and relative trace error 2.2504e-6. This changes the truncation objective and
is not the original density-DMT benchmark; it was not adopted as a fix.

## Verification cost

`examples/dmt_preservation_benchmark.py` uses the same L=64 tilted-Ising
initial state, chi=16, complex128 and 126 gates per step, one CPU thread,
two untimed warmup steps, and five timed steps. Full diagnostics, snapshots
and readout are outside timing. Each configuration starts from the same
state. This short test measures overhead, not long-time accuracy.

| Method | Policy | Full-region verification | Median seconds per step |
| --- | --- | --- | ---: |
| Pauli | warn | default 1D bond checks | 0.385 |
| Pauli | off | disabled | 0.362 |
| Ordinary MPO | warn | enabled | 2.465 |
| Ordinary MPO | warn | disabled | 0.538 |
| Ordinary MPO | off | disabled | 0.519 |

All corresponding final three-site arrays agree exactly between verification
settings within each representation. Ordinary MPO full-region checks dominate
this workload. Default 1D Pauli gate updates already avoid that full scan;
their remaining outer checks have a much smaller cost. These CPU measurements
are from a final run after the long trajectories completed; an earlier
concurrent run gave slower absolute times and a similar cost ratio.
They are hardware/workload dependent; rerun the example for another system.

## Reproduce

```bash
python examples/dmt_ising_paper_energy.py --backend torch \
  --methods pauli-dmt --preservation-check warn --sample-every 16 \
  --output /tmp/dmt-warn
python examples/dmt_ising_paper_energy.py --backend torch \
  --methods mpo-dmt --preservation-check warn --skip-region-checks \
  --sample-every 16 --output /tmp/dmt-warn-fast-mpo
python examples/dmt_preservation_benchmark.py --backend torch --repeats 5
```

To generate the continuation plot, pass the completed result directories as
`--fig7-continuations DIR...` to `examples/dmt_white_figures.py`, along with
the strict MPS/Fig. 8 inputs described in the earlier figure report.

To compare the SVD driver, add `--cpu-svd scipy_gesvd`. Real Pauli storage is
selected explicitly with `--methods pauli-dmt-real`. These are measured
alternatives, not established fixes for the L=64 failure.

## Validation and upstream status

The initial domain/API selection has **333 passed, 1 skipped, 1 failure**.
The failure is `test_package_version_matches_installed_distribution`:
installed distribution metadata is 0.5.0 while both HEAD and the working-tree
project version are 0.6.0. The environment was not modified. The lattice/large
selection has **145 passed, 1 expected failure**. Subsequent focused adapter,
replay, policy and benchmark checks have **115 passed**. These are focused
selections, not a full-suite result.
The final policy file has **21 passed**, including optional-policy routed
nonunitary scale and in-place report tests; the preceding policy/example
selection has **37 passed**.

New regressions cover strict transactional rejection; finite warning commits;
all-error recording with diagnostics disabled; warning deduplication; removal
of stale error reports; skipped full marginal verification; unchanged lossy
local targets; nonfinite active-input rejection; and Torch complex128 dtype
and gradient preservation. Ruff and whitespace checks pass.

The same-task upstream audit was reused with unchanged dependencies:
NumPy 2.5.2, SciPy 1.18.0, Quimb 1.15.0, Autoray 0.8.4,
Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Symmray 0.4.1.dev15+g0374aaa3c and Torch 2.9.1. The public Quimb
`array_split` route is unchanged. **Adopt:** supported CPU SVD driver selection
through `TorchLinalgConfig`. **Defer:** mathematical stability changes pending
independent preservation validation. No dependency upgrades or new shims.
