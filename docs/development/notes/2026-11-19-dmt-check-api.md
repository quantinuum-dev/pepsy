# DMT API and verification costs

**Implemented:** gate replay defaults to bond/trace checks, full-region
verification is opt-in, and strict standalone compression retains one final
full check without repeatedly scanning all regions after every bond.
The protected approximation, cutoff and precision are unchanged.

## Public behavior

| Operation | Full-check default | Full verification when enabled |
| --- | --- | --- |
| `PauliDMT`, `PauliMPO.apply_gate_dmt` | `check_regions=False` | Every requested region after each gate |
| `MpoOptimizer.run(mode="dmt")` | `dmt_check_regions=False` | Every requested region after each gate |
| Standalone compression and `MpoOptimizer.compress_dmt` | `check_regions=True` | Original/final regions once in strict mode |

Explicit full Pauli gate verification now also covers the default contiguous
triples, rather than only explicit region inventories. Relaxed full checks
retain per-bond scans to populate their density-error measurements.

Cheap relative density-trace guards now cover ordinary MPO replay, including
its first complete sweep, and standalone density compression even when
full verification is disabled. Pauli gates retain their explicit
`trace_guard=False` override. Raw standalone operators and ordinary operator
replay do not classify a trace; Pauli operator evolution keeps its existing
resolved-trace policy.

`report.check_regions` records the resolved full-check selection; it is
false when `preservation_check="off"`. Diagnostics alone do not enable full
verification. `None` density errors mean not measured. Full checks can be
selected independently of report collection.

The library failure default remains `preservation_check="raise"`. The
user-selected warning policy continues and retains every measured finite
failure, even without full diagnostics:

```python
run = PauliDMT(rho, max_bond=chi, preservation_check="warn")
rho_t = opt.run(dmt_preservation_check="warn")
# Validation run with full-region scans:
rho_t = opt.run(dmt_preservation_check="warn", dmt_check_regions=True)
```

See the [owning API contract](../../api/operators/mpo_dmt.md#preservation-verification-policy).
No normalization, positivity projection, rank relaxation or truncation
algorithm was installed as a way to pass checks. Invalid numerical inputs,
unresolved density divisions and failed decompositions can still stop a case.

## Measured cost reduction

[Raw timing samples](2026-11-19-dmt-check-api-summary.json) retain the evidence.
Before means the working-tree source snapshot immediately before this API
change, including the earlier crossing-region planning improvement.

Strict Pauli standalone compression, **full verification enabled in both
versions**, Torch complex128, one CPU thread, seed-321 positive mixtures,
Bloch strength 0.2, input rank `chi+16`, zero cutoff and no extra diagnostics.
After warmup, old/new order alternates over three paired measurements.

| L | chi | Width | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 24 | 4 | 0.1372 s | 0.0506 s | 2.71x |
| 32 | 32 | 5 | 0.7262 s | 0.1562 s | 4.65x |
| 64 | 32 | 5 | 2.8925 s | 0.3418 s | 8.46x |

All output cores are bitwise identical between versions. Every protected
window was independently checked against the physical mixture oracle;
maximum Frobenius error is 3.62e-15. Runtime drops because the strict sweep
needs two complete panel inventories, rather than two at each bond plus the
final comparison. Bond and density-trace guards stay active.

Replay cost on the L=64, chi=16 tilted-Ising example, two untimed warmup
steps and three timed steps, complex128 and one CPU thread:

| Representation | Full-region checks, warn | Default local checks, warn | Outer checks off |
| --- | ---: | ---: | ---: |
| Pauli | 3.053 s/step | 0.623 s/step | 0.555 s/step |
| Ordinary MPO | 3.840 s/step | 0.842 s/step | 0.785 s/step |

Saved final three-site arrays agree exactly across settings within each
representation. These are limited timing comparisons, not a claim of
physical or exact long-time dynamics; the known trace drift is retained.
Current explicit full Pauli checking actually scans triples. Earlier dated
timings with `check_regions=True` on default Pauli triples predate that fix
and do not measure the same verification workload.

## Skill and validation

Updated the repository [DMT skill](../../../.github/skills/mpo-dmt/SKILL.md),
its operations reference, UI metadata, catalog and upload manifest. The
reference covers current check defaults, warning/error records, larger-window
budgets, SVD policy, benchmark setup and the limits of energy comparisons.
It links measured findings rather than promoting a single result to a guarantee.

Final domain/public-API selection: **487 passed, 1 skipped, 1 expected
failure, 1 deselected**. The deselected installed-version metadata test was
run in the initial selection and fails because installed Pepsy metadata is
0.5.0 while the project and HEAD declare 0.6.0; no environment changes were
made. Subsequent focused check-cost tests: **15 passed**. Those extra focused
cases cover disabled full-check metadata and the retained scalar trace checks.

Regressions verify absent full-panel contractions under fast defaults,
opt-in triple verification, exactly two full inventories per strict sweep,
transactional rejection of post-split corruption, trace checks on the first
ordinary-MPO sweep, and no relative-density trace test for traceless operators.
Two existing relaxed-error tests explicitly request full scans because those
assertions measure physical-region errors. Their tolerances and references
are unchanged. Skill validation, catalog validation, Ruff, links and
whitespace checks pass. No full-suite/GPU claim.

The upstream audit was refreshed with the unchanged installed versions:
NumPy 2.5.2, SciPy 1.18.0, Quimb 1.15.0, Autoray 0.8.4,
Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Symmray 0.4.1.dev15+g0374aaa3c and Torch 2.9.1. The Quimb `array_split`
signature/dispatch route is unchanged. Official Quimb/Autoray/Cotengra and
Symmray sources were checked; the Symmray array-doc URL remained unavailable.
**Adopt:** remove redundant strict-sweep scans, use local replay defaults and
retain cheap trace checks. No dependency upgrades, new shims or approximate
decompositions. Changes remain uncommitted.
