# 2026-10-08 — Dependency-aware PEPS gate scheduling

- Scope: user approved smarter commuting-gate scheduling and single-site
  absorption, then requested rectangular corner-case checks, commit and push.
- Branch: `develop`. Work began at `6de7551`; the concurrent MPS/backend work
  was committed separately as `f31a625` during this task and is preserved.
- Publication: this task's scheduler, integration, tests, and docs form a
  separate commit. No unrelated code changes or dependency installations.

## Implemented

- Opt-in `gate_order="smart"` recognizes exact small-matrix entry patterns
  for `span(I, Pauli product)` and arbitrary diagonal gates. Commutation uses
  Pauli parity on shared sites; unknown overlapping gates retain dependencies.
  Disjoint fixed gates commute. Trainable/unsupported/selected/non-neighbor
  gates remain barriers. Native block arrays are excluded from this policy.
- Native comparisons transfer only scalar decisions. Repeated gate identities
  and backend/dtype/device constants are reused during compilation; every run
  reclassifies gates so in-place mutations are observed. No numerical
  near-commuting threshold or NumPy transfer of PEPS/device gate arrays.
- A dependency graph provides legal next pair updates. The greedy scheduler
  pulls in required single-site ancestors, prefers continuing the current
  strip, and compares row-first and column-first traversals by refinement
  blocks then strip changes. It is not a global optimality search and can
  require quadratic dependency storage for heavily overlapping queues.
- Adjacent fixed same-site gates with matching native signatures fuse in
  execution order. Transpose/dagger gate options disable fusion, because
  transforming an already-fused product reverses the intended operation order.
  Two-qubit gates never fuse. Complete original IDs and input/compiled/fused
  counts remain available; the original queue is restored after failure too.
- Full-update refinement can span intervening local rotations inside its
  strip, applying them to the retained exact target. Repeated bonds still
  bound target growth. Refinement records include all original gate IDs.
- Rectangular checks exposed NumPy one-qubit gates reaching native Torch
  contraction without conversion. Full-update now converts these small gates
  to the PEPS backend/dtype/device before absorption, matching pair handling.

## Numerical evidence

Independent dense operator comparisons on 2×3 circuits exercise mixed Pauli
axes, reversed gate operands, and fused single-site rotations on NumPy,
Torch, and CuPy, in complex64 and complex128. GPU tests prohibit bulk host
conversion. Dependency tests distinguish XX/YY on the same bond from a single
shared site, retain arbitrarily small non-Pauli perturbations, and observe
mutated gate values on the next compilation.

3×4 and 4×3 full-update tests exercise both orientations, corners, reversed
and repeated bonds, mixed XX/YY/ZZ rotations, in-strip single-site gates,
two refinement passes, exact target bounds, normalized finite output,
dtype/device retention, gate IDs and local fidelity counts. Empty and
single-site-only 3×4 queues include NumPy gate matrices with Torch/CuPy PEPS.
Tests retain original site/user tags while allowing the existing public
gate/normalization bookkeeping tags G/KET.

A 4×4 commuting XX circuit starts from a seeded random product state and
interleaves two RX rotations with each of 24 bonds, repeated three times.
At D=2 and boundary chi=32 with one refinement pass:

| Backend | Column schedule | Smart schedule |
| --- | ---: | ---: |
| Torch CPU elapsed | 6.02 s | 3.12 s |
| CuPy GPU elapsed | 11.06 s | 5.92 s |
| Executed entries per layer | 72 | 39 |
| Refinement blocks per layer | 24 | 8 |

All 24 pair updates and fidelity records remain. The circuit is exactly
representable at this D; dense reference infidelity stays within 9e-16 of
zero. These single-run timings demonstrate this workload only, not a general
speedup. Compiler time was about 3 ms on Torch and 26 ms on CuPy in the smart
runs, including first-use overhead.

The noncommuting 4×4 TFIM reference was also rerun with smart scheduling and
one refinement pass: D=2, boundary chi=32, four real-time steps of 0.1,
`H=-sum ZZ-sum X`, initial `|+>^16`. Final infidelity against the independent
exact Trotter state is 1.65426642e-3 on both Torch and CuPy; norms remain one
within 3e-15. Scheduling preserves the exact circuit but can change truncated
results, so no generic fidelity improvement or global bound is claimed.

Reproducers: `/tmp/pepsy_smart_benchmark.py`,
`/tmp/pepsy_smart_itf_{torch,cupy}.py`; JSON/logs under matching `/tmp` names.
Environment: existing `py312`, Torch CPU and CuPy on RTX A5000, bounded CPU
thread counts. Public API details are in
[the PEPS guide](../docs/api/optimizers/peps.md).

## Validation

An earlier affected run passed 326 tests before the rectangular additions.
Final combined regression: **477 passed**, 98 warnings, 123.37s, including
all 26 smart-scheduler cases. Default smoke: **94 passed**, two warnings,
39.09s. Ruff and `git diff --check` passed. Logs:
`/tmp/pepsy_smart_final.log`, `/tmp/pepsy_smart_smoke.log`.
The preceding task's 24 reproduced baseline BP
failures remain outside scope; no new clean full-suite claim is made.
