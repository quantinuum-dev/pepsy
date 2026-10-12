# 2026-09-26 — PEPS simple-update benchmark design

- Scope: investigate the requested PEPS real-time engine and explain the
  workflow before implementation. No engine or numerical library changes.
- Branch / baseline: Pepsy `develop` / `80f451a`; existing unrelated edits
  preserved. This handoff is uncommitted and unstaged; nothing published.

## Proposed workflow

- Add a benchmark engine alongside MPS/tree, reusing shared lattice mapping,
  wall-state conventions, Hamiltonian, and ordered Strang gate stream.
  This is a simple-update evolution adapter, distinct from the existing
  variational `PepsOptimizer`.
- Build bond-one geometry with `pepsy.tensors.ps_to_peps`, then install the
  existing site-dependent product vectors. Its ordinary constructor takes a
  uniform coefficient angle, not the benchmark's wall profile or paper angle.
  Apply backend conversion explicitly on the ordinary PEPS tensors.
- Initialize external bond gauges with `gauge_all_simple_`. This is a gauge
  transformation, not preliminary imaginary-time evolution. Apply mapped
  gates through `pepsy.operators.gate_simple` with evolution bond limit D.
- Make normalization explicit: `gate_simple` defaults to `renorm=True`, which
  renormalizes bond singular values. Use `renorm=False` initially for meaningful
  unrescaled norm diagnostics, or implement and validate an explicit scale
  ledger if normalization is subsequently needed for stability.
- For measurements, copy the core, call `copy.gauge_simple_insert(gauges)`
  once, then use the absorbed PEPS. This method mutates and returns gauge
  bookkeeping, not a new PEPS. Do not modify the evolution representation or
  absorb the gauges twice.
- Reuse `pepsy.bp.compute_boundary_expectation`, the compatibility wrapper
  for Quimb's local-expectation path, with normalized local Z expectations
  and independent boundary `max_bond=chi_boundary`. The benchmark imbalance
  is `(sum_outside Z - sum_inside Z) / N`. Its second moment needs separate
  correlators and must not be inferred by squaring its mean.

## Norm and loop diagnostics

- `absorbed.make_norm().contract_boundary(...)` estimates norm squared.
  Report both this scalar and its square root; normalized local observable
  denominators are not substitutes for a separate global norm contraction.
- Quimb `norm_gloop_expand(gauges=...)` returns the norm. If using its
  fixed-point tree reductions, converge gauges on an independent snapshot
  representing the same state and retain scale. Rebuild contraction-value
  caches for each changed snapshot.
- Pepsy `loop_cluster_expand(absorbed, norm="2norm", gloops=4)` estimates
  norm squared and supplies BP convergence diagnostics. Four-site loops
  include square plaquettes; six-site clusters are a possible later level.
  The alternative edge-based loop series has a different cutoff convention.
- Report the BP baseline, corrected estimate, correction ratio, boundary
  estimate, loop cutoff, and convergence residual. These are approximate
  contractions of the same evolved PEPS; loop corrections do not repair
  evolution truncation and need not improve monotonically.
- The BP skill records a historical Quimb norm-scaling bug. The installed
  implementation now uses exponent-aware normalization and passed the scaled
  probe below; do not treat the historical claim as a current failure.

## New validation (CPU-only investigative probe)

- Activated the device-local py312 environment and used local Pepsy source.
  Installed Quimb: `1.15.1.dev66+ge927f06e1`; Autoray:
  `0.11.1.dev3+g1b476b305`; Cotengra: `0.8.3.dev7+g1d7fd333f`;
  Cotengrust: `0.2.1`; Symmray: `0.4.1.dev7+g83fb22865`;
  Torch: `2.6.0+cu124`.
- On a 2x2 product state, applied field/ZZ/field gates with D=4, cutoff=0,
  and renorm=False. State error against dense evolution: 1.35e-15.
  Exact norm squared: 0.9999999999999961; boundary chi=16 result:
  0.9999999999999962 (imaginary roundoff 1.7e-16).
- All four boundary local Z values matched dense values (0.509563673574003).
  Snapshot regauging converged in three iterations with residual 4.7e-14,
  preserving the state to 8.3e-16.
- Full 2x2 loop region: Quimb returned norm approximately 1, and approximately
  3 after multiplying the state by 3. Pepsy loop-cluster returned norm squared
  0.9999999999999961 with BP converged. This checks API/scaling conventions,
  not accuracy or cost for a truncated 5x6 PEPS.
- No production job was changed. No GPU performance or full suite claimed.

## Open design choices

- Evolution D and boundary chi remain unspecified. GPU, larger lattices,
  truncation accuracy, and loop costs require later validation.
- User clarified that dt is user-selected and results should be saved at
  selected times. Keep dt and save_times independent. Proposed scheduler:
  shorten a step when necessary to land exactly on the next requested time,
  regenerate the gates for that actual interval, then resume the selected dt.
  Record actual intervals/times; never globally replace dt with 0.2 or relabel
  nearby times. Step clipping is a proposed implementation detail.
- Asked whether the intended correction is loop-cluster or edge-based loop
  series; no answer yet when this note was written. Loop-cluster is the
  proposed first implementation, not an accepted user decision.
- References reviewed: [Quimb real-time simple update](https://quimb.readthedocs.io/en/latest/examples/ex_real_time_simple_update.html)
  and [2D algorithms](https://quimb.readthedocs.io/en/latest/tensor/tensor-2d.html),
  plus the installed APIs and [BP skill](../.github/skills/belief-propagation/SKILL.md).
