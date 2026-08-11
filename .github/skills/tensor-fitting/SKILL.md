---
name: tensor-fitting
description: Implement, review, debug, profile, and document Pepsy FIT tensor-network compression, including FIT.run_gate/run_eff, one-site and two-site sweeps, MpsOptimizer fit/dmrg and mix modes, active bond growth, exact target blocks, timing, complex64 norm stability, and native Symmray U1/U1U1 fermionic behavior. Use for changes under pepsy.fitting or FIT-backed MPS/MPO compression; compose with mps-optimizer for wider gate-stream, layout, measurement, or mode-dispatch work.
---

# Tensor Fitting

## Purpose

Use this skill for the variational compression kernel shared by Pepsy's MPS,
MPO, boundary, and sampling workflows. Keep approximation choices explicit,
preserve native array semantics, and measure performance without weakening
correctness.

Read [references/fit-architecture.md](references/fit-architecture.md) before
editing the solver or its MpsOptimizer integration.

## Workflow

1. Read repository instructions and the task-relevant FIT/MPS documentation.
2. Identify which independent control is changing:
   - local variational block size (`fit_block_size`);
   - sweep direction sequence (`fit_sweep_sequence`);
   - target gate/layer count (`fit_layer_size`);
   - target representation (`fit_target_strategy`);
   - exact target cutoff (`target_cutoff`);
   - output SVD cutoff/`chi`;
   - environment contraction strategy;
   - unitary working-norm stabilization;
   - adjacent single-pair structural convergence.
3. Trace the full call path from public API to `FIT.run_gate`; do not patch only
   the dense leaf when Symmray can reach the same path.
4. Implement native-first tensor operations. Use Quimb `Tensor.split`; never
   call NumPy SVD on Symmray arrays or densify charge blocks.
5. Preserve the gate-window invariant: local circuit FIT changes only
   `[xmin, xmax]`. Use `run_eff` only for an intentionally global fit.
6. Add diagnostics for algorithmic decisions and failed partial work without
   synchronizing full GPU states per local update.
   Use `timing_sync_device=True` only for profiling asynchronous accelerators,
   and separate effective contraction, SVD, writeback, and environment costs.
7. Validate dense complex64/complex128, active bond growth, alternating
   sweeps, exact-target separation, and native symmetry/fermionic cases.
8. Update API docs when defaults, controls, timing fields, or backend status
   change.

## Non-negotiable invariants

- Target construction and output compression are separate error sources.
- Dense layered targets may factor a gate exactly, but must not truncate or
  materialize the state before the output FIT split.
- Two-site FIT may grow visited bonds up to `chi`; it must not globally pad an
  MPS as a side effect.
- U1, U1U1, dual-leg, dummy-mode, and fermionic phase metadata remain native.
- Compression loss is recorded before unitary working-state renormalization.
- The public stabilization control is `stabilize_unitary`; it covers FIT
  and mixed-mode MPO compression without changing the recorded loss.
- Reuse FIT's final canonical center/norm for stabilization; do not sweep the
  same interval a second time.
- Keep `run()` and `run_eff()` fixed-sweep semantics independent of adaptive
  `run_gate()` controls; wrappers may report them as `fixed_sweeps`.
- A failed sweep retains a timing record with `status="failed"`.
- Experimental algorithm names must not dispatch to a materially different
  sequential fallback.

## Validation

Activate `/Users/rezah/envs/genpy` before Python commands on the documented Mac
checkout. Run the smallest relevant tests first, then the MPS/MPO and symmetry
regressions listed in the architecture reference. Finish with Ruff, docs, and
the repository's default smoke suite when the change is cross-cutting.

## Composition

- Use `mps-optimizer` for queue parsing, layouts, control events, and behavior
  shared by non-FIT replay modes.
- Use `pepsy-fermion-operators` when gate construction, Jordan-Wigner order,
  or physical-sector conventions are also changing.
- Use `symdmrg2` only for the separate Hamiltonian ground-state solver.
