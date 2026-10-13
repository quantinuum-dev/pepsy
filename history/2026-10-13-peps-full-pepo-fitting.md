# 2026-10-13 — Full TFIM PEPO targets and three fitting routes

- Scope: one full bond-two TFIM Trotter PEPO per fit, using existing Pepsy and
  Quimb construction/solver functionality; ALS, column L-BFGS, and full global
  optimization with sequential BRA--PEPO--KET direct compression.
- Branch / baseline: `develop` / `43dfc4a`.
- Status: working-tree edits only; nothing staged, committed, or published.
  Earlier adaptive-cap work and unrelated pending edits are preserved.

## Implementation

- `PepsOptimizer.run_pepo` uses Quimb `PEPO.apply(contract=False)` for the
  untruncated fixed target and a separately applied/contracted/compressed
  copy for the warm start. Existing `SweepOptimizer` and `GlobalOptimizer`
  perform fitting; the gate queue is preserved. Target norms are measured
  once per attempt and reused for fitting and acceptance.
- ALS uses whole-lattice sweeps and DMRG/FIT boundaries with bounded cap
  recovery. `optimizer="scipy"` uses the existing complete-column L-BFGS
  solver, with columns sharing the same fixed full target. Column/global
  fitting retain fixed caps by default.
- Global `mode="direct"` reuses Pepsy `contract_layered`/`CompBdy` sequential
  direct compression inside Quimb `TNOptimizer`. Candidate norm layers are
  BRA/KET; `overlap_layer_tags` independently selects BRA/PEPO/KET with a
  larger overlap cap. Explicit overlap layer validation prevents overlap
  factors from also being folded into KET. The existing native Quimb
  `mode="mps", mode_="direct"` path remains available.
- `contract_layered(preserve_backend=True)` retains the backend scalar for
  autodiff; default Python scalar formatting is unchanged.
- The runnable `examples/tfim_pepo.py` uses Pepsy `rx`, `rzz`, and
  `build_pepo_from_gates`. Onsite half steps do not increase operator bonds;
  each commuting ZZ edge appears once. The complete Strang step is therefore
  exactly represented by this bond-two PEPO. No replacement PEPO builder,
  compression kernel, or numerical optimizer was introduced.

## Numerical diagnosis

The added three-step zero-state test initially failed for ALS at step two.
An initial local overlap exceeded one by approximately `2.17e-8`, persisting
at caps `(32,128)` through `(256,1024)`. Increasing the FIT budget to 40 or
switching to fresh direct guesses did not fix it. Zero boundary cutoff did:
small singular channels had been dropped during one-site FIT initialization,
so increasing chi alone could not recover them. Full-PEPO boundary seeding
and metrics now default to rank-only truncation (`cutoff=0`); explicit
cutoffs retain precedence. Warm-start PEPS cutoff remains dtype-aware.
No acceptance or local validity tolerance was weakened.

## Validation

Reused the isolated validation environment and installed upstream audit from
the [cap-growth session](2026-10-13-peps-failure-growth.md). Inspected installed
Quimb operator application, rotation generators, and boundary layer handling.
Classification: **adopt** existing public operator/compressor/optimizer APIs;
no dependency update or installed-library modification.

- Final full-PEPO workflow suite: **22 passed**. Independent dense matrices
  validate the complete 2×3 Strang operator on NumPy/Torch; independent gate
  replay validates its 4×4 zero-state action. ALS, column L-BFGS, and global
  fits use the lazy target and agree with dense acceptance fidelity. All three
  complete three steps from zero with dense circuit fidelity at least 0.99.
- Both Pepsy sequential-direct and native Quimb direct directional Torch
  gradients pass centered dense finite-difference checks. Tests verify
  disjoint layers, separate norm/overlap caps, raw backend scalar preservation,
  rollback at a ceiling, retained growth, operator/input preservation, and
  explicit cutoff precedence.
- Broader boundary/global/sweep/growth/timing/public API selection:
  **503 passed**. Log: `/tmp/pepsy-pepo-final-domain.log`. This is not the
  full extended suite. Existing Quimb/NumPy deprecation and rank-deficient QR
  warnings remain; gradient claims are limited to the tested fixtures.
- Ruff on `src tests` and the example, documentation link checks, and
  `git diff --check` passed.

Efficiency versus native Quimb boundaries and a long higher-D 4×4 trajectory
remain unmeasured. Native Symmray full-PEPO fitting is explicitly unsupported
pending separate validation. No stopped benchmark or persistent job was
restarted. Public usage is documented in the
[PEPS guide](../docs/api/optimizers/peps.md#one-full-pepo-step-per-fit).
