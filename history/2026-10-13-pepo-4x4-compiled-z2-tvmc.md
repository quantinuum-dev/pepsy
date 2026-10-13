# 2026-10-13 — Live 4×4 PEPO validation and compiled flat-Z2 tVMC

- Scope: user requested live 4×4 `D=3` PEPO fitting/cap tests and compiled
  PEPS tVMC with flat Symmray Z2, boundary MPS or CTMRG, cutoff zero. Later
  asked whether Quimb `balance_bonds_()` works for native Symmray PEPS.
- Branch / baseline: `develop` / `43dfc4a`.
- Commit status: working-tree edits; nothing staged, committed or published.

## Implemented

- [Cap probe](../examples/tfim_pepo_chi_probe.py) streams flushed JSON for
  fits and explicit independent cap diagnostics, using a direct 16-qubit
  TFIM oracle. Completed two steps at `(18,72)`, two fixed steps at `(81,324)`,
  and failure recovery from `(1,2)` to `(16,32)`.
- [Retained evidence](../docs/development/notes/2026-10-13-pepo-4x4-d3.md)
  and JSON distinguish metric validity from contraction accuracy. Fixed
  `(81,324)` reports dense loss to roundoff, with exact output norm; smaller
  valid caps produce measurable drift. No automatic healthy-step probes.
- [Native Torch amplitudes](../src/pepsy/vmc/torch/amplitude.py) export
  rank-only boundary-MPS **inference** graphs. Exact contraction accepts
  compiled functional parameters; compressed functional/gradient calls retain
  stabilized original contractions because export drops their custom VJPs.
  [Log Jacobians](../src/pepsy/vmc/torch/sr.py) pad/chunk fixed exact batches.
  A narrow missing-`allclose` batching-rule shim preserves the batched forward
  graph and uses output-wise VJPs for affected complex-SVD backward calls.
- [TDVP](../src/pepsy/vmc/torch/tdvp.py) adds explicit real/imaginary
  coordinates, rotating the complex force before real projection. This
  handles active, generally nonholomorphic SVD compression. Holomorphic
  defaults and the paper's uncentered minSR remain available.
- Binary lazy pair flips preserve bosonic Z2. The
  [flat-Z2 example](../examples/tvmc_flat_z2.py) uses rotated TFIM
  `H=-sum XX-sum Z`, shared native operators/connections/sampler/solvers,
  complex128 flat tensors and real-coordinate midpoint tVMC.

## Measured and validated

- Live NumPy 4×4 `D=3` fits completed: fixed `(81,324)` costs about 109 s per
  step, versus 24 s at `(18,72)` on this session. Timings are observations,
  with some overlapping probes, not a controlled throughput comparison.
  Exact norm after two low-cap steps is `1.015075104`; fixed-cap norm is one
  to roundoff. Dense step-two losses are `0.100640445` versus `0.090595760`.
- `tests/test_peps_pepo_4x4.py`: **2 passed**, 69.82 s. One-axis full-PEPO
  fit checks dense improvement, norm/fidelity, rank and input preservation;
  smaller lattice checks the independent Strang oracle.
- Final compiled-TDVP selection: **24 passed** in 36.35 s, including genuinely
  truncated 3×3 flat-Z2 boundary MPS, explicit coordinate finite differences,
  sufficient-cap CTMRG values/Jacobians and second-order integration checks.
- Related compile, TDVP, VMC API/estimators, PEPO/growth/timing and public
  API/layout selection: **210 passed, 4 deselected**, 70.30 s. The four known
  NetKet-dependent checks remain excluded because the environment contains
  only an incomplete NetKet namespace (see the prior handoff).
- Initial CUDA **Inductor** on RTX A5000: 3×3 flat-Z2 `D=4`, boundary `chi=2`,
  complex128 cutoff zero; compiled amplitudes and *native* coordinate Jacobians
  matched eager arrays. A stronger finite-difference test then found native
  derivatives biased (about `9.2e-6` directional error). Stabilized original
  derivatives match to `1.3e-9`, but export drops these custom VJPs and produces
  the native result again. Consequently compressed Torch compilation is now
  explicitly inference-only; equality with native eager gradients alone was
  insufficient. Exact compiled derivatives remain supported. Dynamo graph-break
  and recompilation/cache-limit warnings occurred; no speedup is claimed.
- Flat-Z2 sampled CLI completed one real-coordinate midpoint step, kept
  `D=4` and its native charge blocks, with residual about `5.0e-8`.
- Dense bond balancing preserved a small state's amplitudes to `3.3e-16`.
  Sparse and flat Symmray Z2 balancing both failed in Quimb's repeated-index
  `einsum('abc,abc->a')`. No native balancing shim or densification was added.
- Read the complete user-linked Symmray GPU/JAX example via a read-only HTTP
  fetch after the web tool could not open it. Its code uses equal-sector flat
  Z2, pack/unpack, JIT/vmap, HOTRG cutoff zero and scaled amplitudes. Its saved
  eager/JIT outputs have different precisions; it is not a TDVP gradient test.
- Existing Pepsy JAX factories validated on CUDA with complex128: bosonic
  3×3 `D=4`, boundary `chi=2` JIT loss-gradient directional error `4.3e-10`
  relative; [fermionic 2×3 example](../examples/flat_z2_jax_amplitudes.py)
  JIT amplitude error `8.7e-14`, compiled gradient error `1.1e-10` relative,
  retaining native Z2 and Fock phase maps. GPU emitted CuDNN minor-version
  autotuning warnings but completed these contractions/gradients; environment
  was not changed. Compile plus validation was about 31 s for the latter.
- Default smoke: **94 passed**, 34.41 s. Ruff and diff checks passed. Later
  focused/source refinements are recorded with their final results below.

## Final checks after the derivative correction

- Full related selection rerun: **211 passed, 4 deselected** in 68.21 s.
  Includes final stabilized finite-difference tests; the exclusions remain
  the same four unavailable-NetKet checks, not numerical tVMC tests.
- Final flat-Z2 compiled-inference/stabilized-derivative tVMC CLI completed
  one sampled midpoint step; the JAX CUDA validation example also completed.
- Final Ruff (`src tests` and the four changed examples), local documentation
  links and `git diff --check` passed. No additional full-suite claim.
- All bounded probes have finished. Evidence is retained in the checked-in
  JSON/note above; working-tree files remain uncommitted.

## Upstream audit and limits

- Environment unchanged: Quimb `1.15.1.dev91+gb3bf45a50`, Autoray
  `0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Cotengrust
  `0.2.1`, Symmray `0.4.1.dev18+g44fc5ae53`, Torch `2.14.1+cu126`, JAX
  `0.11.2`. Installed export, PEPS constructors and linalg policies inspected.
  Official Quimb/Autoray/Cotengra sources checked; Symmray ReadTheDocs was
  inaccessible, so its official repository and installed source were used.
- **Adopt:** native flat Z2 arrays, Torch export/vmap/compile, native boundary
  MPS/CTMRG and the existing coordinate/SR machinery. **Compatibility shim:**
  only the unsupported complex-SVD `allclose` VJP batching rule.
- **Defer:** compiled Torch CTMRG/HOTRG, native Symmray PEPO fitting, analytic native
  symmetry gauge projection, small-o sparse Jacobian and production long-time
  dynamics. The dense PEPO and native flat-Z2 examples are separate validated
  paths; no automatic representation conversion is claimed.
- Current flat constructor requires equal sector sizes and rejects odd total
  `D=3`; the requested 4×4 `D=3` PEPO evidence uses dense arrays. Compiled flat
  tests use `D=4`. Cutoff zero disables threshold truncation; the boundary cap
  can still truncate and must be converged independently.
- SVD derivatives are validated on the tested resolved charts, not arbitrary
  degenerate cuts. No shared dependency edits, long background jobs, previous
  benchmark restarts, commits or pushes. Full repository suite not run.
