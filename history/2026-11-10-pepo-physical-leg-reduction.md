# 2026-11-10 — Gate-dependent physical legs in PEPO QR/LQ

- Scope: implement the user's grouping choice in Pepsy: retain acted physical
  legs in reduced factors and leave untouched legs on Q; joint actions retain
  both. Preserve the preceding twelve-site default work and existing archives.
- Branch / baseline: `develop`, `add66d9`, with the earlier twelve-site changes
  already uncommitted. Nothing staged, committed or published in this session.

## Implementation

`prepare_reduced_bond_pair(which='upper'|'lower')` now groups the untouched
operator leg directly onto Q. Omitted `which` preserves the all-physical
compression layout; explicit `both` retains both legs. Ordinary/prepared
cluster gates and native PEPO full updates select the grouping automatically.
The lower-level cluster gate accepts an explicit `(upper_gate, lower_gate)`
joint target with `which='both'`; no stream fusion or truncation reordering
was introduced. Native one-sided Symmray PEPO gates remain unsupported.

The reduced target and its reconstruction share one implementation. Full
update traces the physical spectator on Q without creating a separate bra
index for it. Prepared program keys retain physical-index/layout distinctions;
the cache regression now includes a repeated same-leg update after growth
and selector changes. SU gauge extraction/inverses, amplitude, solvers, PSD
policy and source barriers are unchanged. Scalar metadata records the selected
physical layout/target dimensions; Eli configuration and pair logs expose it.

## Validation

Python 3.12 and dependency versions match the preceding
[twelve-site audit](2026-11-10-twelve-site-cluster.md). Adopt existing public
Quimb QR/LQ, tensor contractions and native backend dispatch; no dependency
change, installed-source edit or compatibility shim.

- Broad affected BP/compression, native full update, paired gates, API/layout:
  **510 passed, 35 failed, 39 deselected** in 75.37 s. All failures are SU
  warm-start calls to unavailable upstream `Tensor.multiply_index_diagonals_`.
  Loading unchanged HEAD `_su_warmstart.py` reproduces that AttributeError.
  Tests selected by the parameter name `simple-update` had been excluded;
  remaining named SU tests and one paired SU case still expose this issue.
- After adding the joint G/adjoint oracle and physical-layout assertions,
  **32 targeted tests passed**. The new matrix covers NumPy, Torch CPU/CUDA,
  complex64/complex128, unequal upper/lower dimensions, reversed site order,
  exact gated operators, dense weighted costs at truncation, and selector
  mismatch rejection. Existing tests cover AOT/eager derivatives, mixed-side
  rank growth, cache freshness and native CUDA/CuPy full updates.
- Native Symmray gauge/reduced-compression regressions: **3 passed**.
- Eli loop-cluster workflow: **38 passed**, including the final metadata
  assertions (17.07 s).
- Full repository fail-fast: **334 passed, 2 skipped, 1 failed**, in 52.01 s,
  at the previously recorded native fermionic long-range-hopping readout
  failure in `test_bp_symmray.py`. Full log:
  `/tmp/pepsy-one-sided-full-suite-20261110.log`. The suite is not green.

## Size and bounded timing evidence

A bulk D6 qubit PEPO target changes from `(24,4,4,24)` to `(12,2,2,12)`;
the four-index environment likewise has sixteen times fewer entries. These
are shape reductions, not measured full-trajectory speedups.

Synthetic CPU Torch complex128 pilot: same 4×4 D2 state, SU weights, lower
two-site unitary, cap2, four ALS sweeps, Hermitian-only metric, greedy paths,
one thread. One warmup followed by three repetitions per configuration;
times include preparation, fresh environment, solve and gauge writeback.
The baseline explicitly retains both physical legs, reproducing the former
grouping. No physical output snapshots or production simulation are involved.

| Patch | Former/new median ms | Former/new normalized local error |
| --- | --- | --- |
| Six sites | 12.916 / 11.999 | 0.300646 / 0.302014 |
| Twelve sites | 12.719 / 11.904 | 0.320497 / 0.322646 |

Raw trials: `/tmp/pepsy-one-sided-reduction-20261110.json`. The reduced target
and environment each have sixteen times fewer entries (1024→64 and 4096→256).
The selected outer subspaces differ at finite bond cap; the slightly higher
pilot errors are retained. Smaller arrays do not establish universally faster
execution or better physical accuracy. Actual 10×10 full-run accuracy/runtime
remain unmeasured; existing saved campaign results are unchanged.
