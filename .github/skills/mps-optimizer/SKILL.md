---
name: mps-optimizer
description: 'Run, review, debug, or extend pepsy.MpsOptimizer for MPS gate-stream replay, MPO/SVD/DMRG/swap compression, persistent graph layouts, canonical-center diagnostics, normalization, control events, and exact-mode transitions.'
---

# MPS Optimizer in pepsy

Use this skill for `pepsy.MpsOptimizer` and its implementation in
`src/pepsy/optimizers/mps/optimizer.py`. Read the public API documentation at
[`docs/api/optimizers/mps.md`](../../../docs/api/optimizers/mps.md) and the closest
tests before editing.

## Execution modes

- `fit` / `dmrg` / `dmrg1` / `dmrg2` / `dmrg3`: local variational compression;
  `dmrg1` uses adaptive two-site growth followed by one-site refinement,
  `dmrg2` uses its required two-site warm-up (two sweeps by default) followed
  by one-site refinement, and `dmrg3` uses adaptive three-site growth followed
  by one-site refinement. Adaptive rank-growth phases end only when all
  attainable active-bond ceilings are reached; rank stagnation is not an early
  exit. A `dmrg1` window already at those ceilings starts directly with
  one-site FIT. An under-capacity non-adjacent `dmrg1` window requires
  `n_iter >= 3`: two block sweeps plus at least one refinement sweep. Its
  default `fit_patience=2` is a two-sample same-phase norm window, i.e. one
  stable comparison. A two-site window is a structural special case for both
  `dmrg1` and `dmrg2`: perform exactly one two-site update, no one-site
  refinement, then advance to the next gate without consuming the remaining
  `n_iter` budget. Compose with
  [`tensor-fitting`](../tensor-fitting/SKILL.md) for FIT kernel, target, rank
  growth, symmetry, stability, or profiling changes.
- `mpo`: direct non-local gate/MPO replay with bond truncation.
- `svd`: local SVD compression; `swap`: swap-and-split with swap-back.
- `perm`: swap-and-split with lazy logical-to-physical tracking.
- `mix`: transactional FIT with per-step MPO fallback. Two-site FIT grows
  active bonds directly; fixed-rank one-site FIT retains MPO rank warm-up.
- `exact`: fully contracted TensorNetwork replay, without MPS canonical metadata.

Keep `exact` separate from MPS code. When switching from exact to an MPS mode,
rebuild an MPS from the explicit physical indices and canonicalize it. Do not
switch a persistent-layout optimizer into exact mode because that would lose
the physical/logical layout contract.

## Persistent layouts

For repeated evolution, use:

```python
opt.apply_layout("quality")
opt.run()
opt.run()  # reuses the same physical order
```

`logical_order[position]` is the logical site stored at a physical MPS
position. Use `logical_site(position)`, `position(site)`, `remap_sample`, and
`to_dense()` for readout. A persistent layout never swaps the MPS back.

The reorder is free exactly when `p.max_bond() == 1`: rebuild the product MPS
with tensors in the new order and do not call an SVD swap. If the initial MPS
is entangled, raise by default. Only `allow_lossy_reorder=True` may enable the
one-time SVD reorder, and it must use the caller's `cutoff` rather than a
hardcoded exact cutoff. `cap` control events are rejected because they change
the MPS length; measure/reset events keep recording logical labels.

Do not recommend `run(use_layout_finder=True)` for iterated evolution. It is a
deprecated compatibility path that temporarily reorders, runs, and swaps back.

## Canonical-center contract

`optimizer.info_c["cur_orthog"]` is algorithm state, not cosmetic metadata.

- `_current_orthog` may scan only when the state metadata is unknown.
- Pass the tracked range through Quimb `info` arguments and canonicalization.
- Local Pauli expectations should use `local_expectation_canonical` when
  available and move the center from the tracked range to the support.
- Norm diagnostics should canonicalize to one center and use its tensor norm;
  do not replace this with a global doubled-network contraction.
- Local non-unitary scale control normalizes active canonical tensors and adds
  the removed base-10 scale to `p.exponent`.
- Every rebuilt/replaced live MPS needs its known canonical span recorded.
- Temporary target copies (`p.copy()`) must use isolated metadata and must not
  overwrite the live `info_c` dictionary.
- FIT may consume an optimizer-owned guess/target to remove redundant copies,
  but mixed mode must still isolate the complete trial before mutation.
- After FIT, reuse its final center and norm for stabilization and canonical
  metadata instead of scanning or canonicalizing the active interval again.

After a mutation that invalidates canonicality, either canonicalize explicitly
or invalidate the cache before any canonical-only operation. Unitary one-site
gates preserve the isometry structure; non-unitary one-site gates need explicit
recanonicalization.

Do not transfer diagnostic semantics from `MpsOptimizer` to
`MpsStabOptimizer`. The stabilizer coefficient-MPS simulator has its own sparse
normalized-unitary norm-loss contract; read
`.github/skills/stabilizer-tensor-networks/SKILL.md` before changing it.

`TrajectoryEvent` Kraus branches are normalized physical-state updates. The
trajectory runner applies the selected branch with `non_unitary=True`, calls
`normalize()`, and clears `p.exponent` so the next gate sees a truly normalized
MPS. This does **not** turn ordinary-MPS `track_infidelity` (a local normalized
overlap diagnostic) into the STN norm-loss proxy: keep the contracts separate.

## Backend rules

Use Quimb and Autoray APIs. Preserve Symmray arrays and route symmetric target
gates through backend-compatible split/auto-swap paths; never force a dense
NumPy identity into a Symmray contraction. Keep optional Symmray coverage
guarded with `pytest.importorskip("symmray")`.

Native fermionic FIT environments must use graph-planned contraction directly
on the Symmray arrays so dummy-mode conjugate pairs and graded phases determine
the contraction order; do not replace this with an arbitrary pairwise loop or
a temporary `TensorNetwork`. Gate-window FIT keeps a conjugated native working
MPS, contracts real outside overlap environments, applies dual-leg corrections
before local writeback, and resolves odd dummy-mode global phases afterward.
This convention must honor `R`, `L`, and `RL` directly and may reuse compatible
environments across a direction reversal. A non-adjacent fermionic DMRG target
is warm-started by the same chi-capped native auto-swap replay, analogous to
deterministic sector enrichment, then follows the selected block-to-one-site
schedule normally.
`target_cutoff=0.0` may prune only structural zeros using the smallest positive
absolute cutoff; every representable nonzero singular value must remain.
For a direct full-chain native MPS FIT, compare target and guess virtual charge
maps before sweeping. If target sectors are missing, initialize from a target
copy compressed with the requested native output policy and record
`native_sector_initialization`; never seed through dense arrays. Reject the
same initialization on a partial window because replacing its outside tensors
would violate the fixed-boundary contract; let native block updates grow local
sectors there. Treat an empty native effective tensor as a clear
disconnected-sector error, not as a backend decomposition failure.

## Validation

Activate the shared environment first:

```bash
source /Users/rezah/envs/genpy/bin/activate
```

Focused optimizer validation:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/test_optimize_mps.py \
  tests/test_optimize_mpo.py \
  tests/test_symmetric_tensors.py
```

Also run `tests/test_public_api.py`, `tests/test_package_layout.py`, and
`tests/test_sampler.py` for public API/readout changes. Add regression tests
for layout persistence, logical sample/dense remapping, entangled reorder
errors, exact-mode transitions, no-rescan center reuse, exponent accounting,
and control-event bookkeeping.
