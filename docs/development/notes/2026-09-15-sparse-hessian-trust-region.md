# 2026-09-15 — Sparse/banded-Hessian trust-region for gauge-style optimizers

- Status: **plan + evidence** (nothing implemented yet in this note's scope).
  Hand-off for Codex. Primary work is in **pepsy** (generic); **gaugy** only
  supplies structure; the **gaugy_examples** tutorial notebook already exists.
- Environment: `~/envs/genpy` (Python 3.12) — the only interpreter with
  `gaugy`/`pepsy` importable for the downstream checks below.

## The idea (why)

Gauge/time-compression losses such as gaugy `loss_mpsg` (`C_bulk`) plateau
under first-order solvers (`LD_VAR2`/L-BFGS): the landscape is an
ill-conditioned, curved valley, so a single step scale zig-zags. A
**Hessian-based trust-region** step (Newton-like, damped) uses the curvature to
take correctly-scaled steps and breaks the plateau.

Two measured facts make this cheap and worthwhile for these problems:

1. **Curvature helps here.** On the notebook `C_bulk` problem (L=4, dV=4,
   `color_gmat`, 401 real params), `scipy-trust-krylov` reached
   `C_bulk ~ 2.1e-6` vs `LD_VAR2 ~ 1.4e-5` (~7x lower), gradient-based
   `LD_VAR2` plateaued.
2. **The Hessian is banded.** `C_bulk = sum_j c_j`, and each slice term `c_j`
   depends only on `{V_j, G_{j-1}, G_j}`. Therefore
   `H_ab != 0` only if `a,b` share a slice, and the Hessian is
   **block-tridiagonal along the slice chain**. Verified numerically
   (dV=5, cparam_zyx): off-neighbor blocks are **exactly** 0
   (`max|H| off-band = 0.000e+00`). A block-tridiagonal Hessian is recovered
   from **O(1) colored Hessian-vector products** (Curtis-Powell-Reid), not
   `n` backward passes.

So: exact curvature at ~constant HVP cost -> practical trust-region. This is
the core trick behind Eli Chertkov's LIBO (`time_compression`, branch `libo`,
`docs/libo_sparse_gn.md`, `docs/gpu_trust_region.md`): block-local Gauss-Newton
`2 J^T J` + banded/GPU Moré-Sorensen trust-region.

## What pepsy already has (baseline)

`src/pepsy/solvers/gradient.py` already supports scipy Hessian methods,
including complex params:

- `_SCIPY_HESS_METHODS = {"trust-exact"}` (full n×n Hessian) and
  `_SCIPY_HESSP_METHODS = {"trust-krylov","trust-ncg","Newton-CG"}` (HVP).
- `_evaluate_hessian` (~L463) via `torch.autograd.functional.hessian`
  (O(n) backward passes) and `_evaluate_hessp` (~L480) via `vhp`.
- `_build_param_specs` (~L289) already flattens complex params as [re, im]
  into the real vector (so `color_gmat` matrices are handled).
- `_run_scipy_lbfgs` (~L1021) wires `hess=`/`hessp=` and warns when the full
  Hessian is used with `n > 100`.

So `solver="scipy-trust-exact"` / `"scipy-trust-krylov"` already work; the
missing piece is **exploiting sparsity** so `trust-exact` costs O(colors), not
O(n).

## pepsy improvement (generic, no domain leak)

Add a **structured/sparse-Hessian trust-region** capability. pepsy only ever
sees `f: R^n -> R`, the flat vector, and (optionally) a **Hessian sparsity
declaration** over those indices. It must never learn "slice"/"gauge"/"chain".
This is standard sparse-Hessian estimation via graph coloring
(Curtis-Powell-Reid / Coleman-Moré; the analogue of scipy's `sparsity` arg for
finite-difference Jacobians) — reusable by MPS/tree/PEPS or any partially
separable loss.

Three acceptable input levels (any/all):
1. raw sparsity pattern (CSR/bool mask or `(i,j)` pairs over the n flat
   indices);
2. **interaction groups** `list[Sequence[str]]` — key-sets that interact
   (the supports of the additive cost terms); pepsy expands keys->flat via its
   own specs;
3. **auto-detect** — probe the Hessian with a few random HVPs, threshold to
   find the nonzero pattern, then color. Caller passes nothing but the loss.
   (Cleanest separation; use as the default when no structure is given.)

Where it sits:
- New module `src/pepsy/solvers/structured.py` (keep `gradient.py` lean):
  - `groups_to_sparsity(specs, groups)` / accept raw pattern too;
  - `color_sparsity(pattern)` — greedy star/distance-2 coloring of an
    arbitrary graph (not banded-specific);
  - `assemble_hessian(loss_from_flat, x0, pattern, coloring)` — fill only
    on-pattern entries from `#colors` colored `vhp` calls; symmetrize; add a
    **validate mode** (assert off-pattern HVP components ~0 so a wrong pattern
    fails loudly);
  - optional `detect_sparsity(loss_from_flat, x0, probes=k, tol=...)`.
- `gradient.py` edits: add `hessian_sparsity=` / `hessian_groups=` (and maybe
  `hessian_autodetect=`) kwargs to `GradientOptimizer` (~L2081) ->
  `_optimize_dispatch` (~L1867) -> `_run_scipy_lbfgs` (~L1021); when present
  with a Hessian method, build `hess=` from `assemble_hessian` and skip the
  n>100 warning (cost is now O(colors)).
- Keep `trust-krylov` as the zero-structure fallback (pure HVP).
- Tests `tests/test_solver_structured_hessian.py`: banded assembly matches
  dense `torch.autograd.functional.hessian` on a toy chain; color count is
  `2b+1`; `trust-exact + sparsity` reaches the same optimum as dense
  `trust-exact` with far fewer HVPs; auto-detect recovers the same pattern.

Important: the shared `GradientOptimizer` API and existing solver behavior must
stay unchanged when the new kwargs are absent (strictly opt-in).

## gaugy hook (optional; supplies structure only)

gaugy just declares its partial separability; it adds **no** optimizer code:
- `ColorAnsatz.hessian_groups()` -> `list[list[str]]` = for each slice j,
  `v_keys(j) + gauge_keys(j-1) + gauge_keys(j)` (the support of `c_j`), ~12
  lines over the existing key helpers.
- Consumer passes `hessian_groups=ansatz.hessian_groups()` (or relies on pepsy
  auto-detect and passes nothing).
- Note: in the natural interleaved key order the coupling reaches the two
  gauges flanking a shared V, so the band is width 2 in interleaved-block units
  (=> ~5 colors), still O(1). Do not hard-code "bandwidth" in pepsy; pepsy
  colors whatever pattern/groups it is given.

## Notebook / tutorial (already added)

`gauge_mps/mpsg/curvature_trust_region_tutorial.ipynb` (in the
`gaugy_examples` repo) is a self-contained pedagogical notebook
(numpy/scipy/matplotlib/torch only): a 1-function 2-variable Hessian warm-up,
gradient zig-zag vs Newton, Rosenbrock convergence by curvature level,
trust-region step sizing (`rho`/`Delta`), and a chain-cost demo showing the
tridiagonal Hessian recovered from 3 colored HVPs. It runs clean end-to-end.
The main `color_mpsg.ipynb` was intentionally left unchanged.

## Validation done so far

- `scipy-trust-krylov` vs `LD_VAR2` on `C_bulk` (L=4, dV=4): 2.1e-6 vs 1.4e-5.
- Banded Hessian proof on `C_bulk` (dV=5): off-neighbor blocks exactly 0.
- Chain-cost toy: 3-probe colored-HVP recovery matches the dense autodiff
  Hessian to machine precision.
- Tutorial notebook executes with zero errors via nbconvert.

## Follow-ups / open questions for Codex

1. Implement `pepsy/solvers/structured.py` + the opt-in kwargs; start with
   **auto-detect** (needs no gaugy input) so pepsy stays fully general, then
   add explicit `hessian_groups`/`hessian_sparsity`.
2. Consider a **Gauss-Newton** variant later (given a residual fn + Jacobian
   coloring, build banded `J^T J`; PSD, one Jacobian, matches LIBO). Requires a
   residual-mode loss (separate, larger change; not needed for the first cut).
3. Trust-region driver: for small blocks scipy `trust-exact` on the assembled
   dense-but-mostly-zero Hessian is fine; a banded Cholesky Moré-Sorensen is
   the scale path (defer; LIBO's GPU version is out of scope for pepsy now).
4. Add the tiny `ColorAnsatz.hessian_groups()` hook + a gaugy test once the
   pepsy API lands, then benchmark trust-region vs `LD_VAR2` on `C_bulk` and
   drop a convergence plot into `color_mpsg.ipynb` §1.
