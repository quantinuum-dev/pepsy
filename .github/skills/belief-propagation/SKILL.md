---
name: belief-propagation
description: 'Belief-propagation tensor-network contraction and its loop / cluster / generalized corrections inside pepsy (the `pepsy.bp` subpackage). Use when the user asks to run, build, extend, wrap, or debug: 1-norm BP (`one_norm_bp`) or 2-norm BP for a quimb tensor network; the loop cluster expansion / generalized-loop correction (`loop_cluster_expand`, quimb `contract_gloop_expand`); the loop series expansion (`contract_loop_series_expansion`); disordered-memory / relay-BP convergence robustness (`relay_bp`, `RelayBPResult`); generalized belief propagation (GBP / region graphs, `RegionGraph`, `gen_region_counts`); message reuse / warm-starting; or questions about whether a loop correction needs BP to converge, the cluster-vs-series distinction, the Midha–Zhang free-energy cluster expansion + Kotecký–Preiss error bound, stochastic (MCMC) loop corrections, or how pepsy BP feeds the tensy DEM decoder. Not for QEC-decoder-specific glue (that is the tensy `bp-decoding` skill).'
argument-hint: 'e.g. "run the loop cluster expansion on a PEPS" or "wrap quimb GBP as pepsy region_bp" or "does this correction need BP to converge?"'
---

# Belief propagation & loop/cluster corrections in pepsy

`pepsy.bp` wraps quimb's belief propagation and adds convergence-robust
improvements. Keep wrappers **thin** over `quimb.tensor.belief_propagation`; the
annotated paper trail lives in [`src/pepsy/bp/REFERENCES.md`](../../../src/pepsy/bp/REFERENCES.md).

## When to use
- Run / wrap / extend BP contraction of a quimb `TensorNetwork` in pepsy.
- Add or debug a **loop / cluster / generalized** correction to BP.
- Reason about **convergence** (does a given correction need a BP fixed point?),
  the cluster-vs-series distinction, or error bounds.
- Expose a clean pepsy BP/loop-correction API that **tensy** consumes for DEM
  decoding (keep decoder glue in tensy — see the tensy `bp-decoding` skill).

## Do NOT use for
- QEC-decoder-specific construction (DEM Tanner graph, syndrome conditioning,
  logical marginals, BP-OSD/MWPM baselines) → tensy `bp-decoding` skill.
- Plain MPS/PEPS gate application or boundary contraction → pepsy core APIs.

## Public API (`import pepsy.bp`)
- `one_norm_bp(tn, *, method="l1bp"|"hv1bp"|"d1bp", max_iterations, tol, damping,
  update, diis, init_messages, ...) -> RelayBPResult` — plain 1-norm BP to a
  fixed point (partition-function / nonnegative contractions, e.g. decoding).
- `relay_bp(tn, *, method, num_relays, max_iterations, gamma_range, tol, damping,
  update, memory_first_leg, init_messages, seed, ...) -> RelayBPResult` —
  disordered-memory / relay-BP: per-node random memory (incl. negative) applied
  around quimb `iterate` on the public `messages` dict (quimb `damping` is a
  *uniform* `(old, new)` hook, so per-node disorder is driven here), relayed
  warm-started legs, best-of returned. Forces `local_convergence=False`.
- `loop_cluster_expand(tn, gloops, *, norm="2norm"|"1norm", combine="prod"|"sum",
  messages, run_bp, max_iterations, tol, damping, ...) -> LoopClusterResult` —
  loop **cluster** expansion (quimb `contract_gloop_expand`). `.expand(gloops)`
  reuses the converged messages; `.messages` exposes them.
- SU/simple-gauge bridge helpers: `simple_update_messages_from_gauges`,
  `d1bp_from_simple_update_gauges`, `run_d1bp_from_simple_update_gauges`,
  `simple_update_bp_residual`, and `norm1_gloop_expand`. Use these for
  scalar 1-norm cluster work with externally supplied simple gauges.
- Result dataclasses: `RelayBPResult` (`.bp`, `.converged`, `.iterations`,
  `.max_mdiff`, `.contract()`, `.messages`, `.snapshot()`), `LoopClusterResult`
  (`.estimate`, `.bp_converged`, `.bp_iterations`, `.expand()`, `.messages`).

## quimb substrate (do not reimplement)
`quimb.tensor.belief_propagation`:
- BP families: **1-norm** `L1BP` / `HV1BP` / `D1BP` (partition-function /
  decoding); **2-norm** `L2BP` / `D2BP` (wavefunction / operator norms).
  **Pick 1-norm for decoding-style nonnegative contractions.**
- Loop corrections (methods on `D1BP` / `D2BP`):
  - `contract_gloop_expand(gloops=C, combine="prod"|"sum")` — loop **cluster**
    expansion (Gray et al. 2510.05647). `combine` is `D1BP`-only; `D2BP` is
    product-only and takes `progbar`. `D1BP.__init__` takes **no** `optimize`.
  - `contract_loop_series_expansion(gloops=C, multi_excitation_correct=True)` —
    loop **series** (Evenbly et al. 2409.03108); more fixed-point-sensitive.
  - `contract_with_loops(...)` — simple loop enumeration.
- Region-graph scaffolding: `RegionGraph`, `gen_region_counts` — Kikuchi region
  topology + counting numbers **only** (no `.run()` solver). A packaged **GBP**
  solver (Tindall et al. 2604.24760) is the natural next wrap → a pepsy
  `region_bp`; check whether it has landed in quimb before hand-rolling.
- Convergence toolkit: `run(max_iterations, tol, diis=True, damping, info=...)`;
  `info` returns `converged` / `iterations` / `max_mdiff`. `damping` is uniform.

## Convergence — the key mental model (keep straight)
Loop corrections split by how much they need a **converged BP fixed point**:
- **Loop cluster expansion** (`contract_gloop_expand`): exact contractions of
  growing regions; messages only close each region's **boundary**. A
  system-covering region is exact and message-independent, but finite loop-only
  cancellations and tree/dangling reductions assume fixed-point messages.
  With arbitrary messages, interpret the result as a boundary-closure cluster
  approximation and sweep cluster size.
- **Loop series expansion** (`contract_loop_series_expansion`): inserts
  `I − m⊗m` projectors that vanish **only at the fixed point**; also combinatorially
  divergent in the thermodynamic limit. **Fixed-point-sensitive.**
- **Free-energy cluster expansion** (Midha–Zhang 2510.02290, not in quimb):
  expands `log Z`, only connected clusters, **rigorous** `|log Z − F̃_m| ≤
  n e^{−d(m+1)}` under loop decay `|Z_l| ≤ e^{−c|l|}` (Kotecký–Preiss). Built
  **around** the fixed point → fixed-point-sensitive but gives an *a-priori error
  bar*. Useful as a diagnostic: estimate the loop-decay rate → predicted error.
- **Stochastic loop corrections** (Sim et al. 2603.08427): MCMC-sample the
  `Z = Z_BP × loop-factor` factorization → **unbiased**, statistical (not
  truncation) error.
- **GBP** (Tindall et al. 2604.24760): region messages; structural fix (absorb
  loops into regions), but generalized messages can be **harder to converge**.
- **Relay-BP** (`relay_bp`): the *dynamical* fixed-point hardener (disordered
  memory + relay). Complements the above; does **not** guarantee convergence.

## SU gauges, Gray's correction, and quimb `norm_gloop_expand`
- Corrected point from J. Gray: for the **same closed scalar/norm TN**,
  converged simple-update gauges correspond to a BP / super-orthogonal fixed
  point up to message-gauge normalization. Thus tree-like correlations are
  trivial after `normalize_simple(gauges)`, and `autoreduce=True` is valid.
- Do **not** claim converged SU gauges generally fail tree reductions. The real
  caveats are: gauges must be converged for the same projected scalar network;
  borrowed/open-PEPS gauges are only a warm start; signed amplitude networks can
  be outside the natural nonnegative 1-norm BP regime.
- Gray identified the actual quimb bug we reproduced: in
  `TensorNetworkGenVector.norm_gloop_expand(gauges=...)`, `nfactor` returned by
  `normalize_simple(gauges)` is already the norm scaling, so the final formula
  should effectively be `nfactor * sqrt(loop_product)`, not
  `sqrt(nfactor * loop_product)`. On a two-site tree, current quimb produced
  `z**2 == exact_norm`; the fixed formula gives `z == exact_norm`.
- For scalar BP residual tests from SU gauges, the tight convention is often
  `message_power=1.0`; the symmetric default `sqrt(gauge)` is the clean
  round-trip convention for mapping opposite messages back to an SU-like gauge.

## Gotchas
- `D2BP` = one super-site per lattice site (reaches exact at moderate cluster
  size); `D1BP` on a raw `peps.H & peps` (double the tensors) needs larger `C`.
- Raw DEM factor TNs are nonnegative and good for D1BP, but their factor graph
  can be huge and mostly tree-like. Small generalized-loop corrections may
  barely change rare logical-sector probabilities; use exact/MPS references
  and inspect log-likelihood margins before treating BP+LCE as a decoder.
- `norm="2norm"` + `combine="sum"` is rejected (D2BP is product-only).
- Keep `pepsy.bp` out of the lazy top-level namespace (import `pepsy.bp`); do
  **not** edit `src/pepsy/__init__.py` for it.
- Tests: `tests/test_bp_relay.py`, `tests/test_bp_cluster.py`. Env: py312
  (`source ~/envs/py312/bin/activate`, `NUMBA_CACHE_DIR=/tmp/numba_cache`).

## References
Full annotated list: [`src/pepsy/bp/REFERENCES.md`](../../../src/pepsy/bp/REFERENCES.md).
Core: Gray et al. 2510.05647 · Midha–Zhang 2510.02290 · Evenbly et al.
2409.03108 · Tindall et al. 2604.24760 · Sim et al. 2603.08427 · Müller et al.
2506.01779 · Alkabetz–Arad PRR 3 023073 · Tindall–Fishman SciPost 15 222.
