# 2026-09-29 — Tree DMRG alignment with the roughening angle sweep

- Scope: user requested review of TreeOptimizer DMRG with the same 12-angle
  diagonal-wall roughening setup as the MPS run. Review only; no implementation
  changes, production launches/restarts, staging, commit, or push.
- Baselines: Pepsy `develop` / `a233a40`; examples `main` / `efb7696`.
  Existing edits in both repositories were preserved. This handoff is new and
  uncommitted; numerical probes and their output are under `/tmp`.
- Workflow: repository tree-optimizer skill, FIT environment reference,
  current implementation/API guides, and focused tests. No dependencies changed.

## Verified physical alignment

Tree selection is `--mode tree --tree-optimizer-mode dmrg`; tree alone defaults
to direct compression. The roughening runner and sweep share the lattice,
Hamiltonian, initial-state, angle, and observable builders with MPS.

All 12 child commands and saved-configuration identities were checked for
delta_theta = 3*pi*k/88, k=0,...,11, with explicit lattice 5x6, chi 512,
dt 0.05, depth 200, 8192 samples, complex128, and entropy/XX disabled.
The resolved geometry is open boundaries, diagonal x+y canonical mean-field
wall, snake logical labels, J=-1, hx=1, hz=0. The physical polar angle is
theta_min + delta_theta. The tree layout defaults to alternating-xy and
uses the inverse physical-coordinate map to preserve these logical labels.
The 5x6/depth200/sample8192 settings are explicit comparison settings, not
the generic CLI defaults. No production-size evolution was launched.

## Effective tree FIT controls

The runner forwards fit_n_iter=8 (package standalone default is 4), block size
2, two block warm-up iterations, fit_min_iter=2, fit_patience=2, auto tolerance
(1e-9 for complex128), auto cutoff (1e-12), rsum2, guess-src, random strength 0,
and fit_init_seed=0. RL becomes inward-outward; traversal defaults to auto.
Generic tree DMRG on paths with more than two nodes uses two block-growth
iterations followed by one-node refinement.

These controls are not numerically identical to MPS's iteration semantics:

- Each TreeFIT iteration includes both inward and outward passes. Each MPS
  FIT iteration contains one direction, alternating R/L. Eight therefore
  allows 16 directional tree passes versus eight MPS passes; local block
  counts also depend on the different geometries.
- Tree patience=2 requires two stable comparisons; MPS patience=2 currently
  counts two norm samples, requiring one stable comparison. Phase transitions
  reset both stopping histories.
- Tree's one-node fast path remains enabled. It solves a one-node active
  region exactly and is not the MPS adjacent-physical-pair shortcut. Two
  distinct leaf qubits span at least their leaves and shared ancestor.
  MPS-only single-pair budget/shortcut flags do not control TreeFIT.
- In the small exact-reference probe, all 168 multi-node FIT updates took
  five iterations with block trace (2,2,1,1,1) and reason rtol. The 288
  one-node updates used single_node_exact. This is not production convergence
  evidence. Zero extra initialization strength does not make SRC nonrandom.

## Integration findings

1. **P2 — Tree implicitly caps CPU threads despite omitted --threads.**
   Examples `magnetization/engines/shared.py:5339` passes
   `threads=(1 if args.threads is None else args.threads)` to TreeOptimizer;
   the sampler does the same at line 5496. Pepsy's `_thread_ctx` uses
   threadpoolctl around tree operations (`optimizer.py:1747`). A scoped Torch
   probe measured 4 threads before, 1 inside, 4 afterward, while metadata's
   torch_threads remained null. This caps CPU BLAS/OpenMP, not CUDA kernel
   parallelism. No GPU performance impact was measured; one thread can help
   small CPU operations, but it does not match the user's unforced policy.

2. **P2 — stabilize_unitary is recorded but not implemented for this tree
   runner path.** The shared tree constructor/replay branch does not forward
   or implement the MPS stabilization option. A 3x2, chi1, eight-step,
   dt0.05 probe at delta_theta0.3 produced identical vectors with the flag
   enabled/disabled (max difference 0) and norm 0.9961930926407466 in both.
   This is expected truncation scale loss, not proof of invalid normalized
   observables; the concern is advertised effective behavior and long-run
   numerical scale, not a demonstrated production failure.

3. **P2 — Optional diagnostics are not forwarded or saved for tree.**
   `--finite-check --fit-overlap-diagnostics` are accepted and recorded as
   true, but the TreeOptimizer instance retained fit_finite_check=False and
   fit_overlap_diagnostics=False. Its run branch forwards only progbar/mode
   (`shared.py:2830`), so replay finite_check also remains False. These tree
   capabilities exist in the package. The metadata getter is additionally
   restricted to MpsOptimizer (`shared.py:6945`), leaving fit_diagnostics null.
   Both flags default off, so ordinary default evolution is unaffected.

4. **P3 — Tree quality polling reconstructs accumulated history.**
   `shared.py:1374` uses the compact query only for MPS. Tree norm_diagnostics
   (`optimizer.py:7935`) converts all norm events twice and scans valid events
   on each query. This is growing bookkeeping work, with possible device-to-
   host scalar overhead. No production speedup or memory benchmark was run.

5. **P3 — Some shared optional CLI values are not tree-compatible.**
   `--fit-sweep-sequence R/L` are offered but rejected by TreeFIT's sequence
   normalizer; its default RL works. The tree accepts LR/inward-outward/
   outward-inward programmatically, but the CLI does not expose them.
   MPS batching/target controls likewise are not forwarded to TreeOptimizer.
   These do not affect the reviewed default single-gate, RL configuration.

## New validation

Activated the shared Python 3.12 environment, used local Pepsy source, hid
CUDA, and limited CPU libraries only inside validation subprocesses. No live
job's environment or thread settings were changed.

- Package: test_tree_mps_parity, test_tree_fit_messages,
  test_tree_path_execution, test_tree_fit_priorities: **130 passed** (26.13s).
- Examples: test_roughening and test_roughening_sweep selected with
  `tree or delta_theta or child_command or configuration`: **44 passed,
  1 optional CuPy/CUDA skip, 147 deselected** (7.55s).
- A temporary real-frontend probe compared Tree DMRG and exact replay on a
  3x2 lattice, chi64, dt0.05, two steps, all 12 angles. All **36 states**
  (initial plus two evolved states per angle) agreed with minimum normalized
  overlap fidelity **0.9999999999987095**. This validates the small untruncated
  physics/mapping path, not full-size finite-chi accuracy.
- One Torch CPU frontend run verified actual tree defaults, ignored opt-in
  diagnostic flags, and the temporary thread cap. Two additional chi1 runs
  checked stabilization-flag behavior described above.
- No full suite, production GPU benchmark, or production convergence test.

Temporary evidence: `/tmp/review_tree_roughening_20260929.py`,
`/tmp/tree_roughening_audit_20260929_en1xgsdd/summary.json`, and
`/tmp/tree_stabilization_audit_20260929_c48st6an/summary.json`.
The results above retain the important findings if temporary files disappear.

Suggested follow-up is to make thread policy and effective tree diagnostics
explicit in the runner, then decide how to expose stabilization and distinct
tree iteration semantics. Those changes were not part of this review request.
