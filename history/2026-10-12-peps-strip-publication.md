# 2026-10-12 — PEPS sweep and global publication review

- Scope: review, validate, commit, and push the pending PEPS optimizer work.
- Branch / baseline: `develop` / `192561a`.
- Commit status: this entry accompanies the reviewed implementation commit;
  the user authorized publication to `origin/develop`.

Reviewed strip batching, exact target ownership, ALS spectral/CG selection,
boundary guess reuse, run controls, and global differentiation against the
API documentation and numerical regressions. Earlier review fixes are
included; see the [implementation note](../docs/development/notes/2026-10-12-peps-strip-global-alignment.md).
The publication review corrected a stale opt-in label in the changelog.

Fresh validation uses the existing isolated validation interpreter; no shared
dependencies were changed. Fresh results:

- Combined sweep/strip, PEPS driver/batching, boundary convergence/engine,
  full-update, global batches/JAX/safeguards, and public/package selection:
  **790 passed** in 383 seconds.
- Standalone global optimization, fermionic boundaries, gate ordering, and
  environment reuse: **75 passed**.
- Default smoke: **94 passed**.
- Ruff and `git diff --check` passed; local links in the changed API guides
  and implementation note resolved.

These selections overlap. The full-suite fail-fast audit was manually
interrupted during JAX cluster compilation after 245 seconds: **571 passed,
2 skipped**, no failures reported. Both skips require unavailable `mpi4py`.
This is a partial audit, not a full-suite pass. It used one BLAS/OpenMP thread
per library; the 790-test selection used the environment's default threading.

The remote advanced with documentation-only changes through `59c8c2d`;
publication integrates these without rewriting either branch's history. Long-run
fidelity/runtime comparisons remain unmeasured; JAX/Symmray ALS support is
outside this patch.
