# 2026-10-05 — Shared MPS and stabilizer trajectory execution

- Scope: user requested comparable trajectory maturity for MpsOptimizer and
  StabilizerMpsSimulator.
- Branch / baseline: develop / `7a01b05`.
- Commit status: local working-tree changes; no commit or push requested for
  this follow-up.

Both frontends share local dispatch/continuation, scheduling and memory
helpers, and execution diagnostic annotation. STN now accepts local
`memory_budget` with tableau/record allowance, performs retention preflight,
continues counted prefixes at caps and respects accelerator MPI worker
defaults. Frame Kraus outcomes share normalized expectations. A reproduced
rare importance-sampled branch failure at probability 1e-30 is fixed without
changing forced-projector defaults.

See the [audit and implementation evidence](../docs/development/notes/2026-10-05-stabilizer-trajectory-parity.md)
and [public STN guide](../docs/api/optimizers/stabilizer_tn.md). Final shared
domain selection: 464 passed / 38 skipped; dedicated parity selection:
39 passed / 2 CUDA skips. Actual Torch CPU and NumPy are validated; MPI policy
has unit coverage only. Ruff and whitespace checks passed. Ordinary MPS GPU
gate/SVD and cross-parent Kraus batches remain capability-gated and do not
batch STN physical gates. No complete GPU performance parity is claimed.

Full-suite check with `MPLBACKEND=Agg` and `--maxfail=2` stopped at 1017 passed,
12 skipped and two Hamiltonian failures. Exported clean `7a01b05` under `/tmp`
reproduced the same counts, tests and `SVD_real requires a real Torch tensor`
error: `test_ham_builder_converts_generic_mpo_to_configured_backend` and
`test_ham_builder_automaton_preserves_shared_structure_on_backend` in
`tests/test_ham.py`. This is not a clean full-suite result. The initial run
without Agg aborted in the macOS plotting backend; the headless repeat avoided
that environment issue. Logs: `/tmp/stn-parity-full-agg.log` and
`/tmp/stn-parity-baseline-full.log`. Final backend-signature selection passed
9 tests; documentation links, Ruff and whitespace checks passed.
