# 2026-09-30 — Boundary amplitudes by default and proposal-only draws

- Scope: user explicitly requested removing the standalone exact-amplitude
  default, adding an option to skip amplitudes, and explaining cap convergence.
- Branch: `develop`; prior task baseline `f01f596`, current HEAD `5316cb0`
  after concurrent work. This session did not stage, commit, or push. Concurrent
  work staged shared sampler files during validation; the index was preserved.
- Supersedes the compatibility-default decision in
  [the preceding amplitude handoff](2026-09-30-peps-boundary-amplitudes.md).

## Behavior

`PepsSampler` now defaults to `amplitude_mode="boundary"`. `"exact"` is explicit.
`"none"` skips every amplitude evaluator in serial, grouped, chunked, and
streamed sampling. It returns `ps=None`, accumulated proposal probabilities,
zero log averaging weights and uniform normalized weights. Diagnostics identify
`weight_kind="proposal"`; `log_mean_weight=None` avoids inventing a norm
estimate. Reading unavailable log amplitudes raises clearly.

With no explicit amplitude cap, boundary amplitudes inherit χ′. If neither is
set, the boundary sweep is uncapped but still uses cutoff. Proposal-engine
auto selection is unchanged: no χ/χ′ still selects exact conditionals; supplied
caps select a boundary proposal. The amplitude method never falls back to a
full-network exact contraction. Large caps and vanishing cutoff/solver errors
recover q = normalized |Ψ|²; q does not recover complex phase or global scale.

Changing the default exposed boundary overflow/underflow for extreme physical
tensor scales. Both amplitude methods now share cached normalized physical
slices. Boundary contraction keeps physical exponent metadata separate from
backend float32 arithmetic and contracts only its final one-dimensional
boundary through the public Cotengra zero-aware scaled contraction. Quimb's
generic final-contract options do not accept `check_zero`; use the final 1D
tree's public `contract` method instead. No installed dependency was changed.
The preceding active-task upstream audit remains applicable.

The roughening CLI exposes `--peps-sample-amplitude-mode none`. Its selected-time
diagonal observables then average proposal draws uniformly. Amplitude fields
are omitted from snapshots and `amplitude_evaluated=False` is saved. Prior
3×3 experiment data/notebook are preserved; no experiment rerun was requested.

## Validation evidence

New tests forbid amplitude evaluators in none mode, compare identical seeded
draws/q across modes, check all draw interfaces on NumPy/Torch, and enumerate
a small system to show larger χ/χ′ recovering Born probabilities at cutoff zero.
Existing exact-plan tests now request exact mode explicitly. Extreme physical
scale tests cover both exact and boundary modes, including zero amplitudes and
large exponent metadata.

The amplitude-focused selection passed 68 checks. The downstream PEPS,
observable, and sweep selection passed 91 checks; both saved-schema regressions
passed again after the stability fix. Changed-file and full Pepsy Ruff passed.
The broader CPU suite result is recorded below after completion.

A JAX GPU probe failed in existing future-environment/conditional construction,
before amplitude evaluation (cuBLAS autotuning failure in Quimb-MPS; a DMRG
proposal comparison also disagreed). CPU JAX validation is run separately;
this task makes no JAX GPU compatibility claim. An obsolete pre-fix test run
was interrupted after 202 passes; its two failures were tests assuming exact
amplitudes by default and have been made explicit reference tests. Both passed
in their focused rerun. The installed/project version metadata mismatch remains
an independent environment failure, not addressed by changing installations.
