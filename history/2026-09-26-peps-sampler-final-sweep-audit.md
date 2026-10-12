# 2026-09-26 — Final PEPS sampling strategy and corner cases

- Scope: user's requested final review/implementation of the simple cached
  future + conditioned single-layer boundary sampler.
- Branch / baseline: `develop` / `80f451a`, ahead 8 at start.
- Commit status: working-tree edits only; nothing staged, committed, published,
  or changed in production jobs. Preserved prior work and unrelated changes.

## Implemented

- Simple default (`row_cache_max_bytes=0`); dense transfers remain opt-in.
- Prepare only future-side Quimb environments; lazily build DMRG right-side
  boundaries. Reject missing required caches and unsupported periodic sweeps.
- Natural-log `log_probability`, exponent-stripped explicit likelihood
  contractions, and log accumulation in exact probability evaluation.
- Whole-configuration validation before zero-branch exits, finite option
  checks, and explicit integer-array conversion requirements.
- Documented dimensions, cutoff mapping (χ=marginal_chi, χ′=sample_chi),
  sitewise factorization of joint row draws, and truncation support limitations.

## Evidence

See [detailed audit](../docs/development/notes/peps_sampler_final_sweep_audit.md)
and [current API](../docs/api/sampling/samplers.md).

- Focused: 77 passed (NumPy/Torch/JAX, including two JAX CPU devices).
- CUDA/CPU matrix: 12/12 passed; source preservation and queried proposal
  consistency verified.
- Additional sampler/API/layout: 161 passed, 1 pre-existing version failure.
- Smoke: 152 passed, same version failure (installed 0.4.0 vs project 0.5.0).
- Ruff, whitespace checks, and local documentation links pass.
- Active-truncation before/after samples match; relative q differences <3.1e-15.
- Reproduced χ′=1 losing one nonzero Bell-pair configuration; χ′=2 restores
  both probabilities. Importance weights cannot repair lost proposal support.

## Remaining limits

No claim of a universally positive/support-complete truncated proposal, full
compiled batching, globally overflow-proof amplitudes, or JAX GPU coverage.
Prior JAX cold-start costs remain. Current Quimb emits its existing mode/method
compatibility FutureWarning. Unsupported custom tags still fail the norm builder.
The sandbox helper failed before file access (including apply_patch); authorized
edits used checked replacements through automatically reviewed shell escalation.
