# 2026-09-30 — Proposal-only sampling by default

- Scope: user requested keeping proposal mode as the default and asked for
  the chunk-size defaults. Changed only the Pepsy amplitude default.
- Branch / baseline: `develop`, `dacc200`. Existing PEPS/chunk edits and
  concurrent tree changes preserved. No staging, commits, or publication.

`PepsSampler` now defaults to `amplitude_mode="proposal"`. Ordinary calls
skip amplitude contraction, retain q, return `ps=None`, and use equal weights.
`"none"` remains an alias; `"boundary"` and `"exact"` require explicit opt-in.
The legacy result-record default stays `"exact"` for BP/manual records;
the direct sampler always passes its selected mode.

Chunk defaults are unchanged: `sample_batch(chunk_size=None)` uses all shots
in one batch, `iter_samples(chunk_size=128)` streams batches of 128, and
`"auto"` remains opt-in. The downstream roughening runner still supplies its
own settings, including chunk size 32; no downstream files were changed.

Updated docs/changelog and made amplitude-specific numerical regressions
explicitly request boundary amplitudes, preserving their existing assertions.
The new default regression forbids amplitude evaluators for serial, collected,
and streamed calls with exact, Quimb-MPS, and DMRG proposals. No contraction
algorithm or dependency changed; prior upstream audit applies.

An initial migration run identified 48 tests depending on implicit amplitudes
(381 others passed); their amplitude requests were made explicit. Final focused
sampler/API selection: 491 passed, 47 JAX cases deselected, one known installed
version mismatch (34.65 s). Smoke: 92 passed and the same mismatch (28.65 s).
The mismatch is installed Pepsy 0.4.0 versus project 0.5.0. Full Ruff and
whitespace checks passed. JAX cases and the expensive 4x4 statistical suite
were not rerun for this default-only change.
