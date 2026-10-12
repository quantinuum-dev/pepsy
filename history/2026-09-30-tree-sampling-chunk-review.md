# 2026-09-30 — Review of TreeSampler chunking

- Scope: careful review of the newly pulled TreeSampler chunk implementation.
- Branch / baseline: `develop`, `dacc200`.
- Commit status: review only; this handoff is new and uncommitted. Existing
  PEPS implementation, documentation, and test edits were preserved.

## Finding

- **P2 — Avoid retaining all chunk outputs before concatenation.**
  `sample_arrays` stores every `(configs, probs)` pair in `chunks`, then
  allocates full concatenated outputs while those pairs and the full random
  draw array remain live. Thus a call can run out of memory during final
  collection even when individual contraction batches fit. Preallocating
  backend-native output arrays and filling their slices would remove this
  additional full-output copy without changing RNG order. No fix was made.
  See [collection loop](../src/pepsy/sampling/tree.py).

The density-transfer index contraction and seeded draw slicing matched the
reference checks. No numerical correctness regression was found. Chunking
still retains full output and random draws, and the documented 1 GiB tile
target is not a total memory limit. Smaller chunks repeat shot-independent
environment construction; no production throughput claim is made.

## Validation performed in this review

- Existing shared py312 environment, local `src` first, CPU-only processes,
  one BLAS/OpenMP thread; no environment changes.
- `tests/test_tree_sampler.py`: **50 passed, 3 skipped** (CUDA unavailable).
- `tests/test_public_api.py tests/test_package_layout.py`: **53 passed,
  1 failed**. The failure is the previously documented installed Pepsy 0.4.0
  versus project 0.5.0 metadata mismatch, unrelated to chunking.
- Additional temporary probe: **96 dense cases passed**, covering NumPy and
  Torch CPU, float32/float64/complex64/complex128, binary/ternary plans,
  optional physical roots, physical dimension 3, chunk sizes 1/8/100,
  37 shots, and a 128-byte environment target to force tiled contractions.
  Configurations matched unchunked seeded sampling; probabilities matched
  unchunked results and independent configuration scoring. Maximum absolute
  scoring difference across all dtypes: 1.22e-7.
- Native U1U1 Symmray product-state probe: chunked/unchunked configurations,
  probabilities, and occupation decoding matched.
- Allocation probe: NumPy, 16-site product tree, 32,768 shots, chunk 512.
  Current collection peaked at **13,292,925 Python-tracked bytes** for
  **4,456,448 output bytes**. A temporary preallocated collector peaked at
  **10,369,211 bytes**, returning exactly identical arrays. This is a single
  tracemalloc measurement, not process RSS or GPU memory validation.
  Temporary script: `/tmp/pepsy_tree_chunk_review.py`.
- `python -m ruff check src tests` and `git diff --check`: passed before
  adding this handoff; whitespace checked again afterward.

Installed versions: Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Symmray
0.4.1.dev7+g83fb22865, NumPy 2.5.2, Torch 2.6.0+cu124. Inspected the
constructor/sample signatures and Autoray's Torch concatenation dispatch.
GPU execution and full-package validation remain unverified in this review;
earlier GPU evidence belongs to the
[implementation handoff](2026-09-30-tree-sampling-chunks.md).
