# 2026-10-05 — Prefix continuation and GPU gate/SVD batches

- Scope: user's three follow-ups: preserve prefixes at branch caps, batch
  compatible GPU gates/SVDs, and broaden circuit/slow/real-MPI validation.
- Branch / baseline: `develop`, `2039fbc10264b2f17c8f179cfdad05e1e2663f3d`.
- Commit status: working-tree edits only; nothing staged, committed or pushed.
  Earlier precision/memory edits and unrelated existing changes were preserved.

## What changed

- Local automatic MPS runs continue from counted prefixes before a possible
  cap overflow, including Pauli macros and threaded coalescing. Continuation
  is serial; retained output remains count-aware, with count-one leaves.
- Adjacent ascending two-site swap-mode segments can batch gate contractions
  and SVDs across compatible Torch CUDA/CuPy parents through Autoray. Each
  parent keeps its own rank. Ordinary replay owns ledger/metadata updates.
  Retained U slices are copied using the existing differentiable helper so
  they neither pin the whole batch allocation nor detach Torch gradients.
- Fixed MPI configuration hashes for backend arrays nested in plans: actual
  values replace process-dependent Torch storage IDs. NumPy-only fingerprints
  remain compatible with existing checkpoints.
- Added diagnostics, API documentation and deterministic regression coverage.

## Validation and evidence

The [detailed note](../docs/development/notes/2026-10-05-mps-frontier-batching.md)
records upstream choices, benchmark methodology, test results and limits.
The new gate-batch suite passed **44 tests**, including 18-qubit cases,
different per-parent ranks, complex64/128, six cutoff modes, normal and
stabilized Torch gradients (with/without truncation), and storage ownership.
Two- and three-rank real MPI runs passed **32 cases per rank** using the one
available RTX A5000, covering both GPU backends and streamed reductions.

All **32 slow cases passed**, including native 3x4 Hubbard stress cases;
the initial expanded suite also exposed two diagnostics test-double failures,
fixed with defaults for the new optional fields and verified on recheck.
Final consolidated domain run: **933 passed, 2 skipped, 32 slow deselected**.
The skips are unavailable Torch Metal and second JAX devices. The detailed
note distinguishes the expanded initial run from the final passing selection.
Final branch-bound/memory regression run: **89 passed, 1 skipped**, including
zero-probability mixture outcomes under a tight retained-state budget.
The full-suite attempt still stops at the three baseline BP compression
failures, reproduced on clean HEAD during the preceding precision/memory task.
Ruff and whitespace checks pass. A temporary mpi4py installation under `/tmp`
was used; the shared Python environment was not modified.

## Remaining limitations

Batching currently covers one-gate adjacent swap-mode segments. Other modes,
larger segments, routing, native symmetry, normalization/diagnostic options
and multiple local workers retain their existing gate path. Local automatic
continuation does not change low-level or MPI scheduling policies. Real
multiple-device/multiple-node execution could not be tested with one GPU.
Small matrices benefited substantially in the measured gate-stage workload;
larger matrices showed little benefit. These are scoped results, not general
end-to-end speedup claims.
