# 2026-10-01 — Factor becomes the TreeSampler default

The user requested making factor the default after confirming that large-bond
trees are the intended workload. Baseline: `develop` at `25e9e18` (published).
Classification: **adopt** the existing exact factor route as the constructor
default. No factor contraction, grouping threshold, numerical scaling,
truncation policy, dtype, cache budget or workspace target was changed.

New dense `TreeSampler(state)` instances select `strategy="factor"`.
`strategy="standard"` remains an explicit alternative. Native Symmray sampling
retains its existing block-sparse algorithm. The default `chunk_size=None`,
128 MiB cache, 512 MiB workspace, `backend="auto"` and `threads=None` remain
unchanged. Old serialized instances lacking strategy/cache attributes retain
their original route through the existing `getattr(..., "standard")` fallback.

The [API guide](../../api/sampling/tree.md) now documents the default and
uses implicit factor in the chunk-1,000 native example. Its memory caveat and
standard alternative remain: the
[small-chunk measurements](2026-10-01-tree-factor-small-chunks.md)
showed large-bond benefits and a small-bond counterexample. That earlier
note's recommendation to keep standard was the decision at that time;
the subsequent explicit user request changes the policy, not those measured
results. There is no automatic strategy selection based on bond dimension,
and factor is not claimed to minimize memory on every tree.

## Environment and upstream audit

Rechecked installed versions before the default change: Quimb
`1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev7+g83fb22865`, NumPy `2.5.2`,
Torch `2.6.0+cu124` and cupy-cuda12x `14.1.1`. Inspected the `TreeSampler`
constructor and `_FactorSamplingContext(sampler, cache_bytes, workspace_bytes)`
signatures. Dependencies and the factor execution route are unchanged, so
reuse the same-task [upstream audit](2026-10-01-tree-sampler-correctness-resume.md#upstream-and-environment-audit)
and prior independent backend/gradient evidence. No compatibility shim or
upstream-library edit was introduced.

## Validation

The existing factor default-settings matrix now constructs the sampler without
a strategy argument, checks its factor selection, and compares sampled
configurations and Born probabilities with explicit standard and dense
statevector references. It covers NumPy, Torch CPU/CUDA and CuPy, single-site
and multi-site trees, physical roots, chunked/unchunked calls and persistent
random-generator draws. Other factor tests now name their standard reference
explicitly, preserving independent comparisons. Tests specifically inspecting
the standard density/vector contractions select standard explicitly.

Focused validation: **512 passed, two skipped**, four existing compatibility
warnings in 101.25 seconds. Ran `python -m pytest -q -ra -o addopts=''` on
`tests/test_tree_sampler_validity.py`, `tests/test_tree_factor_sampler.py`,
`tests/test_tree_sampler.py`, `tests/test_tree_canonical_regions.py`,
`tests/test_tree_entropy.py`, `tests/test_tree_unitary_stability.py`,
`tests/test_public_api.py` and `tests/test_package_layout.py`, with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1` in the activated environment. Torch
CPU/CUDA, CuPy and native Symmray paths were exercised. The skips are the
known CuPy float32 subnormal-source limitation and a check needing two CUDA
devices. Ruff (`python -m ruff check src tests`), documentation links and
whitespace checks pass. Log: `/tmp/pepsy_tree_factor_default_validation.log`.
Publication is recorded in the
[handoff](../../../history/2026-10-01-tree-factor-default.md).
**Defer** actual-checkpoint performance and Torch/autograd peak-memory
measurements; the prior synthetic timing/allocator results remain scoped to
their recorded workloads. Full-package validation is outside this focused
default change.
