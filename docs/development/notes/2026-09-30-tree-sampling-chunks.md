# 2026-09-30 — Optional TreeSampler chunks

The user requested 1,000-shot chunks for the CuPy complex128 Tree DMRG
chi=256 roughening run, retaining 8,192 total samples. The unchunked sampler
failed at its first-child transfer: a `(256,256,256,256)` environment occupies
64 GiB, and CuPy's transpose/reshape requested another 64 GiB while roughly
73 GiB was already allocated. Splitting shots alone cannot remove that tensor.

`TreeSampler(chunk_size=None)` preserves the default. A positive integer
enables both shot batching and exact child-index tiling of the density
environment. Tiles target at most 1 GiB, or one child-index slice if larger;
this is an intermediate budget, not a total-memory guarantee. Parent indices
are grouped as `aAcd`, and each tile is released before descending the tree.
There is no truncation or CPU fallback. Uniform random draws are generated
once in the existing order and sliced across batches; finite-precision
threshold effects remain possible. Symmray's already sequential sampling is
unchanged. The examples CLI exposes `--tree-sample-chunk-size` only for tree
runs, records explicit values, and omits the inactive default from run identity
so older live MPS parents remain compatible with later children.

## Dependency audit

Installed: Quimb `1.15.1.dev75+g4112e304a`, Autoray
`0.11.1.dev9+g1291702f9`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev11+g1a3481803`, CuPy `14.0.1`, Torch `2.11.0`.
Inspected CuPy's installed `_flatten_transpose`, which transposes then reshapes
and can allocate a contiguous copy. Classification: **adopt** existing native
einsum and Autoray concatenation; **defer** dependency changes and unrelated
contraction/canonicalization work. No installed libraries were modified.

Consulted the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray repository](https://github.com/jcmgray/symmray), and
[CuPy einsum API](https://docs.cupy.dev/en/stable/reference/generated/cupy.einsum.html).
The Symmray Abelian-array documentation page was unavailable; installed source
and repository information were used instead. No upstream feature replaces
the need to bound this sampler's explicit intermediate.

## Validation

- Sampler/public API/package layout selection: 101 passed, 2 skipped; failures
  were a sandbox worker termination in the native Symmray test and the known
  installed package version 0.4.0 versus source 0.5.0 mismatch. The Symmray
  test passed when rerun outside the process-restricted sandbox.
- Final focused chunk/backend selection on cuda:1: 17 passed, including
  NumPy/Torch/CuPy tiled transfers against a dense reference, seeded samples
  and Born probabilities, physical-root handling, and invalid chunk sizes.
- Example sweep and tree layout/output tests: 63 passed. They check default
  compatibility, opt-in identity/forwarding, and exactly eight shots split
  into batches 3+3+2 at each depth in a small saved-output run.
- Ruff is not installed in the selected environment. Syntax and whitespace
  checks passed; no full-suite claim is made.

Runtime logs and provenance are recorded under
`/tmp/pepsy_examples_runs/launch_tree_dmrg_chi256_gpu1_20260930_dt01_depth60_chunk1000/`.
The earlier failed output directory is preserved.
