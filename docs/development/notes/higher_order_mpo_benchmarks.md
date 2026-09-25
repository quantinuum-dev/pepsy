# Finite higher-order MPO accuracy benchmark

The maintained regression is
[`tests/test_mpo_benchmarks.py`](../../../tests/test_mpo_benchmarks.py). It uses
a four-site open chain with varied one-site, nearest-neighbor, long-range, and
shared-support terms. It compares the tensor-network construction with three
dense small-system references:

- first-order left-to-right Lie--Trotter;
- a finite p=2 cluster expansion containing each exact two-site correction and
  products of corrections on disjoint bonds;
- the exact dense matrix exponential, used only as the accuracy oracle.

The test suite also replays the Trotter gate stream through
`MpoOptimizer` in both `mode="svd"` and `mode="mpo"`, plus a small DMRG/FIT
replay. This verifies the gate transpose convention and compares the result
with the independently embedded dense Trotter product.

Torch and JAX parameterized `MPOBasis` constructions are differentiated and
checked against central finite differences. The JAX finite-difference test is
marked `slow` because tracing the full optimal construction is substantially
more expensive than the numerical MPO tests; the existing JAX-JIT smoke test
remains separate.

The slow large-chain regression uses the transverse-field Ising family

```text
H(J, hx, hz) = J sum_i Z_i Z_{i+1} + hx sum_i X_i + hz sum_i Z_i
```

with `L=32`. One `MPOBasis` is reused for two different `(J, hx, hz)` sets,
which checks that parameter rebinding changes coefficients without recompiling
the term topology. Orders 2 and 3 of the cached MPO exponential are compared
with first-order Lie-Trotter and second-order Strang Trotter MPOs. A four-
substep Strang MPO supplies the reference, so this test remains outside the
dense `2**L` regime. `MpoOptimizer` replays the Trotter gate streams with
`mode="mpo"`, zero truncation cutoff, and a fixed working bond dimension.

The extended slow regression adds an independent `hy` field and a long-range
three-site coupling

```text
gxyz sum_i X_i Y_{i+1} Z_{i+3}
```

to the same family. It exercises MPO Algorithms 1, 2, and 3 at `L=32` and
replays one-, two-, and non-contiguous three-site Trotter gates through
`MpoOptimizer(mode="mpo")`. The direct MPO backend now accepts arbitrary
one-dimensional gate supports; the SVD and DMRG paths intentionally retain
their existing one-/two-site contracts. Because the three-site replay is more
expensive, it uses `chi=8` and a two-substep Strang reference.

The benchmark records Frobenius errors for `order=1, 2, 3` and checks the
expected Taylor and first-order Trotter convergence. It does not assert that
Trotter or the cluster expansion must have a particular error ordering: they
are independent baselines with different approximation structures. Larger
timing and memory studies run through the maintained external harness
`../pepsy_examples/higher_order_mpo/benchmark.py`, so normal package test
execution remains deterministic and lightweight. It records cold and cached
timings, Python allocation peaks, process RSS, raw/reduced stored-block counts,
and final bond dimensions for each order, mode, and storage policy.

## History-plan implementation

Algorithms 1--3 keep the paper's order and coefficients, but their structural
work is compiled separately from numerical tensor evaluation. Algorithms 1 and
2 use a topology-only elimination plan keyed by source and target histories.
Algorithm 3 uses batched insertion plans: each site and insertion-position
pair stores valid left/right history lists and base local-block indices, then
executes one backend-native batch of physical matrix products. It therefore
does not materialize the full order-`N + 1` history reference or a scalar
record for every left/right history pair.

This follows the useful MPSKit/TensorKit design principle of fusing local MPO
tensors and separating structural bond changes from numerical data. The plan
cache stores no Torch/JAX arrays or autodiff graph; coefficient rebinding
recomputes numerical local blocks while reusing the symbolic plan. A scalar
reference test in `tests/test_mpo.py` covers the batched Algorithm 3 path for
orders 1--3.

## 2026-08-31 upstream compatibility audit

The active environment reports Quimb
`1.15.1.dev37+gdf03dbe79` at `df03dbe7989fe19eeb78ca78ea19a87b44da631a`.
The inspected Quimb APIs are `MatrixProductOperator` construction with
`shape="lrud"` and `MPO.compress_all(max_bond=None, cutoff=1e-10,
canonize=True, ..., mode="auto", inplace=False, **compress_opts)`. Pepsy's
final compression continues to use the narrower `MPO.compress` boundary, so
this upstream surface is classified **defer** for the progress work. The
current Quimb changelog's new deterministic `sdc` compression is likewise
left opt-in through existing `compress_opts`; Pepsy's progress feature only
observes the stage and final bond sizes.

Focused validation for this change: the new progress/timing regression and
the related MPO compression/backend tests pass (`21 passed`); Python compile
and Ruff checks are clean.

## 2026-08-31 native Symmray dense-order audit

The active Pepsy environment reports Quimb
`1.15.1.dev38+gfcb3998f9`, Autoray `0.11.1.dev1+gc56f64427`, Cotengra
`0.8.3.dev6+g08fe1a3a1`, and Symmray `0.3.2.dev6+ga17699db6`. The installed
Symmray API probe confirmed `AbelianArray.to_dense(index_maps=...)` and
`unfuse_all()`. A native MPO contracted by Quimb is first fused into charge
sectors; calling its raw Symmray `.to_dense()` without maps returns packed
sector order, which is not necessarily the original computational basis.

Pepsy now adopts the public `index_maps` API at its Quimb MPO boundary. The
returned `PepsyMatrixProductOperator` remains an ordinary Quimb
`MatrixProductOperator` subclass, but restores the physical basis order for
`to_dense()` and retains the basis metadata after numerical compression clears
the semantic history attachment. The `unfuse_all()` route remains a narrow
fallback when metadata cannot be recovered. This is classified as an
**adopted upstream API with a Pepsy compatibility shim**; no installed
dependency was modified.

Focused validation includes interior one-/two-site native terms, higher-order
native histories, and the post-compression native output. The direct dense
and native outputs now agree in computational-basis order, while the MPO
tensors remain native Symmray arrays before explicit dense conversion.

## 2026-09-01 backend execution audit

The active environment reports Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev6+g08fe1a3a1`, and Symmray
`0.3.2.dev6+ga17699db6`. Autoray's `ar.do(..., like=...)`, `ar.to(...)`,
`ar.infer_backend(...)`, and registered `array`/`zeros`/`stack`/`tensordot`/
`linalg` dispatches were inspected in the installed environment and against
the upstream documentation.

The adopted boundary is: structural channel/history plans keep NumPy integer
indices and masks, while all numerical local blocks, generated identity and
zero blocks, sparse virtual transforms, Algorithms 1--4, and final ordinary
Quimb MPO arrays follow the requested `to_backend=` converter. Quimb's
`apply_to_arrays` remains the final compatibility check after materialization
and optional numerical compression. Empty private sparse tensors now retain a
backend reference so they cannot silently materialize as NumPy.

Native Symmray `symmetry=` compilation remains a separate NumPy block-sparse
path and is still rejected together with `to_backend=`; extending that route
would require a native Symmray backend conversion design rather than treating
Symmray blocks as ordinary dense Torch/JAX arrays.

Focused validation covers Torch for every canonical history mode and the
`sparse`/`block_sparse`/`reduced` storage policies, plus JAX block-sparse
execution and final `chi` compression. This is classified as an **adopted
Autoray dispatch contract with a narrow empty-sparse compatibility fix**; no
installed dependency was modified.
