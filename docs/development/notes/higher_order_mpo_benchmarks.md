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
