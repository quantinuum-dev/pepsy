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

The benchmark records Frobenius errors for `order=1, 2, 3` and checks the
expected Taylor and first-order Trotter convergence. It does not assert that
Trotter or the cluster expansion must have a particular error ordering: they
are independent baselines with different approximation structures. Larger
timing and memory studies should run outside the package repository so normal
test execution remains deterministic and lightweight.
