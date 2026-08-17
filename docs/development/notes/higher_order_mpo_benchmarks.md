# Finite higher-order MPO accuracy benchmark

The maintained regression is
[`tests/test_mpo_benchmarks.py`](../../../tests/test_mpo_benchmarks.py). It uses
a four-site open chain with three non-commuting nearest-neighbor terms and
compares the tensor-network construction with three dense small-system
references:

- first-order left-to-right Lie--Trotter;
- a finite p=2 cluster expansion containing each exact two-site correction and
  products of corrections on disjoint bonds;
- the exact dense matrix exponential, used only as the accuracy oracle.

The benchmark records Frobenius errors for `order=1, 2, 3` and checks the
expected Taylor convergence of the MPO path. It does not assert that Trotter
or the cluster expansion must have a particular error ordering: they are
independent baselines with different approximation structures. Larger timing
and memory studies should run outside the package repository so normal test
execution remains deterministic and lightweight.
