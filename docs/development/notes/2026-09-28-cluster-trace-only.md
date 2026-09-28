# Trace-only connected-cluster evaluation (2026-09-28)

## Scope and method

On Pepsy `develop` baseline `4e398e4`, an uncommitted scalar path was
added to compiled MPO and Pauli PEPO cluster products and to a single
`PauliPEPOBasis`. For a connected site set `C`, it evaluates the ordered
local target `W_C` and its normalized trace `m_C = Tr(W_C)/d^|C|`.
It obtains the scalar connected residual `k_C` by subtracting every proper
partition into smaller connected clusters. The full normalized trace is the
sum of products of `k_C` over partitions of the lattice into supported
connected clusters. A first-available-site subset recursion shares repeated
subproblems, without listing every cluster collection.

The factor order and all finite-lattice directed/parallel bond occurrences
are retained. Local target representatives can be reused under the existing
verified Hamiltonian/geometry plan because trace is invariant under site
permutation. Cached data contain only topology. The default state budget is
100,000 subproblems and raises if exceeded. It never drops collections or
uses SVD, tree factorization, MPO/PEPO assembly, or boundary contraction.

This is the complete *chosen-order* cluster expansion trace. It is not
necessarily the trace of a representation after local bond truncation,
collection truncation, PEPO tree-rank capping, or boundary approximation.
It is a full bosonic trace; partial and graded fermionic traces are not
implemented. Local dense matrices still grow as `d**p`, and the subset
recursion can itself be large at high order.

## CPU measurement

Uniform open 5×6 X + ZZ model, coefficients 0.2/0.3, order four, step 0.01,
NumPy CPU with one BLAS thread. Timing used `perf_counter`; memory is
`tracemalloc` Python peak from before compilation through the first trace
call, and excludes native allocations. These are separate fresh processes
within one script, not a head-to-head assembled-operator benchmark.

| Route | Compile | First trace call | Python peak | Normalized trace |
| --- | ---: | ---: | ---: | ---: |
| Pauli PEPO basis | 0.390 s | 1.728 s | 27.88 MiB | 1.0002805400051584 |
| Graph MPO product | 1.342 s | 1.405 s | 31.19 MiB | 1.000280540005091 |

The two routes agree to about `6.8e-14` on this observable. The scalar
trace avoids the expensive numerical graph MPO assembly documented in
[the earlier 5×6 record](2026-09-28-graph-mpo-5x6-numerical.md).
No timing ratio against a full PEPO or converged MPO is claimed.

## Validation

- Independent two-factor noncommuting 2×2 checks at orders two through four
  compare both scalar routes with constructed MPO/PEPO traces. At order
  four they also match the trace of dense global ordered exponentials.
- An independent periodic 2×2 order-four reference includes both bond
  occurrences per length-two periodic edge and C4 symmetry reuse.
- A finite state budget raises; zero-step traces normalize correctly;
  a mixed host-identity/Torch-value MPO retains gradients.
- Single-basis coefficient overrides give fresh trace values. Two-site
  ordered products match dense Torch parameter/step gradients and
  JAX `jit(value_and_grad)` values/gradients.
- Broad cluster/MPO/PEPO/API/layout selection: 247 passed, then the
  complete trace-only file passed 11 after two added cases. Full Ruff,
  relative documentation links and whitespace checks passed. See the
  [session handoff](../../../history/2026-09-28-cluster-trace-only.md).

An additional CPU Torch Dynamo probe of the complete two-site MPO
trace call failed under both full-graph and ordinary capture in
Autoray dtype dispatch (torch.dtype had no split/name attribute).
This is a current compiler limitation of the local target path; eager
Torch autograd and JAX JIT remain validated. No compiler fallback
or full Torch graph-capture claim is made.

The numerical path uses existing local matrix exponential and backend
dispatch behavior; no installed dependency, contraction algorithm,
compression, or canonicalization implementation was changed.
