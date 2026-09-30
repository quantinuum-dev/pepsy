# 2026-09-25: qMERA contraction preflight

`QMeraBuilder.estimate_contraction_cost` builds the same static local-cone
networks used by compiled qMERA energy. It asks Cotengra for a contraction
tree before parameter optimization, reports `log10` of total forward complex
FLOPs and `log2` of peak concurrent elements times `element_bytes`, and primes
unsliced paths in `QMeraContractionPathCache`. Repeated topologies share one
estimate. Normalized losses include numerator and denominator contractions;
unnormalized losses plan numerator contractions only. The returned
`contraction_opt` and `path_cache` should be passed together to compiled Pepsy
loss/optimizer APIs using the same schedule, terms, backend, and gate registry.

The metric is a path estimate for one forward energy evaluation. Total FLOPs
sum local terms. Peak bytes take the largest single local contraction, assuming
cones are evaluated sequentially. The memory estimate excludes autograd tapes,
parameters, optimizer state, and framework allocations. `element_bytes=16`
models complex128; callers should adjust it for their contraction dtype.
Native Symmray block costs are rejected because dense shape estimates would be
misleading. Cotengra slicing is not encoded as a raw path, so sliced trees keep
Cotengra's reusable optimizer behavior instead of priming a raw path.

Installed Cotengra: `0.8.3.dev7+g1d7fd333f`. The implementation uses its
public `array_contract_tree`, `total_flops(dtype="complex")`, `peak_size`, and
`get_path` APIs. Cotengra documents path reuse and persistence through its
`ReusableHyperOptimizer` in the [basics guide](https://cotengra.readthedocs.io/en/main/basics.html)
and [tree API](https://cotengra.readthedocs.io/en/main/autoapi/cotengra/core/index.html).
The notebook passes an in-memory optimizer (`directory=False`); users can opt
into persistent Cotengra cache directories through
`build_qmera_contraction_optimizer(directory=...)`.

The four-site TFIM notebook was executed into `/tmp/pepsy_examples_executed/`.
It prints the estimate before 25 JAX Adam steps and retains its compiled XLA,
term-sum, and Torch AOT energy/gradient checks. A focused regression verifies
that subsequent compiled loss construction uses primed paths without a new
search. A one-term odd `(1, 5)` retained 2D grid also matched direct and
compiled energy using the primed cache. This does not benchmark large systems
or measure runtime peak memory.
