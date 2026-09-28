# 2026-09-28 — Fixed cluster JIT gradient boundary

## Scope and evidence

This is a CPU check of fixed MPO and Pauli PEPO coefficient/time gradients under
JAX JIT and Torch compilation. Structural `compile_exp()` reuses geometry; it
is distinct from tracing a tensor computation. The installed environment has
JAX 0.10.2, Torch 2.6.0+cu124, Quimb 1.15.1.dev66, Autoray 0.11.1.dev3,
Cotengra 0.8.3.dev7 and Symmray 0.4.1.dev7. The earlier
[upstream audit](2026-09-28-fixed-cluster-autodiff.md#backendcompiler-audit)
remains applicable. Installed `jax.jit` and `torch.compile` signatures were
checked at the start of this task. Official [JAX tracing](https://docs.jax.dev/en/latest/tracing.html)
and [PyTorch fullgraph](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)
guides informed the probe boundaries. No installed library changed.

**Measured:** complete scalarized three-site order-three and 2×2 square
order-four fixed MPOs and PEPOs pass `jax.jit(jax.value_and_grad(...))`
against independent dense matrix exponentials at zero coupling, zero time
and nonzero parameters. Two-site complex-time cases pass analytic
value/gradient checks at zero and nonzero coupling. The JIT function returns a scalar array, not an MPO/PEPO object.
These checks establish the finite cases tested; they do not prove 5×6 p=4
compilation cost, speedup, arbitrary topology or JAX object-return support.

**Adopt:** both local matrix exponentials, the fixed MPO split, and an
actual two-site PEPO fixed tree factorization pass Torch
`torch.compile(backend="aot_eager", fullgraph=True)` with correct represented
values and parameter/time derivatives at zero and nonzero values. This backend checks graph
capture and backward, not production code-generation speed.

**Compatibility shim:** the fixed MPO identity uses a native Torch tensor
factory; PEPO scalar alignment between two native Torch tensors uses tensor
arithmetic for dtype promotion. These avoid Autoray dtype lookup during
Torch tracing while preserving device and promotion. PEPO tree shape
products use Python `math.prod` rather than NumPy scalar operations under
Dynamo. Regression coverage includes a mixed float32/float64 scalar and its
derivative.

**Defer:** complete Torch MPO/PEPO builders cannot currently be captured as a
single full graph. Probes stop in installed Autoray dtype conversion and translated
`expand_dims` before Quimb materialization is reached. A default partial-graph Torch compile
also failed in Autoray dtype lookup/graph-break handling, so it must not be
advertised as a supported whole-builder route. Eager Torch autograd remains
covered by existing dense-reference tests. A future numeric-only array/PyTree
assembly boundary would need its own design and broader geometry checks.

JAX traces Python construction once for a given input signature; cache/report
mutation inside a JIT wrapper is trace-time metadata, not a per-invocation
runtime report. Read reports from ordinary evaluator calls.

## Validation

- New JIT regression file: 10 passed, CPU-only.
- Affected fixed cluster, correctness, expansion, MPO and compression suites:
  183 passed in 102.46 seconds, CPU-only. This includes all ten JIT cases and
  the PEPO tree shape-product fix.
- Full-suite and GPU compilation were not run in this task. The earlier full
  suite in the fixed-autodiff handoff predates these edits.
