# 2026-09-25: qMERA Torch full-graph energy

`torch.compile(fullgraph=True)` of the ordinary compiled local-cone loss
entered Autoray while generating pair gates, then Cotengra's lazy expression
wrapper at evaluation time. The opt-in Torch cost freezes each cone's
Cotengra `array_contract_tree(...).get_path()` before tracing, executes its
binary steps with Torch `einsum`, and constructs each scheduled Pauli gate once
per Hamiltonian evaluation. No full-state network is materialized.

The trace supports built-in qMERA pair ansatzes and `rxx`, `ryy`, `rzz`, using
explicit `GateSpec.torch_pauli_words`. Other gate families must provide matching
Pauli rotation metadata or use the standard compiled loss. Native graded
Symmray arrays remain on their original path. The Torch callable also works
without `torch.compile` and accepts `normalized`, `energy_per_site`, and `real`.

Installed versions: Quimb 1.15.1.dev66, Autoray 0.11.1.dev3, Cotengra
0.8.3.dev7, Symmray 0.4.1.dev7, Torch 2.6.0. We inspected the installed
Cotengra path/tree and lazy-expression implementations, Autoray dispatch,
and the [PyTorch full-graph contract](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html).
The earlier [upstream audit](qmera_2d_retained_hierarchy_2026_09.md) covers
Quimb/Cotengra/Autoray/Symmray official sources in the unchanged environment.
Classification: **adopt** Cotengra's public contraction path and Torch's
native operations; **defer** Inductor until Python 3.12 development headers
are available. No dependency shim or installed-package edit was made.

A four-site periodic TFIM with four ZZ and four X terms matched the standard
loss and every parameter gradient with `backend="aot_eager", fullgraph=True`.
The example notebook completed the same check. In one CPU measurement of
12 warmed forward-plus-backward calls after the final static-slot change,
median times were 49.9 ms (standard compiled loss), 21.5 ms (new Torch eager
loss), and 77.8 ms (AOT eager).
These exclude path construction and AOT compilation. AOT eager validates
capture here; it was slower, so no compiler speedup is claimed. A minimal
Inductor scalar compile failed at C compilation because
`/usr/include/python3.12/Python.h` is absent. AOT eager needs no such header.

JAX was rechecked on the same four-site, eight-term TFIM. The test explicitly
calls `jax.jit(jax.value_and_grad(loss)).lower(params).compile()` and compares
the XLA result with eager JAX energy, every eager gradient, and Pepsy's direct
local-cone energy to `1e-10`. Pepsy's Optax Adam update is decorated with
`@jax.jit` in `src/pepsy/solvers/gradient.py`. A fresh notebook execution
completed 25 JAX Adam steps from `-4.0000000000` to `-5.1873339644`; its
weighted local terms sum to the same final energy. The complete qMERA suite
passed 96 tests after this addition.

The end-to-end API audit found that `QMeraEnergyOptimizer.compiled_loss_fn`
previously forwarded `torch_fullgraph=True` to an evaluator that did not
accept it. The optimizer now prepares the Torch-only callable for both
`compiled_loss_fn` and `run(compiled=True)`; incompatible JAX or uncompiled
solver requests fail early. Running all full-graph tests together exposed
Torch Dynamo tracing Python list deletion with symbolic indices. The current
contraction plan assigns permanent IDs to intermediates before tracing and
appends results without deleting list entries. All eight full-graph tests
then passed together, followed by the complete 98-test qMERA suite and a
fresh TFIM notebook execution. The public API example now defines its Torch
backend, parameter cast, and backend-matched cones explicitly.
