# Square routing, reference compression, and reports — 2026-09-29

## Implemented contract

Explicit square layouts outside the specialized NN Pauli contract now route
the graph PEPO's virtual wires. Manhattan paths depend only on geometry;
overlapping wires use product sectors, including independent pass-through
states at physical sites. The original A/B/C union interaction plan still
defines every cluster and residual. Auto layout retains the old selection.

Reference compression chooses capped local tree SVD subspaces from exact
residuals once, explicitly on the host. Replay uses U-dagger times the fresh
residual and fixed U tensors, keeping ranks and bases fixed. Projection
happens before routing/materialization. It defines a differentiable
approximation, not a derivative through rank adaptation. The explicit plan
owns immutable host constants, not old reference autodiff graphs. Local
reference reconstruction errors are measured directly, avoiding cancellation
in differences of squared norms. They are not global/current error bounds.

Reports count actual exponential inputs/batches, complete product requests,
lower square contractions and graph partition products. Counts live in one
call; no numerical data is read or transferred for work counts. Storage is
an estimate before optional final Quimb compression, excluding workspace.

## Compatibility audit

Reused the same active task's primary-source audit recorded in
[factor reuse and graph autodiff](2026-09-29-factor-reuse-graph-autodiff.md).
Rechecked installed versions and signatures without modifying dependencies:
Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124, JAX 0.10.2.

- **Adopt:** existing PEPO public constructor (`shape='urdlbk'`, explicit
  cyclic metadata), public TensorNetwork contraction, Autoray array/matmul
  operations, and Pepsy's existing thin SVD helper for host preparation.
- **No new compatibility shim:** replay performs no SVD/QR and does not
  modify TorchLinalgConfig, vendor libraries, or global registrations.
- **Defer:** global variational PEPO compression, rank-changing derivatives,
  native charge/fermion/string routing, and GPU performance claims.

## Evidence

Dense SciPy references cover crossing disjoint diagonal interactions,
next-neighbor routing through an occupied site, higher-body interactions,
and periodic routing. Torch finite differences check projected objectives
at ordinary and zero parameters, including a branched tree. JAX jit/grad
checks complete projected square materialization. Separate complex64 and
complex128 checks verify dtype and absence of reference-graph retention.
Spies compare reported exponential batches and lower contractions with
actual calls, including the generic fifth-order square path.

A small two-disjoint-diagonal Pauli probe at step -0.07j reduced estimated
square site storage from 16,640 to 3,840 bytes with reference rank two,
while its dense difference was 3.5e-18. This is a scoped example, not a
general compression or speed guarantee. Other tests deliberately truncate
and check nonzero reference error and finite-difference gradients.

See the [handoff](../../../history/2026-09-29-square-routing-compression-reports.md)
for final validation and publication scope.
