# Direct MPS autodiff compilation, 2026-10-08

## Scope and implementation

This supports Gaugy's fixed-probe global MPSEngine optimization. Dense direct
replay keeps the supplied target state separate from the evolving V state.
There are no installed-dependency edits or Symmray algorithm changes.

- **Adopt:** public Quimb `MatrixProductOperator`, `gate_with_submpo_`, packed
  network reconstruction, and existing direct state compression.
- **Compatibility shim:** for dense Torch/JAX two-site direct gates with
  `cutoff=0`, construct an exact fixed-rank operator MPO from identity and gate
  entries. Quimb's `gate_nonlocal` otherwise performs a separate operator SVD
  with a default data-dependent cutoff, even when state cutoff is zero.
- **Compatibility shim:** establish the initial dense Torch/JAX canonical
  center with a full sweep, without numerical isometry discovery in Python.
  Preserve the center dictionary and existing canonical-boundary validation.
- **Adopt:** opt-in `register_jax_linalg(stabilized=True,
  qr_rank_policy="adaptive")`. SVD remains the existing truncation-safe rule.
  QR retains finite native VJPs, using a relative Tikhonov extension only for
  zero pivots or nonfinite native VJPs. Zero cotangents give zero derivatives.
  Native QR remains the default. Reset restores native SVD and QR mappings.
- **Compatibility shim:** scalar/dtype helpers avoid NumPy exception probes
  for Torch metadata; deferred JAX zero-norm errors use a conditional runtime
  callback. Torch's host validation remains an intentional graph break.

Finite QR extensions are not proofs of differentiability at singular charts.
The first experiment regularized all small QR pivots and produced incorrect
composed gradients at bond one; the final adaptive policy preserves finite
native derivatives there. Independent finite-difference regressions cover
gapped, actually truncated global replay and reversed nonlocal gate supports.

## Dependency audit

Python 3.12; Torch 2.6.0+cu124, JAX 0.10.2, Quimb
1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a, Cotengra
0.8.3.dev8+g8954240f2, Symmray 0.4.1.dev15+g0374aaa3c.

Inspected installed `gate_nonlocal`, `MatrixProductOperator.from_dense`,
direct compression and Autoray QR dispatch. Reviewed the official
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The requested Symmray `abelian_arrays.html` page was unavailable. Native
Symmray paths retain their prior implementation and are outside this change.

## Validation and limits

CPU tests use one BLAS/OpenMP thread and float64/complex128. The focused
optimizer, QR, public API, layout, Hamiltonian and PEPS-global-JAX selection
passed 294 tests. Backend and final QR/nonlocal-JIT tests passed 81 tests;
these groups overlap in eight QR tests. Ruff passed. The complete Pepsy suite
was not run. The Gaugy MPS suite validates downstream ownership, targets,
compiled gradients, metrics and optimizer state separately.

**Defer:** local direct JAX gauge derivatives failed a four-site bond-one
finite-difference check; Gaugy rejects that compiled combination. Global
direct JAX and zero-cutoff Torch direct are the validated compiled paths.
Torch adaptive-cutoff compilation hit Dynamo errors; use eager Torch for
nonzero cutoffs. Torch `aot_eager` works with graph breaks for direct replay;
Inductor's complex scalar backward failed. No Inductor acceleration, general
singular-chart accuracy, GPU validation, or optimal performance is claimed.

These are working-tree changes. Concurrent PEPS full-update changes and
their history are separate and must not be included in an MPS commit.
