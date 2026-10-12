# 2026-10-01 — JAX callbacks for host gradient solvers

Implemented in the working tree: SciPy/NLopt retain their existing solver
controls and use JAX JIT value/gradient callbacks for native JAX parameters.
This adopts public JAX autodiff/device APIs and the installed solver callback
interfaces. No installed library or tensor-network backend registration changed.

The installed signatures are `scipy.optimize.minimize(..., jac, hess, hessp,
options, callback)` and `nlopt.opt.set_min_objective(self, *args)`. Versions:
JAX 0.10.2, Torch 2.6.0+cu124, SciPy 1.17.1 and NLopt 2.11.0.
Tensor-network dependencies remain Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
Cotengrust 0.2.1 and Symmray 0.4.1.dev7+g83fb22865.

The repository-required upstream review checked the
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray abelian-array documentation URL returned an internal error;
the official repository was available. Classification: **defer** changes to
these dependency integrations; they do not own the solver callback restriction.
No compatibility shim, dependency upgrade, or contraction change was needed.

Packing uses separate real/imaginary coordinates and differentiates that real
vector, avoiding ambiguity in JAX complex-gradient conventions. Host iterates
and derivatives cross the CPU/device boundary; the objective stays on the
original JAX device. One device is required for all JAX parameters.
The supported precision follows the caller's JAX x64 setting.

See the [solver API](../../api/solvers/gradient.md) and
[validation handoff](../../../history/2026-10-01-jax-host-solvers.md).
