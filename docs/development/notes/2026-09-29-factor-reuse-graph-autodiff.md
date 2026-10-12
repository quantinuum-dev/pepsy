# 2026-09-29 — Factor reuse and graph backend compatibility

Scope: the requested individual-factor exponential reuse and differentiable
graph PEPO materialization. Pepsy baseline `b343ced`; shared Python 3.12
environment, CPU-only validation. No dependency or backend registration change.

## Installed environment and primary review

Inspected versions: Pepsy 0.5.0, Gaugy 0.1.0, Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124 and JAX 0.10.2.
Public signatures inspected: Autoray `do`, Quimb `TensorNetwork.to_dense`,
and Pepsy `GraphActivePEPOBlocks.to_tensor_network`. The existing Pepsy
backend tree-splitter signature and graph block materializer were inspected.

Reviewed primary [Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changes](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The requested Symmray Abelian-array documentation URL returned an internal
error; the primary repository and installed version were available.

## Decisions

- Adopt: existing Autoray reshape/transpose/stack, Quimb graph contraction,
  and Pepsy's exact fixed-index tree split. Graph matrix-unit insertion now
  reshapes coefficients instead of contracting against a host identity.
- Adopt: existing binding-aware `ClusterReusePlan` for individual factors.
  Reuse is within an ordered factor; scales/order stay intact. Graph numerical
  caches are function locals; square per-factor caches are evaluation data.
- Compatibility: keep legacy NumPy auto SVD/rank selection and return types.
  Backend auto with no cap uses exact fixed splits; capped backend auto uses
  the existing SVD implementation. No custom SVD derivative or driver change.
- Compatibility: retain the prepared square topology cache's two-argument
  key; graph edge-index topologies occupy separate keys.
- Defer: native Symmray graph construction, rank-changing derivatives, GPU
  performance, and square virtual routing for diagonal/higher-body terms.

## Numerical findings

New per-factor reuse exposed two integration issues, both corrected:

1. A constant graph factor cached first on a host-only cluster could meet a
   live Torch factor later. Align local factors before ordered multiplication,
   using the existing backend/dtype alignment helper without resolving bindings.
   Uniform square factors also share a common backend with reuse on or off.
2. Smaller square batches exposed Torch 2.6 singleton 2x2 exponential errors
   up to `6.14e-11` versus SciPy in the existing periodic Gaugy regression.
   Pooling independent factor representatives in backend batches, and avoiding
   a singleton tail when the batch budget allows, restores that regression's
   original tolerance without duplicated exponentials. A nine-factor onsite
   check verifies batches `[7, 2]`, values against analytic sine/cosine Pauli
   exponentials, and all gradients. No Torch internals are patched.

Pooling also exposed inaccurate singleton Torch reference exponentials in
older full-cutoff tests. One new dense result agreed with SciPy to `6.7e-16`,
while the previous Torch reference differed by `8.3e-12`. The affected tests
now use differentiable 32-term dense Taylor references for their small-norm
inputs, independently checked against SciPy. No acceptance tolerance was
relaxed. Graph/square reuse-count tests cover residual traces as well as
materialization.

Graph `compact()` retains tensor-valued zeros so zeros carry parameter
derivatives and JAX tracers never require host predicates. Graph `to_dense()`
preserves the backend. Declared NumPy operator precision still participates
in promotion: float64/complex128 operators can promote float32 coefficients.

Reports retain NumPy SVD reconstruction diagnostics. Backend/fixed graph
reports use `None` for uncomputed error/norm; this does not certify zero
truncation error. Fixed channels can have substantially larger bonds.

See the [handoff](../../../history/2026-09-29-factor-reuse-graph-autodiff.md)
for final test scope, measured target counts, and publication status.
