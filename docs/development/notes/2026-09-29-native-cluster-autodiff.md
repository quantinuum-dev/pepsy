# 2026-09-29 — Native spin cluster MPO autodiff and compression

User scope is Pauli/spin construction only. This supersedes the native
Torch/JAX and recursive/streaming restrictions in the earlier
[charge-factorization record](2026-09-29-native-cluster-charges.md).
No fermionic construction or reverse dependency on Gaugy is added.

## Implementation and numerical contract

`operators._cluster_native` carries structural charges alongside residual
cores, including gaps, collections, direct sums and recursive products.
This replaces the earlier value-dependent flux inference. Exact fixed
splits act inside each allowed sector and preserve zero-valued derivative
channels. Independent coefficient groups must conserve charge for every
binding, proved from the cached static local matrices. Live coefficients,
factor scales and the step are supported; local operators stay static NumPy.

`_mpo_sparse` builds Symmray blocks on their original backend and device,
using static gathers and functional assembly. It retains tensor zeros, dtype
and gradients. Native direct and uncapped recursive construction therefore
support Torch autograd and JAX `jit(value_and_grad)`.

Native intermediate compression prepares an orthonormal environment with
a reverse sweep, then applies the requested total cap and cutoff to the
combined sector spectrum. Numerical null directions are removed even with
no explicit cutoff. Only singular spectra are copied to the host for the
discrete allocation; factor arithmetic stays on the source device. Pepsy's
existing paired-factor projector VJP handles retained Torch factors.
When the entire row space is retained, the constant identity split avoids
introducing an unnecessary moving singular-vector gauge.

An initial public Quimb native QR compression experiment had finite but
incorrect truncated gradients (about 0.06 error against finite differences).
That implementation was replaced, not accepted with weaker tolerances or
global regularization changes. Final truncated checks disable the stabilized
QR/SVD registry and compare all live derivatives with finite differences of
the same compressed construction.

`FirstDegreeMPO.compress_adaptive` shares this NumPy/Torch sector compressor,
validates unproven input charge flow before projecting, and preserves custom
physical index and site-tag conventions. `compress_fixed_rank` now explicitly
rejects native inputs rather than silently dropping symmetry metadata.
Reports label the discrete allocation `native-sector-projector`, with
`differentiable=False`; this does not disable local-chart Torch derivatives.

## Validation

Shared Python 3.12 environment; one BLAS/OpenMP thread, `JAX_PLATFORMS=cpu`;
CUDA tests ran on an NVIDIA RTX A5000. Broad Pepsy selection:

```text
tests/test_native_cluster_mpo.py tests/test_native_cluster_autodiff.py
tests/test_joint_cluster_parity.py tests/test_mpo.py tests/test_mpo_cluster.py
tests/test_mpo_cluster_recursive.py tests/test_mpo_cluster_compression.py
tests/test_symmetric_tensors.py tests/test_public_api.py tests/test_package_layout.py
```

**555 passed**, four existing deprecation warnings, 262.48 seconds. This run
preceded the final public compression guards and index-name preservation;
the final native/autodiff/compression/recursive recheck passed **92 tests**
in 111.08 seconds, covering those guards and metadata preservation. Gaugy's existing
joint/materialization/binding/package selection passed **137 tests**, six
existing Quimb warnings, 42.53 seconds. No full repository suite was run.

New tests use independent SciPy ordered matrix products and finite
differences for coefficient, factor-scale and step gradients. Coverage
includes U1/Z2 Torch CPU/CUDA, JAX tracing, zero and nonzero re-evaluations,
graph crossings, connected full-cluster recursion, both compression sweep
directions, total caps, actual truncation, invalid zero bindings and explicit
JAX compression errors. Existing U1U1/Z2Z2/repeated-charge tests also exercise
NumPy fixed splits. A separate float32 probe retains complex64 native blocks
and a live float32 gradient. These are small correctness checks, not timing
or large-lattice memory/convergence claims.

## Upstream audit

The same-task audit linked in the previous record is reused; the environment
is unchanged: Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124, JAX 0.10.2.
Public installed `MatrixProductOperator.compress(form=None, create_bond=False,
**compress_opts)`, `AbelianArray.to_dense(index_maps=None)` and
`AbelianArray.from_blocks(blocks, duals, charge=None, symmetry=None, **kwargs)`
were inspected again. The unavailable Symmray array documentation is recorded
in the earlier audit; installed public source supplied that check.

**Adopt:** public Symmray construction and Autoray dispatch, plus Pepsy's
existing fixed splits and paired-factor projector derivatives. **Defer:**
Quimb native QR as the differentiable compressed-assembly path after its
finite-difference failure. No dependency change, library patch or upstream
compatibility shim was introduced.

## Limits

Exact fixed sector construction can produce large bonds. Truncated Torch
derivatives are first-order, require gauge-invariant use of factor pairs,
a locally unchanged rank and a resolved retained/discarded gap; rank-change
points are not differentiable. Native JAX intermediate/adaptive compression
is explicitly unsupported; use uncapped fixed assembly for tracing. No
large-lattice GPU performance or higher-order derivative claim is made.
Fermionic histories are outside the user's requested scope.
