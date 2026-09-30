# 2026-09-29 — Native cluster MPO charges and CUDA correctness

This follows the [joint-product review](2026-09-29-joint-product-parity.md).
The earlier hopping and zero-cutoff failures are now fixed in the working
tree. This note supersedes those specific unresolved findings, not the
review's broader performance and adaptive-rank limits.

## Implementation

`operators._cluster_native` factors neutral NumPy residuals sector by sector.
Each operator-Schmidt split combines the sector spectra before selecting the
cutoff or total rank cap. Null singular vectors remain in their own sectors;
exactly zero residuals retain the existing one-channel shortcut. Nonzero
forbidden entries raise; no tolerance silently clips charge violations.

After direct assembly, physical charge transfers determine virtual flux on
every populated channel. This also handles identity/background gaps and
crossing or nested graph collections. Both open boundaries are neutral;
inconsistent flux raises. Numerically disconnected channels impose no
operator contribution and receive a consistent neutral-root assignment.

The repeated-physical-charge tests exposed a second error in native MPO
`to_dense()`: mapping only fused total charges cannot restore the ordering
of degeneracy offsets. The conversion now uses public `unfuse_all()` on the
contracted physical result, applies the individual physical basis maps, and
reshapes to the requested output groups. Virtual tensors remain block sparse
during contraction. Ordinary Torch arrays bypass this native-only conversion.

No tensor-network algorithm moved into Gaugy. Its changes are downstream
regressions and status documentation using public Pepsy APIs.

## Validation

Environment: the session's shared Python 3.12 environment, one BLAS/OpenMP
thread, `JAX_PLATFORMS=cpu`, CUDA visible. GPU: NVIDIA RTX A5000.

Final Pepsy command, with `python -m pytest -q -o addopts=''`:

```text
tests/test_native_cluster_mpo.py tests/test_joint_cluster_parity.py
tests/test_mpo.py tests/test_mpo_cluster.py tests/test_mpo_cluster_recursive.py
tests/test_mpo_cluster_compression.py tests/test_public_api.py
tests/test_package_layout.py
```

Result: **316 passed**, two existing API deprecation warnings, 68.46 seconds.
The new native suite covers independent SciPy ordered exponentials in all
four supported bosonic groups; three-site generators; repeated physical
charges; requested physical index reordering; graph gaps/crossing/nested
collections; zero steps/generators; global rank-cap selection; strict charge
rejection; and explicit rejection of unsupported live tensor backends.
Four-site random neutral residuals reconstruct with float32, complex64,
float64 and complex128 cores retaining the input dtype.

Joint dense MPO (direct and recursive), graph PEPO and routed square PEPO
tests compare full/truncated cluster targets and all five live derivatives
with independent SciPy references at nonzero and zero parameters on CPU and
CUDA. There are eight CUDA cases in Pepsy. Gaugy adds eight CUDA cases across
direct/auto/graph/square layouts and reuse controls, including frozen rank-one
compression and actual-PEPO trace derivatives versus finite differences.
Its final materialization/binding/package selection reports **137 passed**,
six existing Quimb deprecation warnings. These are correctness checks, not
performance benchmarks. No full repository suite or large-lattice GPU
benchmark was run. Pepsy source/test Ruff, Gaugy changed-test Ruff, whitespace
and new documentation-link checks pass.

During development, the new native tests caught the physical-degeneracy
ordering defect. The wider suite also caught an overbroad native export
guard because Torch tensors expose `indices`; the final guard checks the
native `unfuse_all` capability. All final runs above include both corrections.

## Upstream compatibility audit

Installed versions were unchanged during this task:

| Package | Version |
| --- | --- |
| Quimb | 1.15.1.dev66+ge927f06e1 |
| Autoray | 0.11.1.dev3+g1b476b305 |
| Cotengra | 0.8.3.dev7+g1d7fd333f |
| Cotengrust | 0.2.1 |
| Symmray | 0.4.1.dev7+g83fb22865 |
| Torch | 2.6.0+cu124 |
| JAX | 0.10.2 |

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The [Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable to the browser; installed public implementation/signatures
were inspected instead.

**Adopt:** public `symmray.get_symmetry`, its charge `combine`/`sign`,
`AbelianArray.unfuse_all(inplace=False)` and `to_dense(index_maps=None)`.
Use NumPy SVD on local charge blocks with Pepsy's existing relative cutoff
policy. No compatibility shim, installed-library edits or dependency change.
**Defer:** broader native cluster backends and adaptive-rank differentiation.

## Boundaries

Native construction here is bosonic, NumPy, and direct assembly. Fixed native
factorization, Torch/JAX native cluster materialization, streaming/recursive
native assembly and graded fermionic histories remain unsupported. Dense
fixed splits and explicitly frozen projections support the checked gradients;
this does not supply derivatives through discrete rank changes or through the
reference compression preparation. Large-lattice convergence, memory and GPU
performance require separate measurements.
