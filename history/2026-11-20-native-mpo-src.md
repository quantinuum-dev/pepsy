# 2026-11-20 — Native MPO SRC warm starts

On `develop` at `a5fb031`, extended the existing MPO `guess-src` FIT
default to native Symmray targets. The optimizer compresses a disposable
copy of the active target with seeded sector SVD. Positive cumulative
cutoffs use full sector spectra, matching the MPS policy. Long-range bonds
are moved with native canonization so older Quimb builds can route them
without dense identity tensors. Dense fits retain seeded Quimb SRC.
[MPO optimizer guide](../docs/api/optimizers/mpo.md)

The modern MPO, TEM, public API, and package-layout selection passed 218
tests. The legacy native MPO selection passed 33 tests with one capability
skip. Thirty MPS SRC reference tests passed. Separate Torch CPU and CUDA
replays kept U1U1 fermionic tensors on their original device and agreed
with the untruncated native reference within $1.3\times10^{-15}$ relative
error for `dmrg` and `dmrg2`. These are working-tree changes.
Dense exact-replay probes also passed on JAX CPU and CuPy CUDA with native
gate payloads. The reusable dense path needs Quimb 1.16 for backend-native
SRC and was tested with Autoray 0.11. Symmray 0.5.1.dev2 supplies the native
sector solver. [Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html#enhancements-1-16-0) · [Symmray source](https://github.com/jcmgray/symmray)

The fixed-shot CUDA mode and bond scan is archived in the sibling examples
repository. [Study results](../../pepsy_examples/experiments/tensor_network_error_mitigation/RESULTS.md)
