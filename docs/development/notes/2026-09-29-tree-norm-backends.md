# Tree norm scale and backend repair — 2026-09-29

The [optimizer review](../../../history/2026-09-29-optimizer-commit-review.md)
identified a Torch target-norm dispatch regression and premature overflow or
underflow in unitary stabilization. Both reproduced before the repair.

## Implemented behavior

- Target-norm clipping uses Autoray `clip`, whose scalar bounds work on
  NumPy, Torch, JAX, and CuPy. No backend-specific optimizer kernel was added.
- Incoming and retained norms carry their tensor-network exponents separately.
  Stabilization divides working norms when the exponents match and otherwise
  combines their logarithms with the exponent difference through Autoray.
  The correction remains detached and on the state backend.
- Compression loss uses the same stripped norms before restoration. A common
  exponent cancels without a device-to-host transfer. Device events retain the
  exponent until the public diagnostic readout constructs represented norms.
  Those display values can overflow or underflow without changing fidelity.
- An operator changing the exponent retains the existing host-double ledger
  boundary. This preserves fidelities such as `1e-200` on JAX without x64 and
  other float32-only devices. This scalar diagnostic boundary does not move
  state tensors or the stabilization calculation to the host.
- Unknown-center full contractions strip the exponent on an independent
  network wrapper. Native fermionic states retain their graded contraction.
- GPU validation additionally exposed TreeFIT treating CuPy array `.data` as
  tensor data. Only Quimb Tensor wrappers are now unwrapped; Autoray receives
  the actual array at the scalar readout boundary.

Zero norms and non-finite stored exponents remain errors under stabilization;
finite exponents outside the represented float range are accepted.

The exterior-identity proof and finite-iteration FIT schedule were not changed.
The review's finite-bond trajectory sensitivity remains a separate observation.

## Upstream audit

Installed versions inspected: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`,
Symmray `0.4.1.dev7+g83fb22865`, Torch `2.6.0+cu124`, JAX `0.10.2`, and
CuPy `14.1.1`. GPU checks used an NVIDIA RTX A5000.
Inspected installed `Tensor.split`, `TensorNetwork.copy`,
`TensorNetwork.contract`, and `TreeFIT.run_gate` signatures. Autoray's installed
Torch `clip` dispatch accepts scalar bounds and preserves dtype/device.

Consulted the official [Autoray dispatch documentation](https://autoray.readthedocs.io/en/latest/index.html),
[Autoray source repository](https://github.com/jcmgray/autoray),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray).
The Symmray abelian-array documentation URL was unavailable; installed source
and the official repository supplied the fallback evidence.

**Adopt:** existing Autoray clipping, dispatch, and scalar conversion APIs;
separate scale bookkeeping within Pepsy. **Defer:** dependency upgrades.
No compatibility shim or installed-library modification was needed.

## Validation scope

Regression tests cover disabled tracking, nonzero exponents, every tree DMRG
mode and mix, repeated lossy stabilization, changed operator exponents,
unknown canonical centers, native fermionic states, Torch gradients, and
absence of host scalar reads in the unchanged-scale device ledger. NumPy,
Torch CPU/CUDA, JAX, and CuPy are exercised in the final validation recorded
in the [repair handoff](../../../history/2026-09-29-tree-norm-backend-fixes.md).

Two existing CUDA range checks also failed on the preceding `4e7d08e` source:
complex64 warm-up drift accumulated before the test assumed unit norm. The
test now explicitly normalizes that warm-up state before testing extracted
scale; its existing numerical tolerances are unchanged.

JAX GPU's default matrix-multiplication precision exceeded the same tight
range-test tolerances. All 32 JAX backend checks passed with
`JAX_DEFAULT_MATMUL_PRECISION=highest`; JAX CPU checks use the default setting.
Production precision settings are unchanged.
