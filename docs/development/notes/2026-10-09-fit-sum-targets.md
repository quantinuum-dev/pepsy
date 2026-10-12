# Termwise FIT targets, 2026-10-09

Implemented in the local working tree on `develop`, baseline `21f883b`.
No dependency changes, commit, or publication in this task.

## Algorithm and scope

`FIT([T1, T2, ...], p=guess)` dispatches to a private FIT specialization.
Each term owns ordinary FIT target metadata and uses the same working MPS/MPO.
Directional sweep-cache entries hold tuples of overlap environments. Local
effective tensors are added before the existing one-site update or native
two-/three-site split. This retains cancellation and target scale without
materializing a direct-sum target or truncating its terms independently.
The shared canonical fitted-state norm metric is unchanged.

Dense active-window sums contract the actual outside overlaps once per run,
preserving coefficients outside the interval and leaving outside fitted arrays
untouched. The input must already be canonical outside that interval. Native
fermionic sums reuse the existing conjugated working gauge and native
canonical preparation. The one-site full-chain sum path uses the common
bra-gauge kernel instead of the older single-target ket-gauge implementation.

Target sum norms are evaluated only for optional verbose fidelity, retaining
all complex cross terms. This diagnostic costs quadratically many target
overlaps; ordinary fitting has separate per-term environments. Physical
indices/dimensions and site tagging are validated. Target exponents are
absorbed once into owned tensors. Repeated references share an owned target,
including when `copy_target=False` transfers ownership.

Public documentation: [FIT API](../../api/fitting/local.md#fitting-a-sum-of-target-networks).
Gaugy consumers were not changed; this task implements the Pepsy capability.

## Upstream audit

Installed versions: Pepsy 0.5.0; Quimb 1.15.1.dev90+g6a3906cbe;
Autoray 0.11.1.dev14+g014a3f69a; Cotengra 0.8.3.dev8+g8954240f2;
Symmray 0.4.1.dev15+g0374aaa3c; Torch 2.6.0+cu124.

Reviewed official [Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changes](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The [Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an internal error; installed native operations and official source
were used for that portion of the audit.

Classification: **adopt** public Quimb tensor addition, contraction, and split
dispatch with Autoray/native Symmray arrays. No shim or vendored upstream code.
Installed `Tensor.split` accepts `method='auto'`, `max_bond=None`,
`cutoff='auto'`, `cutoff_mode='rel'`, `absorb='auto'`, and explicit left/right
indices. Existing FIT supplies its resolved controls. `tensor_contract`
accepts `output_inds`, `optimize`, and `preserve_tensor`; network contraction
accepts `output_inds`, `optimize`, `backend`, and `inplace`. Sum fitting uses
these existing public routes; it introduces no decomposition driver.

## Validation

Deterministic tests compare sums against materialized dense references for
MPS/MPO, complex64/complex128, all three entry points, and block sizes 1/2/3.
Additional tests cover layered MPO cancellation with materialization forbidden,
window coefficients and outside ownership, cache reuse through 3/2/1
transitions, Torch gradients against a dense objective, native Z2 bosonic
arrays, native U1U1 fermionic states, and independently constructed native
U1U1 MPOs. Native tests forbid `to_dense`. Single-site zero sums, retagging,
ownership transfer, repeated references, physical-space errors, and network
exponents are included.

Final suite results are recorded in the [session handoff](../../../history/2026-10-09-fit-sum-targets.md).
No CUDA, compilation, or performance benchmark claim is made.
