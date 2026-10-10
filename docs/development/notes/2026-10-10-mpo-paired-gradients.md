# MPO paired-factor derivative follow-up — 2026-10-10

## Problem and implementation

Gaugy's five-site periodic fixed-state study found accurate untruncated
operators but biased derivatives (up to `1.06723e-4` in an independently
verified direction). Raw QR/SVD charts can be poorly conditioned even when
the complete operator is smooth. Degenerate truncated caps can also be
nondifferentiable; a larger cap is not a universal repair.

`MpoOptimizer.run(mpo_factorization="projector")` now uses the existing
paired-factor primitive during gate-to-MPO conversion, outside-window
canonicalization, reductions and direct compression. The default remains
`"qr"`. The new operator helper reuses valid canonical-center metadata;
refactoring already canonical tensors needlessly created more poorly
conditioned charts in the first prototype.

The installed Quimb `compress_between` has a zero-cutoff/full-rank shortcut
that uses QR without passing through a requested split method. The scoped
projector path keeps the outer `max_bond=None` and passes the actual cap in
the nested split options. The public compression routine therefore reaches
the requested split and enforces the same cap there. No installed code or
global QR/SVD registration is changed. Gate construction uses zero cutoff
plus the primitive's documented numerical-null removal; the replay's cutoff
and cutoff mode still control state compression.

`ProjectorDerivativeError` is a `RuntimeError` subclass for unsupported
charts. Diagnostic code can explicitly capture errors and receive NaN
cotangents; it must reject them. Gaugy checks use this context so failed
custom VJPs do not escape through Torch autograd workers. Before this change,
repeated failures in standalone Torch 2.6/Python 3.12 probes printed their
reports but aborted during interpreter teardown. A subprocess regression
now verifies normal exit. Raw paired-factor backward still raises by default.

## Scope and independent evidence

- Dense NumPy/Torch, open MPO storage, direct compression, first derivatives.
  Physical periodic interactions may still use nonlocal gates on an open MPO.
- Fixed numerical rank, a resolved truncation boundary, and gauge-invariant
  paired-factor cotangents are required. Numerical nulls use the existing
  matrix-size times machine-epsilon rule. No regularization constants changed.
- Native arrays, cyclic MPO storage, channels, other compressors, layout
  changes and replay fallbacks reject the new option before replay.
- CPU/CUDA complex128 regressions compare an untruncated three-site replay
  against independently embedded dense gates and derivatives; cap-2 replay
  checks finite differences at two steps. Caps, device, scale and input
  ownership are asserted. Bare two-sided gates also exercise paired replay.
- The paired path fixes the two initially isolated five-site time-scheduler
  cases. A replay of the 24 recorded failures found three passing paired
  cases and rejected the remaining unsupported charts; no universal success
  is claimed for this path.

Gaugy additionally exposes an explicitly enabled, byte-budgeted direct trace
of small **untruncated** operators. It performs no decompositions, preserves
the same finite gate products and cap proof, and matches independent dense
derivatives for all six previously failing untruncated configurations. It
does not replace a capped/truncated objective, and its operator-storage
budget is not a bound on total autograd memory. This downstream path does not
change Pepsy's compression policy.

## Upstream audit and classification

Installed: Pepsy 0.5.0, Quimb `1.15.1.dev90+g6a3906cbe`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Symmray
`0.4.1.dev15+g0374aaa3c`, Torch `2.6.0+cu124`; activated Python 3.12.
Inspected installed `MpoOptimizer.run`, `gate_nonlocal_opt`, MPO dense
construction/canonicalization, `tensor_network_1d_compress_direct`,
`tensor_compress_bond`, and the zero-cutoff shortcut.

- **Adopt:** public split registration, bond canonicalization, nested direct
  compression options, and the existing Pepsy paired derivative.
- **Compatibility shim:** route the cap through nested split options to avoid
  the QR shortcut, only for explicitly requested dense direct projector replay.
  Construction, exact/actually truncated derivatives and cap enforcement are
  regression-tested. Unsupported representations/methods are rejected.
- **Defer:** native symmetry, other compressors, higher derivatives and
  arbitrary rank crossings. No densification of native arrays is introduced.

Checked [Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changes](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The requested Symmray
`abelian_arrays.html` page remained unavailable; installed source and the
official repository were used. No dependencies were upgraded or vendored.

Validation and publication are recorded in the
[session handoff](../../../history/2026-10-10-mpo-paired-gradients.md).
