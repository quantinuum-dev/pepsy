# 2026-10-13 — Independent JAX compression validation and remaining gradient gap

- Scope: wrap Torch compilation, investigate fixed-sweep zip-up/SRC/DMRG/DMRG2
  contraction and stable JAX derivatives, and improve NetKet readiness.
- Initial branch / baseline: `develop` / `9b2bef2`.
- Final baseline: `8f8695b`; another concurrent session committed the core
  implementation as `7a29270` and its publication record as `8f8695b`.
- This session performed no staging, commit or push. Follow-up Mapping support,
  the explicit expected-failure regression and this validation record remain
  uncommitted with the earlier TDVP work.

## Findings and changes

Reviewed public Quimb compressor dispatch, existing Pepsy FIT and JAX QR/SVD
rules against primary literature. Deterministic native Symmray zip-up and
fixed-sweep fitting work; installed randomized native RNG operations do not.
Dense seeded SRC, SDC and density-matrix paths have separate checks.
One-site fitting avoids two-site SVD updates but retains QR and a zip-up
initializer. Few sweeps differentiate their actual initialization and updates,
not a converged fixed point.

Concurrent edits removed a symmetric SVD-factor correction after projector
checks exposed its incorrect behavior. A pre-revision combined run had
159 passes and 18 reconstruction-gradient failures; the final restricted
contract and current tests were rerun. Retained-degenerate reconstruction
still has a real gap even with a gapped rank cut. A new strict expected-failure
case keeps that gap reproducible; a composed-factor rule is required to fix it.
The [review](../docs/development/notes/2026-10-13-jax-compression-autodiff.md)
separates resolved accuracy, finite surrogate derivatives and unsupported
degenerate reconstructions.

## Final independent validation

- CPU JAX decomposition/QR/compression/FIT/backend selection:
  **100 passed, 63 deselected**, 261.17 s.
- Minimal retained-degeneracy reconstruction check: **1 expected failure**,
  2.16 s. This is an unresolved numerical limitation, not a passing accuracy
  certificate for that case.
- Final RTX A5000 JAX CUDA selection: **4 passed, 22 deselected**, 241.40 s
  including compilation. Three native fermionic compression/finite-difference
  checks; one dense-spin native NetKet SR/real-coordinate TDVP integration.
- Torch zero-sector log-output regression: **1 passed**, 5.20 s.
- Earlier 37-case final projector/resolved-spectrum selection passed separately;
  counts overlap and are not additive.
- Ruff, whitespace and local link checks passed. The full-suite attempt was
  terminated after concurrent source/test revisions invalidated its collected
  state. No full numerical suite success or physical trajectory claim.

No dependency upgrades, installed-library changes, extra algorithms, or new
long physical simulations. Native fermionic real-coordinate TDVP charts,
implicit FIT gradients and composed-factor derivatives remain separate work.
