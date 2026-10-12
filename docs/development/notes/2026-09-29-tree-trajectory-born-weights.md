# Tree trajectory Born weights — 2026-09-29

The final trajectory audit found that ordinary TreeOptimizer Kraus sampling
replayed each outcome on a complete optimizer copy. That inherited both
unitary stabilization and finite-bond compression. Either can alter a trial
norm and therefore the sampled Born probabilities.

## Confirmed failures and repair

- Amplitude damping with gamma=0.3 on an excited state returned weights
  0.5/0.5 under constructor stabilization, rather than 0.7/0.3. A zero-weight
  outcome could instead raise during attempted restoration.
- A complete two-site Kraus channel with weights 0.25/0.75, whose second
  outcome entangles a product state, returned 0.4/0.6 with chi=1. Compression
  removed part of the trial norm before the probability calculation.
- Importance sampling could select a positive branch of amplitude 1e-20,
  which TreeOptimizer's default normalization threshold then left unnormalized.

Tree sampling now forms the small local Gram operator `K†K` using the live
array backend and calls the existing exact TTN local expectation on a private
state wrapper. Dense one-site cases use a canonical tensor; multi-site cases
contract the active Steiner subtree with canonical exterior boundaries.
It does not replay FIT/direct trial updates, densify the state, copy optimizer
histories, advance the live RNG, or modify live tensor data and isometry proofs.
Only the selected branch uses the requested compression mode and bond cap.

Working norms strip the common stored exponent before forming ratios. Known
Hermitian Gram expectations discard imaginary contraction roundoff before
scalar readout, including complex64 GPU arithmetic. Selected ordinary-tree
branches normalize with `eps=0`, preserving rare positive outcomes. MPS and
stabilizer normalization policies are unchanged.

These contractions are required for physical sampling probabilities. The
ordinary gate compression-infidelity metric and its inexpensive local norm
readout are unchanged. Probabilities are exact for the current represented
state, which can already contain earlier compression error.

## Upstream decision

**Adopt:** existing Autoray transpose/conjugation dispatch and
`TreeTensorNetwork.local_expectation(..., normalized=True)`; keep the existing
copy/apply compatibility protocol for external optimizer lookalikes.
The inspected local-expectation implementation performs no truncation and
supports private wrapper readout without gauge restoration. Reuse the
[same-session upstream audit](2026-09-29-tree-norm-backends.md#upstream-audit)
and unchanged installed dependency versions. No dependency modification or
upstream shim is required.

See the [trajectory audit handoff](../../../history/2026-09-29-trajectory-final-audit.md)
for tests, backend coverage and the Tree DMRG iteration defaults.
