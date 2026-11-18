# 2026-11-09 — Lattice-DMT capability audit

- Scope: understand [arXiv:2610.40776](https://arxiv.org/pdf/2610.40776)
  and assess Pepsy/Quimb support; no algorithm implementation requested.
- Branch / baseline: `develop`, `ad523b6`.
- Status: this journal is uncommitted; no code changed, committed or published.
  Existing sampling edits and untracked records were preserved.

## Findings

The paper combines operator MPS evolution with geometry-dependent preservation
constraints during truncation. Its distinguishing compression rule is absent
from the inspected Pepsy and installed Quimb source.

Pepsy's [PauliMPO](../docs/api/operators/pauli_mpo.md) already provides native
four-label cores, exact products, local operator transformations and standard
QR/SVD compression. Quimb supplies dimension-four MPS storage, canonicalization,
MPO application, gates and TEBD. Pepsy's existing DMRG replay fits targets;
its PEPS/VMC TDVP is not a two-site operator-MPS TDVP driver. No MPS TDVP,
Krylov subspace-expansion evolution or observable-preserving lattice-DMT
compressor was found in these installed paths. Quimb's ordinary `dm` compressor
is a separate density-matrix compression method, not the paper's DMT.

Potential implementation boundary: native Pauli cores or a Quimb dimension-four
MPS, a geometry-aware bond constraint builder, protected-subspace compression,
and an evolution driver that invokes it at every lossy split. Applying this
compressor after ordinary replay compression cannot restore discarded data.
Trotter replay could provide an initial alternative evolution scheme; it would
require its own timestep and convergence checks rather than reproducing the
paper's integrator.

## Checks and limitations

- Installed metadata: Quimb 1.15.0, Autoray 0.8.4, Cotengra
  0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, Symmray
  0.4.1.dev15+g0374aaa3c. Pepsy metadata says 0.5.0 although the checkout README
  says 0.6.0; imports resolve to this checkout's `src/pepsy/__init__.py`.
- Inspected local source, the paper's main text, end matter and supplement,
  and official Quimb MPS/compression/TEBD documentation.
- A deterministic four-site real Pauli-core probe reproduced coefficients as
  a Quimb dimension-four MPS to 8.4e-17 and preserved them through Quimb
  canonicalization to 5.6e-17.
- Reducing that operator to bond two with `compress_pauli` changed several
  single-site coefficients by .0032–.0081 and one tested adjacent-pair
  coefficient by .0177. This illustrates lack of preservation constraints;
  it is not a transport benchmark or a judgment of SVD's intended objective.
- A separate attempted `PauliMPO.inner` readout on real native cores failed:
  `_complexify` passes `dtype` by keyword to Autoray's positional-only
  `astype` adapter. Direct contractions of selected Pauli coefficients worked.
  This compatibility issue remains unfixed; it was not needed for the
  representation/compression probe.
- No L-DMT dynamics, transport reproduction, GPU benchmark or full tests run.
