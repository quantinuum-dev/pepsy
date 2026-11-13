# Native fermionic MPS evolution and Hubbard audit

## Scope and source state

Audit requested for the Etienne/Henrik Fermi-Hubbard examples, especially
native fermionic MPS evolution with `dmrg` versus direct/Quimb replay and the
main `Fermion` implementation. Baseline: `develop`, HEAD `1d0a894`.
No implementation, Hamiltonian definition, notebook, or dependency was
changed. No commit or push was made. Existing PEPS/FU edits were preserved.

The current `symm_fermions.py`, `symmetric.py`, `_symmetric_mpo.py`, MPS
`optimizer.py`, and MPS `_norm.py` match their HEAD blobs byte for byte.
`/tmp/pepsy_fermion_audit_source_state.json` records their SHA256 hashes.

Environment: shared Python 3.12, local `pepsy/src`, Pepsy 0.6.0, Autoray
0.8.4, Symmray 0.5.1.dev2+g20035beb7, Quimb 1.15.0, Torch
2.14.1+cu126, NumPy 2.5.2. Numerical probes used one CPU thread and
complex128, independently of the existing FU campaign.

## Owners and example routes

- [`Fermion`](../src/pepsy/tensors/symm_fermions.py) is the main implementation.
  `SpinfulFermion` is its compatibility subclass; `SymmFermions` supplies
  factories. Couplings remain explicit at Hamiltonian/gate construction.
- Hubbard convention: hopping `-t`, onsite `U n_up n_down`, chemical
  potential `-mu n`. Native Symmray handles grading; a bosonic JW MPO is a
  separate DMRG/reference representation.
- The sibling example's `experiments/fermi_hubbard/Superconducting-pairing/`
  contains `mps_fermions.ipynb`, `mps_fermions_helpers.py`, and
  `benchmark/run_.py`: DMRG preparation, explicit native Strang pulse and
  relaxation. The gate builder is `build_fermion_strang_stream`.
- `mps_etienne.ipynb`, `fh_etienne_helper.py`, and `benchmark/run_mps.py`
  own serialized Etienne replay. `instruction_to_gate` uses generic
  `Fermion.operator_gate`. Older README references to a `mps/` subdirectory
  and `sim_henrik*.ipynb` are stale in this checkout; those Henrik-named
  files are absent. Original serialized boundary files are also absent;
  the bounded audit generated a 2x2 preparation instruction stream in memory.

## Measured findings

1. **Hubbard Hamiltonian checks passed.** Six independent small-system
   dense/JW spectrum cases passed for total U1 and spin-resolved U1U1,
   including noncontiguous/wrap bonds. Explicit native term, hopping
   exponential, and chemical-potential tests also passed. The selected suite
   finished with 9 passes and one generic-exponential failure described below.
   See `/tmp/pepsy_hubbard_hamiltonian_audit.log`.

2. **Torch native Hubbard DMRG replay executes and matches direct replay in
   the tested case.** A four-site, 32-gate, two-step Strang stream from the
   actual example helper ran in `direct`, `dmrg`, `dmrg2`, and `dmrg3` for
   U1 and U1U1 at chi=16. Every output remained a native Symmray fermionic
   MPS; DMRG/direct normalized fidelities were within 7.2e-13 of one.
   Raw norms were approximately .99999929 (U1) and .99999858 (U1U1), so this
   agreement is not a machine-precision norm-conservation claim. A second
   stronger-time-step chi=2 probe executed all eight symmetry/mode cases,
   kept the requested bond cap, and retained native arrays. Finite-chi
   fidelities differed among compression methods as expected; no physical
   convergence claim follows. Reports:
   `/tmp/pepsy_native_fermion_mode_probe.json` and
   `/tmp/pepsy_native_fermion_truncation_probe.json`.

3. **NumPy replay is blocked by the installed Autoray dispatch.** Native
   NumPy `direct`, `dmrg`, `dmrg2`, and `dmrg3` fail with missing
   `stop_gradient` for NumPy. The failure occurs in initial norm tracking,
   before DMRG fitting. The existing selected native MPS regression suite
   reported 36 failures, all blocked by the NumPy reference setup; that
   outcome does not establish 36 independent DMRG defects. Log:
   `/tmp/pepsy_fermion_dmrg_mode_audit.log`.

4. **Generic `Fermion.operator_gate` has a missing-sector exponential
   defect.** `_gate_from_term` exponentiates only present sparse blocks.
   Operators such as `sx`, `sy`, `sz`, and `double` omit zero generator
   sectors; their exponentials must supply identity there. Even
   `operator_gate("double", 0)` differs from identity by sqrt(3) in
   Frobenius norm; `operator_gate("sx", 0)` differs by sqrt(2).
   Existing spin-gate unitarity and generic/named interaction tests fail.
   The generated 2x2 Etienne preparation consequently fails in both direct
   and DMRG routes with a later empty-SVD `None.get_namespace` error.
   This is separate from the checked Hubbard Hamiltonian coefficients.

5. **Etienne-like one-site FIT has a remaining discrepancy.** An isolated
   process replaced the exponential helper with a diagnostic version that
   copies the fused generator and calls `fill_missing_blocks()` before
   exponentiation. Nothing was patched on disk. That diagnostic restores
   the tested local gate unitarity and allows the generated 44-gate 2x2
   preparation to execute at chi=16. `dmrg2` and `dmrg3` then agree with
   direct replay to floating-point precision. One-site `dmrg` yields
   fidelity .6501418894 at three sweeps, and .7349356008 at ten or twenty
   sweeps. More iterations alone did not resolve this case. Its cause is
   unverified; do not certify the full Etienne one-site DMRG workflow from
   the ordinary Hubbard-stream agreement. The direct diagnostic raw norm
   is .99866652, also precluding an unqualified unitarity/convergence claim
   for the complete replay. Script/report:
   `/tmp/pepsy_fermion_expm_isolation.py` and
   `/tmp/pepsy_fermion_expm_isolation.json`.

## Upstream consultation and validation limits

The requested Symmray ReadTheDocs index and changelog were inaccessible
through the browsing tool. Consulted the official
[Symmray README](https://github.com/jcmgray/symmray/blob/main/README.md)
and installed `symmray/scipy/linalg.py` instead. The README documents native
grading, local Hubbard constructors, coordination-weighted onsite terms, and
the importance of zero blocks for exponentiation. The installed exponential
phase-syncs and exponentiates existing blocks; it does not add missing ones.
The proposed missing-sector correction remains an isolated diagnostic,
not an adopted compatibility change.

No full suite, CUDA replay, production lattice, original author-supplied
boundary replay, or dependency upgrade was performed. No implementation was
changed, so Ruff was not run. Follow-up repair would need focused zero-angle
identity/unitarity tests and a separate investigation of one-site native FIT
on the Etienne spin-mixing stream, preserving phase and charge metadata.
