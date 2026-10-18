# 2026-10-14 — Flat-Z2 TFIM NetKet trajectory path

- Scope: add the symmetry-adapted TFIM X-basis path requested for the existing
  NetKet PEPS TDVP example.
- Branch / baseline commit: `develop` / `7ec0a01`.
- Commit status: uncommitted working-tree changes.

## What changed

- `build_ising_vmc(..., basis="x")` requires a bosonic, flat-Z2,
  charge-zero `SymPEPS`; it constructs the rotated `J XX - h Z` Hamiltonian
  and samples only the even sector with an initialized, ergodic pair-flip
  Metropolis chain.
- `examples/tfim_netket_tdvp.py --basis x` creates the all-`+x` product state
  as the neutral Z2 vacuum and grows bonds with symmetry-preserving second-order
  Trotter gates. The exact oracle and observables are expressed in that basis.
- Flat-Z2 NPZ checkpoint/restart remains unsupported and is rejected clearly.

## Why

The original TFIM global-X symmetry is a charge-parity constraint in the
rotated X eigenbasis. Pair flips preserve that sector and connect all of its
configurations; single-site Metropolis flips would propose only forbidden
odd-sector states.

## How it was checked

- Ruff on the changed Python source and example: passed.
- `git diff --check`: passed.
- Numerical tests and a trajectory were not run in this task.

## Open questions

- Confirm Symmray startup-gate growth and the 4x4 TDVP path numerically on the
  selected GPU environment before interpreting physical results.
- Add a block-metadata-preserving checkpoint format if Z2 runs need restart.

## GPU trajectory attempt

- Device: NVIDIA RTX A5000, JAX 0.11.2 / NetKet 3.22.4.
- The first flat-array gate/normalization path failed inside Symmray contractions.
  Running the symmetric startup in sparse Z2 storage and flattening afterward
  succeeded; the symmetric Trotter handoff retained active bond dimension 4.
- Exact handoff check at `t=0.02`: energy `-15.9999992586` versus exact
  `-16`; startup infidelity `1.01e-11`; `mx=0.997604`, `zz=0.001598`, and
  parity `1.0`.
- The JAX force evaluation initially hit a mesh mismatch because flat charge
  tables were moved to the accelerator. Keeping those tables as static NumPy
  metadata fixed that path.
- TDVP did not accept a step. CG failed the residual guard at 2,000 samples and
  again at 8,192 and 40,960 samples (the latter used `diag_shift=1e-2`). A
  short Cholesky run was stopped during XLA compilation after the requested
  wrap-up; it produced no accepted endpoint.
- Consequently there is no dynamical fidelity or energy-drift result from this
  attempt. Do not interpret the exact startup handoff as a successful tVMC
  trajectory. The solver/QGT failure remains the blocking numerical issue.
