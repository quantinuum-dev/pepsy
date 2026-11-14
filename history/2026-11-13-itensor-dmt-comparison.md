# 2026-11-13 — ITensor DMT comparison

- Scope: compare Pepsy's Pauli DMT evolution with an independent ITensor-based
  DMT implementation on Ising chains and check energy and local observables.
- Branch / baseline: `develop`, `d3612a15`. Existing unrelated working-tree
  changes were preserved; no implementation changes were made.
- Status: this handoff is uncommitted. No files were staged or pushed.

## Method

Used [MPSToolkit](https://github.com/jayren3996/MPSToolkit) commit
`a5d955c52b7c30be9495f49ce2f80fb0cfb60bb0`, version 0.2.0, with ITensors
0.9.32 and ITensorMPS 0.4.2 in an isolated temporary Julia 1.12.7 environment.
Pepsy ran in the checkout's specified Python environment with complex128
Pauli cores; the Hermitian ITensor state and gates used real Float64 operator
coefficients. The independent dense reference used SciPy matrix exponentials.

Open chains of `L` qubits used `H = sum_j h[j]`, with
`h[j] = 0.7 ZZ + 0.23 XI + 0.31 IX + g (0.6 ZI + 0.4 IZ)` and the same
bond Hamiltonian on every bond. The `g = 0` case is transverse-field Ising;
`g = 0.43` adds a longitudinal field. The onsite field is smaller at the
boundaries because fields are included in bond terms. Initial site `j`, with
one-based indexing, was `(I + 0.15 X + (0.10 + 0.25 (-1)^j) Z) / 2`.
The product is a positive, trace-one density matrix.

Both implementations applied exactly the same even-half, odd-full,
even-half gate sequence in ascending bond order, using density DMT,
three-site protection, zero cutoff and the same total bond cap. ITensor used
`dmt_step!` with `gate_backend=:product` and `direction=:R`; no Hilbert-Schmidt
renormalization was applied. Pepsy used `PauliDMT.apply_gate`. Measured trace,
energy, all local X and Z expectations and nearest-neighbor ZZ expectations
after each full step. An independent dense evolution applied the same gates,
and a second dense evolution used the exponential of the full Hamiltonian.

## Results

The first error below is the largest absolute Pepsy–ITensor difference over
all recorded X, Z and ZZ values and all times; the second is the largest
energy difference. `dt = 0.1` throughout this table.

| Model | L | cap | final time | local difference | energy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| transverse Ising | 6 | 32 | 0.8 | 2.21e-14 | 2.56e-15 |
| transverse Ising | 6 | 10 | 2.0 | 1.293e-3 | 2.02e-6 |
| mixed-field Ising | 6 | 10 | 2.0 | 4.863e-4 | 1.81e-7 |
| mixed-field Ising | 6 | 16 | 2.0 | 1.976e-7 | 1.27e-9 |
| mixed-field Ising | 8 | 12 | 1.2 | 5.345e-6 | 1.17e-8 |

The capped runs actually discarded a nonzero residual: for example the
maximum Pepsy reported residual norm was `6.11e-4` for the mixed-field
six-site cap-10 run and `1.44e-4` for cap 16. At cap 32, the transverse Ising
case agreed with the finite-step dense-gate reference within `8.3e-15` on the
recorded local values. Both finite-bond implementations need not choose an
identical approximation; Pepsy performs numerical protected-rank detection,
whereas the tested ITensor implementation uses a QR basis without rank
detection. The precise cause of each finite-bond difference was not isolated.

For six-site mixed-field Ising at cap 16, initial energy was `0.41475`.
At `t = 2`, Pepsy / ITensor energies were `0.4147858244 / 0.4147858231`
with `dt = 0.1`, and `0.4147590304 / 0.4147590249` with `dt = 0.05`.
Exact Hamiltonian evolution retained `0.41475`. Thus halving the step
reduced the energy drift approximately fourfold in this case; DMT protects
the energy against each truncation, not the product formula's timestep error.
Maximum trace errors were below `1.7e-13` in these runs.

Pepsy's `cutoff='auto'` and zero cutoff produced identical recorded double
precision values in the six-site mixed-field cap-10 case; this is a single
case, not a general equivalence claim. Temporary scripts, raw CSV data and the
isolated Julia manifest are under `/private/tmp/pepsy-itensor-compare.HBn1al`.
They may be removed by the system. The focused command
`python -m pytest -o addopts='' -q tests/test_pauli_dmt.py tests/test_pauli_dmt_cutoff.py`
passed 97 tests with six Torch rank-deficient QR warnings. The default smoke
selection had deselected all 97, so the marker filter was explicitly disabled.
No full Pepsy test suite or speed benchmark was run, and these results do not
establish positivity after DMT truncation.
