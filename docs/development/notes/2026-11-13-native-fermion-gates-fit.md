# Native fermionic gates and reversed-gate one-site FIT

The Etienne replay exposed independent defects in local exponentiation and
native one-site FIT. The Hubbard hopping, onsite interaction, chemical
potential, charge convention and source angles are unchanged.

## Local exponential

Sparse generators such as `Sx` and doublon number omit all-zero charge blocks.
The exponential must contain identity on these blocks. Filling after fusion
is insufficient: fusion can already have removed component sectors within a
total-charge block. `_gate_from_term` now fills an owned native copy before
reshaping into a matrix, then delegates exponentiation and fermionic phases
to Symmray. The cached observable remains sparse and unchanged.

Tests compare one-site real/imaginary-time exponentials with independent
SciPy exponentials for U1, U1U1, Z2 and Z2Z2. Two-site zero-angle tests include
holes/doublons across a noncontiguous bond. Existing spin-gate unitarity tests
cover finite-angle two-site generators. The native PEPO reference sandwich
also needs its sparse observable filled before fusion so its within-charge
matrix dimensions describe the same physical space as the unitary.

## One-site FIT

A short native circuit reproduces the error: start from four half-filled
sites, apply hopping on `(0,3)` and `(1,2)`, then fit hopping on `(1,0)`.
The final reversed gate changes a virtual leg's direction. A site's two
environment legs can then both be dual or both nondual.

The old correction assumed exactly one of two present legs was dual. This
changed a state already equal to the target: the captured Etienne update went
from fidelity one to approximately .65276214 in three one-site sweeps.
`FIT._prepare_fermionic_effective_tensor` now corrects each dual environment
leg independently. Oppositely oriented legs retain the previous behavior;
equal orientations get zero or two corrections as their metadata requires.
No dense conversion, extra block sweep, target substitution or direct
compression fallback was added.

Native MPS initialization now establishes canonical form directly instead
of invoking Quimb's numerical identity scan. A dense `eye(d)` cannot supply
the charge-index metadata required by native Symmray arrays. The reversed-gate
regression constructs a fresh optimizer on the entangled incoming state and
tests NumPy/Torch, U1/U1U1, R/L/RL one-site sweeps, and named block modes.
One-site checks require the actual FIT backend with zero block warm-up and
no fallback.

## Dependencies and upstream consultation

The environment changed during this session. Final checks used:

| Package | Version / Git revision |
| --- | --- |
| Autoray | `0.11.1.dev17+g7a474cab3`, `7a474cab367dd0be2acbb1b80e9279da5c8bd528` |
| Quimb | `1.16.1.dev1+g3371ff40c`, `3371ff40c25d52e6bf60cbf3766918c782dad9f4` |
| Symmray | `0.5.1.dev2+g20035beb7` |
| Cotengra | `0.8.4.dev8+g646239731` |
| Torch / NumPy | `2.14.1+cu126` / `2.5.2` |

The installed GitHub Autoray resolves `ar.do("stop_gradient", numpy_array)`.
All 36 native replay checks previously blocked by Autoray 0.8.4 passed after
the upgrade. No local dispatch shim or dependency-constraint relaxation was
needed; the existing package requirement is already `autoray>=0.9`.
The source changes here do not perform an environment upgrade.

Consulted the official [Autoray source](https://github.com/jcmgray/autoray),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/), and
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html).
The Symmray ReadTheDocs pages were unavailable through the browsing tool;
used the [official README](https://github.com/jcmgray/symmray/blob/main/README.md)
and installed `fill_missing_blocks`, exponential and native decomposition
implementations. **Adopt:** public sparse-block filling and upstream dispatch.
**Compatibility correction:** native canonical initialization without an
unsupported dense identity scan. No installed library was patched.

## End-to-end evidence

`/tmp/pepsy_etienne_fixed_validation.py` generates bounded periodic Etienne
preparation streams from the sibling example and appends native real-time
Hubbard Strang gates. Original serialized author boundary files are absent;
this is a generated-stream regression, not a new paper-reference result.
All arrays remain native fermionic tensors, with normalization disabled.

- **2x2, chi=16, 68 gates:** direct, dmrg, dmrg2 and dmrg3 agree with native
  exact replay to floating-point precision; norms differ from one by less
  than 2e-11.
- **4x2, chi=256, 136 gates:** dmrg and dmrg2 agree with direct replay with
  fidelities .9999999999999868 and .9999999999999875, respectively; raw norm
  errors are below 4e-11.

The complete reports are `/tmp/pepsy_etienne_fixed_validation.json`,
`/tmp/pepsy_fermion_fix_environment.json`, and the focused/domain test logs
listed in the corresponding session handoff. These results do not establish
finite-chi convergence, CUDA replay or a production lattice's accuracy.

## Regression status

The non-slow symmetric-tensor/native-MPS/MPS/MPO selection covered 662 passing
cases and one skip, with 30 slow cases deselected. Its full run initially
reported 661 passes and one sparse-reference fusion failure in
`test_fermion_to_mpo_accepts_arbitrary_neutral_term_support`; filling that
reference's physical zero components before fusion resolved the final case
in a focused rerun. No assertion or tolerance was weakened. The default
smoke suite passed 112 tests. Ruff and `git diff --check` passed.

The full-package attempt was interrupted after 622 passes, 66 skips and two
failures in `test_fermionic_open_rho_sweep_supports_native_symmetries` (U1 and
U1U1). Both fail while densifying an empty Symmray BP density term. Repeating
them in a private source copy with all three implementation fixes restored
to HEAD reproduces both failures. They are pre-existing BP issues, outside
the gate/FIT repairs; the full suite is not claimed green. The separate
3x4 exact-evolution stress run was also interrupted and is not certified.
