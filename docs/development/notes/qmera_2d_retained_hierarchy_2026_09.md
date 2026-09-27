# 2026-09-25: retained-register 2D spin qMERA

The opt-in `hierarchy="retained"` builder keeps an ordered physical-wire
register per virtual 2D coarse cell. An isometry block groups child cells,
executes local two-qubit brickwall sweeps, and retains up to `bond_qubits`
wires. Disentanglers join matching register positions across x/y block
faces. Colored boundary rounds execute in round order. Preparation executes
coarse to fine, block circuit then boundary circuit; the adjoint is the
requested fine-to-coarse disentangler then isometry order. The existing
site-retention 2D and explicit-mode fermion paths remain the default.

Each grid axis absorbs a one-cell tail into the preceding block, following
the existing 1D retained-register rule. For example `5 / 2 -> 2 + 3`,
`7 / 3 -> 3 + 4`, and `9 / 4 -> 4 + 5`. The Cartesian product covers
every child cell without a singleton edge, including periodic grids and
rectangular 2x3 blocks. `QMeraLayerSpec.input_grid_shape`,
`output_grid_shape`, `isometry_cell_blocks`, and `retained_registers`
make the hierarchy inspectable. Balanced retention spreads choices along
ordered block wires; explicit retention keeps the 1D `(scale, block)` key.

Scope: spin qubits, local brickwall unitary-completion circuits, boundary-face
disentanglers of width 2 on both axes. This does not add rectangular isometry tensors.
Physical-wire supports let the existing direct, lightcone, compiled Cotengra,
Torch, JAX, and GradientOptimizer paths consume the new schedule without a
new backend implementation. Large 4x4-block contraction cost has not been
benchmarked.

Upstream audit reused the installed Pepsy environment: Quimb
1.15.1.dev66, Autoray 0.11.1.dev3, Cotengra 0.8.3.dev7, Symmray
0.4.1.dev7, NumPy 2.5.2, Torch 2.6.0, and JAX 0.10.2. Installed
`TensorNetwork.gate_inds` and `cotengra.array_contract_expression`
signatures were checked. Official
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray) were inspected.
The [Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable; native Symmray behavior was left on the existing 2D path.
Classification: **adopt** the existing upstream gating/contraction APIs; no
shim or dependency change.

Validation: `tests/test_optimize_qmera.py` passed 88 tests. The new cases
check 2x3, 3x3, and 4x4 odd-grid block coverage, periodic seams, explicit
retention, direct/eager/compiled local-energy agreement, Torch gradients,
and JAX JIT gradients. A separate Torch `aot_eager` compile probe on an
odd 2D hierarchy matched eager energy and gradients; upstream graph-break
warnings remain. Public API/layout tests passed 57 cases with the installed
distribution-version check excluded: this checkout declares 0.5.0 while
shared environment metadata reports an older version. Ruff and
`git diff --check` passed. The full repository suite was not run.
