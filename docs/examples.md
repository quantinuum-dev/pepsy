# Examples

## Primary walkthrough

- [Tutorial: Contract a PEPS Norm](tutorials/contract_norm.md)
- [Tutorial: Simple-update Initialization and Relay-BP](tutorials/bp_simple_update_relay.md)

This is the recommended first end-to-end walkthrough for interactive usage.

It demonstrates:

- preparing `ket` and `norm` with `build_bra_ket`
- initializing boundaries with `BdyMPS`
- running `contract_boundary(...)`
- inspecting `BoundaryContractResult.cost` and `.fidel`

## Additional walkthroughs

- [Tutorial: Fidelity Diagnostics](tutorials/fidelity_diagnostics.md)
- [How-To: Choose Parameters](howto/choose_parameters.md)
- [How-To: Tune Sweep Solvers](howto/solver_tuning.md)

## qMERA examples

- `examples/qmera_scale_plan_6x6.py` — generic heterogeneous 6x6 PBC RG
  schedule with scale-specific blocks.
- `examples/qmera_fermion_hubbard_2d.py` — native `U1U1` 2D multimode
  Fermi--Hubbard schedule, grouped cones, and direct-state validation.
- `examples/qmera_fermion_hubbard_4x4_pbc.py` — explicit 4x4 PBC square
  disentangler/isometry RG schedule.
- `examples/qmera_majorana_2d.py` — native `Z2` Majorana and pairing gates.

## Extended examples

The package repository keeps lightweight runnable examples under `examples/`.
Larger runnable workflows and experiment scripts are maintained
in the separate `pepsy_examples` repository. The tests in this repository keep
the deleted Relay-BP examples' numerical coverage without depending on local
example files.

For a cleaner, docs-first narrative, start from [tutorials](tutorials/index.md).
