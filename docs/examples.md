# Examples

Start with a tutorial, or use the repository's MPS magnetization notebook.

## Walkthroughs

| Task | Guide |
| --- | --- |
| Prepare and contract a PEPS norm | [First contraction](tutorials/contract_norm.md) |
| Interpret contraction diagnostics | [Fidelity diagnostics](tutorials/fidelity_diagnostics.md) |
| Combine simple update and belief propagation | [Simple update and Relay-BP](tutorials/bp_simple_update_relay.md) |
| Set bond dimensions and sweep counts | [Choose parameters](howto/choose_parameters.md) |
| Configure an optimization solver | [Tune sweep solvers](howto/solver_tuning.md) |

## Repository example

The [examples directory](https://github.com/quantinuum-dev/pepsy/tree/develop/examples)
contains the MPS magnetization example:

- [MPS magnetization notebook](https://github.com/quantinuum-dev/pepsy/blob/develop/examples/MpsMagnetization/mps_simulator.ipynb).

For operator construction, use the [API reference](api/index.md). For
qMERA workflows, use the examples in the [qMERA API guide](api/optimizers/qmera.md).

Larger workflows and experiment scripts are maintained separately in
`pepsy_examples`.
