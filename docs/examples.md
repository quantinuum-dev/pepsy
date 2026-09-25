# Examples

Start with a tutorial, then use the repository scripts for larger examples.

## Walkthroughs

| Task | Guide |
| --- | --- |
| Prepare and contract a PEPS norm | [First contraction](tutorials/contract_norm.md) |
| Interpret contraction diagnostics | [Fidelity diagnostics](tutorials/fidelity_diagnostics.md) |
| Combine simple update and belief propagation | [Simple update and Relay-BP](tutorials/bp_simple_update_relay.md) |
| Set bond dimensions and sweep counts | [Choose parameters](howto/choose_parameters.md) |
| Configure an optimization solver | [Tune sweep solvers](howto/solver_tuning.md) |

## Repository scripts

The [examples directory](https://github.com/quantinuum-dev/pepsy/tree/develop/examples)
contains runnable scripts. See the
[operator examples](https://github.com/quantinuum-dev/pepsy/blob/develop/examples/operators/README.md)
for MPO/PEPO construction and ordered products.

Other examples in this checkout:

- [Pauli MPO and flat PEPS contraction](https://github.com/quantinuum-dev/pepsy/blob/develop/examples/pauli_mpo_trace_flat_peps.py).
- [MPS magnetization notebook](https://github.com/quantinuum-dev/pepsy/blob/develop/examples/MpsMagnetization/mps_simulator.ipynb).

For qMERA workflows, use the examples in the [qMERA API guide](api/optimizers/qmera.md).

Larger workflows and experiment scripts are maintained separately in
`pepsy_examples`.
