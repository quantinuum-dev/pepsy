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

## Runnable scripts

- `examples/QMeraEnergy/qmera_energy.py` builds a small schedule-first qMERA,
  evaluates local ZZ energy from rebuilt and compiled lightcones, and can run a
  short optional Torch Adam optimization with `--steps`.
- `examples/SimpleUpdateGen/long_range_peps.py` demonstrates a long-range PEPS
  simple-update term: raw `quimb.SimpleUpdateGen` fails on the non-adjacent
  endpoints, while `pepsy.SimpleUpdateGen` routes the gate through SWAPs and
  reports the resulting max bond, gauge count, and cluster energy.
- `examples/RelayBP/simple_update_relay_comparison.py` compares exact
  contraction with plain D1BP, simple-update-initialized D1BP, and
  simple-update-initialized Relay-BP on a small loopy factor network.
- `examples/RelayBP/odd_cycle_stress.py` provides deterministic positive
  odd-cycle cases where polarized parallel D1BP stalls and Relay-BP converges.

For a cleaner, docs-first narrative, start from [tutorials](tutorials/index.md).
