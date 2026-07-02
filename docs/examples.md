# Examples

## Primary walkthrough

- [Tutorial: Contract a PEPS Norm](tutorials/contract_norm.md)

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

- `examples/SimpleUpdateGen/long_range_peps.py` demonstrates a long-range PEPS
  simple-update term: raw `quimb.SimpleUpdateGen` fails on the non-adjacent
  endpoints, while `pepsy.SimpleUpdateGen` routes the gate through SWAPs and
  reports the resulting max bond, gauge count, and cluster energy.

For a cleaner, docs-first narrative, start from [tutorials](tutorials/index.md).
