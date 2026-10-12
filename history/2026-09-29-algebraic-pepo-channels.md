# 2026-09-29 — Pauli algebra and exact PEPO edge delinearisation

- Scope: implement the authorized next PEPO reduction without changing the
  ordered local exponential/connected-cluster target or autodiff contract.
- Branch / baseline: `develop`, `c1ff0f8`.
- Commit status: Pepsy work remains unstaged and uncommitted. No push.
  Existing code, journals and numerical MPO delinearisation work preserved.
  Gaugy's selective local documentation/test commit is `5da6a0b`.

## Implemented

New opt-in `preparation="algebraic"` uses the declared qubit Pauli algebra
on each cluster, independent of coefficient values. Exact rational
elimination selects linearly independent complete edge slices and transfers
constant combinations to the other endpoint. Reduction happens before
square routing and again afterwards, preserving branching/loop connectivity
and the protected singleton rail. Full channels require no SVD or QR in
preparation or replay. Optional reference QR caps remain approximations.

Replay projects onto the proved residual algebra with fixed Walsh
contractions, then uses existing fused tensor assembly. Arbitrary external
residual matrices are explicitly projected by `bind_assembler`. Public
evaluation validates the Pauli inventory. Local dense exponentials and
connected subtraction remain unchanged. Gaugy only forwards the mode and
measures the actual returned PEPO.

Implementation: `_cluster_channel_pauli.py`, `_cluster_channel_linear.py`,
`_cluster_channel_pepo.py`, `_cluster_channel_assembly.py`,
`cluster_channels.py`. Tests: `test_cluster_channel_algebra.py`, expanded
`test_cluster_channel_pepo.py`, and Gaugy's `test_cluster_channels.py`.
See the [API](../docs/api/operators/cluster_channels.md) and
[evidence](../docs/development/notes/2026-09-29-algebraic-pepo-channels.md).

## Measurements and limits

The benchmark has 72 rows across reference/symbolic/algebraic preparations,
one/six references and caps 1/4/full. Relative to identical-slice sharing,
exact loop bonds shrink 9→7 and one higher-body bond 25→13. Branch/crossing
cases gain no further chi reduction. Preparation is slower in these small
controls, and replay improvement is not established. Keep the mode explicit.
Full-channel operator/gradient relative errors are ≤1.85e-16 / 2.61e-15
against the same selected cluster target. Tight-cap errors depend on gauge.

This is local exact reduction, not global symbolic minimality. Pauli
closure, fixed factors and remaining routed blocks are still enumerated.
Rational fill-in, local exponentials and routing width can grow. No large
lattice scaling, environment-optimal compression, native/fermionic PEPO or
end-to-end Torch exponential capture claim is made.

## Validation

Shared Python 3.12 environment, single BLAS/OpenMP thread, CPU JAX:

- Focused algebra/PEPO checks: **49 passed**, 21.32 s. Includes arbitrary
  in-algebra residuals and adjoints, independent exact ordered-product
  references, zero coefficients, Torch CPU/CUDA and single precision, JAX
  trace JIT and Torch full-graph `aot_eager` assembly capture/backward.
- Broader Pepsy selection: **274 passed**, two existing compatibility
  warnings, 88.90 s. Files: the above, `test_cluster_channels`,
  `test_cluster_channel_assembly`, `test_cluster_channel_structure`,
  `test_cluster_channel_frontier`, `test_pepo_routing_compression_reports`,
  `test_graph_pepo_autodiff`, `test_graph_pepo_product`,
  `test_square_cluster_plan`, `test_public_api`, `test_package_layout`,
  `test_mpo_delinearize`.
- Gaugy downstream: **167 passed**, six existing Quimb warnings, 45.14 s,
  across channel, joint-parity, binding, materialization-option,
  exponential-API and package tests.
- Ruff on `src`, `tests` and the benchmark; Gaugy Pyflakes and whitespace
  checks pass. These are scoped selections, not full repository suites.

The first compile probe failed in an Autoray argument-translation wrapper;
fixed 2-by-2 contractions resolved it without modifying upstream code.
Same-task upstream audit reused with unchanged dependencies. Logs:
`/tmp/algebraic-pepo-focused.log`, `/tmp/algebraic-pepo-regressions.log`,
`/tmp/gaugy-algebraic-pepo.log`. Timed benchmarks had no concurrent tests.

Concurrent MPO `preparation="automaton"` work appeared in shared helpers
during validation. It was preserved and is not part of this PEPO handoff's
implementation claims. A final compatibility run follows those shared edits;
its record is appended below. Coordinate future shared-file edits with that
separate work instead of reverting its added modes or cluster arguments.

Final compatibility run after those edits: **134 passed**, 60.56 s, covering
`test_cluster_channel_algebra`, `test_cluster_channel_pepo`,
`test_cluster_channels`, `test_cluster_channel_assembly`,
`test_cluster_channel_structure`, and `test_cluster_channel_frontier`.
These overlap the broader 274 checks and are not additive. Log:
`/tmp/algebraic-pepo-final.log`. Local documentation links, Ruff and final
whitespace checks pass. No Pepsy changes were staged or committed.
