# 2026-10-14 — Pepsy FIT in NetKet boundary differentiation

NetKet `dmrg`/`dmrg2` now dispatch to the public
`pepsy.fitting.compress_jax_fit` adapter, with one-/two-site blocks.

The default `differentiation="auto"` converges dense boundaries through
`FunctionalFIT` and uses its checked real-coordinate adjoint solve. Both
solves must satisfy their residual tolerances. `max_iterations` counts
complete RL cycles (default 100); an explicit `unrolled` mode differentiates
finite directional sweeps (default two). The target is contracted exactly
within each site, with no target-bond truncation. Tags, outer legs and stored
scales survive reconstruction. The dense default can change amplitude values
and evaluation cost compared with the preceding two-sweep Quimb alias.

The existing [upstream audit](2026-10-14-functional-fit-and-native-compression.md#upstream-audit-and-decisions)
applies to the unchanged dependency stack. Quimb's installed public callable
dispatch and compression signatures were inspected for this integration.
This is an **adopt** decision: no upstream registry mutation or vendoring.

Native flat implicit alignment remains **deferred**. A temporary blockwise
polar prototype encountered configuration-dependent metadata escaping custom
VJP closures, graded dummy-parity tracing, and failed independent one-site
complex derivative checks. That prototype was removed. Native `auto` retains
the existing Quimb finite-sweep FIT; explicit native implicit requests raise.
No native implicit gradient guarantee is claimed.

JAX batches conditional QR pullbacks into masked branches containing
`stop_gradient`, preventing their transposition in an implicit solve.
Primal selection with a nonsingular inactive native chart now keeps both
pullbacks linear and transposable. Mixed singular/resolved batch regressions
cover the registered QR and explicit `vmap` routes.

See the [owning API](../../api/vmc.md#jax-boundary-compression) and
[validation handoff](../../../history/2026-10-14-netket-implicit-fit.md).
