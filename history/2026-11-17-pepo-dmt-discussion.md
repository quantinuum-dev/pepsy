# Native PEPO DMT discussion

The user asked for design thinking, not PEPO implementation. Confirmed that
`PepsOptimizer` accepts native PEPOs with full update and SU warm starts, and
that finite-cluster SU/BP PEPO update paths already exist. No dedicated
PEPO DMT mode or local-density-preservation guarantee is currently provided.

The [design note](../docs/development/notes/2026-11-17-pepo-dmt-design.md)
distinguishes SU norm weights from physical partial-trace environments, the
nonseparable loop case, when an MPO-like protected SVD could apply, and
independent accuracy/performance validation. No numerical behavior changed.

The user also requested committing and pushing the preceding original DMT
precision/API review. Include that focused change and this design note; leave
unrelated sampling code, sampling changelog and local policy uncommitted.
