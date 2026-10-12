# 2026-10-08 — Fix strip caps; compare Quimb and larger refinement

- Scope: user authorized the preceding review fix, requested comparison with
  Quimb FU and rereading arXiv:1405.3259 for stability, and asked for an
  efficient refinement beyond the reduced pair with caching in mind.
- Branch / baseline: `develop`, `ad8ed05`, with pre-existing working-tree
  changes preserved. Nothing staged, committed, or published.
- Implemented separate norm/overlap caps for fixed/adaptive strip refinement,
  checked overlap/target handoff, and cache provenance after index alignment.
  Added four Torch/CuPy regressions and API/changelog documentation.
- **206 focused tests passed**, no skips; Ruff and whitespace checks passed.
  No fresh full-repository suite or large-D/long-time certification.
- Inspected installed Quimb FU and generic ALS, reread paper III B/Appendix B,
  and inspected figures. Quimb full-tensor FU is a larger variational space
  than our reduced-pair FU; current strip refinement already changes full
  tensors. Eigenvalue flooring differs from support-truncated pseudoinverses.
- Measured a temporary shared-layer-target refinement prototype using the
  existing strip kernel: three 3x3 D2 tests gave 28–33% lower global infidelity
  than current local-strip refinement, at 8–14% higher single-run CPU time.
  This is a proposed next integration, not a new production mode.
- Broader-region design and stability suggestions remain prototypes. Do not
  infer authorization to change solver policy or defaults from this handoff.

The [detailed evidence and proposed cache-aware design](../docs/development/notes/2026-10-08-full-update-caps-quimb-refinement.md)
records exact numerical results, source/environment limitations, test scope,
and temporary reproductions. This implements the cap finding in the
[preceding review](2026-10-08-full-update-strip-cap-review.md).
