# 2026-11-17 — PEPO all-L and global-objective probes

- Scope: improve all-L protection/global objective understanding and test
  different sizes and symmetry observables. Preserved pre-existing unrelated
  sampling changes; no commit or push requested.
- Added opt-in `dmt_svd_mix_seeds` for weighted objectives. A candidate mixing
  a discarded singular direction is accepted only if all hard maps pass, the
  weighted cost improves, and the optional pair-error bound permits it.
- On 2×2 and 2×3 parity densities, protecting every L and using the full
  physical density as objective, one mixed seed recovers the full state to
  numerical precision while reducing a bond from 2 to 1. This depends on
  environment-null cross terms and is not a general PEPO algorithm. The
  default pair-growth guard rejects that seed; the test disables it explicitly.
- One 2×3 TFIM timestep with all eight L constraints agrees with independent
  state-vector densities; it has no meaningful loss. Its next step can fail
  the strict solver. The 4×5 all-L workload still exceeds the map budget.
- Extended the reproducible TFIM example to 2×3, 3×3 and 4×5 and measured
  Ising energy, spin-flip-odd Z magnetization, and X-reflection defect. A 3×3
  all-local soft objective improved several observables compared with its
  feasibility-only result, but neither positivity nor general improvement is
  guaranteed. See [results](../docs/development/notes/2026-11-17-pepo-all-l-symmetry.md).
- Focused PEPO, 6×6, Ising example, joint-update and public-API suites passed
  121 tests with 8 accelerator skips after the final changes. Ruff and
  `git diff --check` passed. No full-suite or large-system guarantee claimed.
