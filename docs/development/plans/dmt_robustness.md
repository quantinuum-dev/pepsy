# DMT robustness, efficiency and accuracy

User-authorized workstream, in the order below. Baseline: `develop` at
`eaedf789`, including the uncommitted region-contraction review changes.
Use the [DMT API](../../api/optimizers/pauli_dmt.md) as the current contract;
this plan records work, not a claim that future features are implemented.

| Order | Work | Status / acceptance |
| --- | --- | --- |
| 1 | Isolate large-chi numerical/native failures | Identified native Accelerate QR heap corruption; scoped exact workaround passes clean-exit checks and 6×6 chi192 through t=0.3 |
| 2 | Reduce QR/SVD and distant-gate overhead | Exact-fit map/rank work removed; public exact interval transfer methods now integrated in both drivers; gate-stream routing remains separate |
| 3 | Cache static rank-aware candidates | Implemented bounded per-cut immutable plans; remapping/copy/dynamic-rank tests pass |
| 4 | Improve physical-region selection | Implemented `complete_dmt_regions` with explicit required supports and subset elimination; preserves all requirements |
| 5 | Establish matched trajectory convergence | Measured 3×2/3×3 MPS/DMT/FIT dt/chi studies through t=2; chi256 matches exact 3×3 circuit, second-order dt error; 6×6 remains unconverged |
| 6 | Assess and prototype positivity-preserving representation | Historical control only; user explicitly declined the XX† direction on 2026-11-20. Do not pursue constrained positive-factor fitting in this workstream |

Each stage needs focused regression checks and a dated evidence record before
promotion. A failed or unconverged run is evidence, not an accuracy reference.
If a stage is unresolved, document its limits before starting dependent work.

[Evidence and reproduction](../notes/2026-11-20-dmt-robustness.md) record this
first implementation/research pass. The
[Torch interval follow-up](../notes/2026-11-20-dmt-torch-interval.md) records
the public integration and CPU factorization controls. Current priorities are
longer Torch 6×6 chi/dt convergence. CuPy is explicitly deferred by the user
until a suitable environment is available. The NumPy/Accelerate workaround is scoped
maintenance, not evidence of Torch/CuPy large-chi stability. The positive-factor
example remains a historical research control, outside the active direction.
