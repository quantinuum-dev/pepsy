# 2026-10-05 — CUDA MPS trajectory review

- Scope: extend the performance/API review with actual CUDA measurements.
- Branch / baseline: `develop` / `bec773a`, with prior uncommitted fixes.
- Commit status: new review documentation only; production code unchanged in
  this task. Nothing staged, committed or published.

Confirmed an RTX A5000 and Torch CUDA execution. Compared synchronized NumPy,
Torch CPU and Torch CUDA trajectories at three MPS sizes. CUDA became faster
at the tested chi=128 case; extra local GPU workers did not help. Temporary
shared-block/batched-Kraus prototypes reduced kernel time about 4–4.7x and
adjacent-event stream synchronizations from nine to one. These are proposed
optimizations, not integrated changes or full-run speedup promises.

See the [CUDA review](../docs/development/notes/2026-10-05-mps-trajectory-cuda-review.md)
for exact workloads, environment, measurements and API proposals. Dense
probabilities, small-shot CPU/CUDA histories and final states, and device/dtype
preservation checks passed. Existing CUDA/CuPy selection: 16 passed, two known
infidelity-comparison failures, 48 deselected. No full-suite rerun or large-state
dense-reference validation. Documentation links and diff checks passed.
