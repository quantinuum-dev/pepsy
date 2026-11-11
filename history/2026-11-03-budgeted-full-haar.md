# Budgeted full Haar vectors

Branch: `develop`; baseline `ad523b6`. Implemented in the commit containing
this handoff; no release is implied.

The requested Gaugy 23-spin full-Haar comparison requires an explicit opt-in
beyond the historical 20-spin guard. `haar_random_state` now accepts the
keyword-only `max_dense_bytes` together with `L_max>20`. It validates the
budget and a six-vector estimate before RNG allocation, preserving seeded
samples and the default guard. See the [API](../docs/api/tensors/constructors.md).

Validation: constructor suite **28 passed**, including a normalized 21-spin
sample and rejected budgets before allocation. Full package suite was not
run. Public API/layout checks: **55 passed**, two existing alias deprecations.
Downstream CPU/CUDA small-system overlaps are independently checked in
`gaugy_examples/tests/test_eli_mps_haar.py`; production measurements belong
to that repository.
