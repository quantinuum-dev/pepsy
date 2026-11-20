# 2026-11-19 — Full-update boundary FIT planner and warm reuse

Pepsy `develop` at `e2857dc` and `pepsy_examples/main` at `2398e96` are the
baselines. The changes are local. Full update now forwards the existing
boundary FIT contraction planner override. The Eli runner defaults that
planner to `auto-hq` in one-site DMRG and retains the explicit `greedy` option.

The D=6, χ=120 GPU audit measured nine cache hits on the second gate and
verified that a later invalidated cut refits from its previous MPS. See the
[timing and validation record](../docs/development/notes/2026-11-19-full-update-warm-boundary-fit.md)
and the [full-update API](../docs/api/optimizers/peps.md#boundary-caps).
