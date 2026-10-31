# 2026-10-30 — batching independent prepared SU schedules

Baseline: `develop` at `61a77d8`. The user requested further Torch compiler
and loop improvements before considering a JAX implementation.

`PreparedSimpleUpdate.apply_many` batches compatible ready two-site updates
across independent schedules, preserving each site's order and each plan's
cutoffs and fallback. The single-plan `apply` uses the same resumable executor.
Weight conditioning is batched within an update and unnecessary default-weight
allocations are avoided. No decomposition, stabilization or normalization
formula changes. The [API guide](../docs/api/operators/gates.md#prepared-dense-simple-update)
owns the public contract.

The [downstream record](../../gaugy_examples/research/diagnostics/2026-10-30-pepo-torch-scheduling/README.md)
contains profiles, compiler ablations and matched timings. Complete nonzero-state
GPU value/gradient time changes from 3.392 to 2.926 seconds, with identical
loss and a maximum gradient difference of `7.08e-17`. Two-site kernel calls
fall from 104 to 50 without changing individual gates.

The repeated downstream 30-step fit completes in 91.954 seconds with 32
evaluations, versus 110.208 seconds in the prior saved prepared fit. Final
cost differs by `3.41e-13` and coordinates by at most `4.51e-10`. These are
budget-limited fits, not convergence or physical-fidelity certificates.

Inductor remains excluded. Direct AOT graph lowering passes small CUDA cases
but fails the CPU physical-gradient check and the full saved GPU gradient.
The CPU ablation passes with only forward compilation or decompositions alone,
and fails when the backward is lowered. Forward-only compilation agrees on
saved full gradients but is slower than eager execution. No failed compiler
implementation is included in the package.

Validation: 174 focused tests across prepared replay, heterogeneous schedules,
per-plan rank caps, CPU/CUDA/AOT-eager, scale extraction, zero support, public
API and package layout; downstream Gaugy's affected selection passes 239.
The existing singular-chart/deprecation warnings remain. This is not a full
suite and does not resolve the previously documented native-Z2 failures.
Before publication, isolated exports of the selected Git indexes also pass
nine shared-schedule tests and all five downstream prepared-engine tests.

Installed versions are unchanged: Torch 2.14.1+cu126, Quimb 1.15.1.dev101,
Autoray 0.11.1.dev14, Cotengra 0.8.3 and Symmray 0.4.1.dev25. The upstream
Quimb/Cotengra changelogs, Autoray source and Torch backend guidance were
consulted; the Symmray documentation endpoint was unavailable. No dependency
or installed-library changes were made. Classification: adopt shared schedule
execution; retain compiler adapters as diagnostics; defer Inductor and JAX.
