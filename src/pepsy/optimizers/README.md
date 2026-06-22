# pepsy.optimizers

This package contains the high-level optimizers that sit above Pepsy's tensor,
operator, boundary, and solver layers. The core package remains `pepsy`;
`pepsy_examples` is for examples and external testing, and `tc_gauge` is an
important downstream time-compression consumer that depends on Pepsy behavior.

## Layout

- `_shared.py`: cross-optimizer helpers that are genuinely shared.
- `mps/`: MPS gate-stream optimization.
  - `optimizer.py`: `MpsOptimizer`.
  - `compression.py`: extraction target for compression backends.
  - `normalization.py`: extraction target for non-unitary normalization logic.
  - `diagnostics.py`: extraction target for fidelity/progress records.
- `mpo/`: MPO gate-stream optimization.
  - `optimizer.py`: `MpoOptimizer`.
  - `targets.py`: extraction target for gate-pair and DMRG target builders.
  - `compression.py`: extraction target for compression backends.
- `peps/`: PEPS/PEPO gate-stream optimization.
  - `optimizer.py`: `PepsOptimizer`.
  - `gates.py`: extraction target for gate routing and target application.
  - `warmstart.py`: extraction target for warm-start construction.
  - `routing.py`: extraction target for sweep/global backend routing.
  - `diagnostics.py`: extraction target for infidelity/progress records.
- `sweep/`: local PEPS slice optimization.
  - `optimizer.py`: `SweepOptimizer`.
  - `environments.py`: Quimb MPS boundary store and engine selection helpers.
  - `local_objective.py`: extraction target for local objective assembly.
  - `traces.py`: extraction target for sweep traces and progress summaries.
- `global_opt.py`: whole-network variational optimization helpers.

## PEPS optimizer stack

`PepsOptimizer` is the outer gate-stream driver. It builds exact two-site
targets, compresses warm starts to the requested PEPS bond dimension, and then
optionally refines with `SweepOptimizer` or `GlobalOptimizer`.
It exposes `boundary_engine` and `boundary_options` so sweep cleanup can use
the same boundary implementation choices as `SweepOptimizer` directly.

`SweepOptimizer` is the local PEPS slice optimizer. It keeps two environment
stores:

- `bdy` for the trial norm `<state|state>`.
- `bdy_overlap` for the overlap `<target|state>`.

The current default store is `pepsy.boundary.states.BdyMPS`, whose `mps_b`
dictionary contains reusable boundary MPS entries keyed as `Y{i}_l`,
`Y{i}_r`, `X{i}_l`, and `X{i}_r`. `SweepOptimizer` selects a row or column,
attaches the needed left/right environments, optimizes the packed local slice,
and then advances the boundary for the next slice with
`pepsy.boundary.sweeps.CompBdy`.

## Boundary engines

The default dense Pepsy boundary engine is:

```text
build_bra_ket(...) -> BdyMPS(...) -> CompBdy.move_bdy/move_step_bdy(...)
```

That path uses local FIT/DMRG-style boundary updates and works well for the
dense backends it was designed around. It is less suitable for Symmray-backed
networks, where Quimb's native boundary contraction and environment routines
can preserve backend semantics better.

`SweepOptimizer` also supports `boundary_engine="quimb-mps"` (or `"auto"` for
Symmray-looking inputs). This builds local row/column environments with
Quimb's `compute_x_environments(...)` and `compute_y_environments(...)`, while
scalar sweep-time normalization and infidelity use Quimb's
`contract_boundary(...)` through Pepsy's public `method="mps"` metric helpers.
`PepsOptimizer(boundary_engine="auto")` keeps this same policy when it delegates
to sweep cleanup.

## Import style

Use clean class imports at API boundaries:

```python
from pepsy.optimizers import MpsOptimizer, PepsOptimizer, SweepOptimizer
```

Use implementation leaves when a test or internal change needs module globals:

```python
from pepsy.optimizers.sweep.optimizer import SweepOptimizer
```

## Editing notes

- Prefer package namespaces such as `pepsy.boundary`, `pepsy.optimizers`, and
  `pepsy.tensors`; do not revive removed root-level flat modules.
- Add focused tests near the optimizer or boundary behavior being changed.
- For Symmray behavior, keep optional dependencies optional with
  `pytest.importorskip(...)`.
