# pepsy.boundary

This package owns PEPS-like boundary contraction, normalization, and fidelity
helpers. It is one of the central layers used by `pepsy.optimizers`, and API
changes here should be made carefully because downstream packages such as
`tc_gauge` rely on Pepsy boundary behavior.

## Modules

- `metrics.py`: public norm, overlap, normalization, fidelity, and
  infidelity helpers.
- `states.py`: `BdyMPS`, the reusable boundary-MPS store.
- `sweeps.py`: `CompBdy`, the FIT/DMRG-style boundary update engine.

## Core flow

The standard Pepsy boundary path is:

```text
build_bra_ket(ket, bra?) -> BdyMPS(...) -> contract_boundary(...)
```

`build_bra_ket(...)` prepares a tagged double-layer tensor network. The ket is
tagged in place with `KET`; the bra layer is tagged with `BRA`; shared internal
bra indices are reindexed with an `_*` suffix.

The word `flat` has a specific meaning in this package. `tn_flat` and
`flat=True` refer to one already-flattened effective lattice layer whose local
bra/ket or operator contractions are complete. They do not mean an arbitrary
stack of separately tagged layers. `tn_double` and `flat=False` are the
multi-layer boundary path, normally used for the tagged BRA--KET network.

`BdyMPS` initializes reusable row and column boundary MPS environments. Its
`mps_b` dictionary uses keys like:

- `Y{i}_l` and `Y{i}_r` for column-sweep environments.
- `X{i}_l` and `X{i}_r` for row-sweep environments.

`CompBdy` updates those environments with `move_bdy(...)` or
`move_step_bdy(...)`, then contracts a final boundary network in `run(...)`.

The local boundary solver is selected with `fit_mode`:

- `"direct"`, `"src"`, `"src-mps"`, `"zipup"`, `"sdc"`, `"sdcr"`, and
  `"dm"` use Quimb boundary compression directly, without FIT. Their
  supported `*-first` and `*-oversample` variants are accepted as well;
  `"src-mps"` is the readable alias for Quimb's canonical `"srcmps"`.
- `"eff"` is the compatibility default and performs cached one-site sweeps.
- `"two-site"` forms neighboring boundary wavefunctions, splits them with a
  backend-native SVD, and can discover bond subspaces up to `fit_max_bond`.
- `"dmrg2"` performs two-site FIT warm-up sweeps followed by one-site
  refinement; it is distinct from the fixed two-site `"two-site"` mode.
- `"global"` uses the reference global FIT solve.
- `"dmrg"` is an alias for `"eff"`.

Direct Quimb modes support `fit_layer_mode="joint"` (default) or
`"sequential"` on the multilayer, non-flat boundary path. Sequential mode
applies the selected direct compressor to each tagged layer in `layer_tags`
order, which can be useful for a BRA--PEPO--KET boundary. It is intentionally
unavailable for variational FIT modes, whose target remains the complete local
layered network. `contract_flat(...)` is separate: it uses `flat=True` for one
effective layer and requires the joint policy.
The default `fit_layer_order="input"` preserves the supplied semantic order;
`fit_layer_order="auto"` is opt-in, requires explicit tags whose layers are
known to be mathematically interchangeable, and orders dense layers by an
estimated intermediate tensor size.

Each boundary FIT can optionally use `fit_init_strategy="guess-direct"`,
`"guess-src"`, or `"guess-sdc"`. These compress a copy of the exact
boundary target and give that disposable result to FIT as its initial guess.
The exact target and live/reusable boundary remain unchanged; the default
`"direct"` strategy preserves the historical behavior. Native Symmray
boundaries fall back to `"direct"` with a diagnostic warning because dense
Quimb guesses cannot preserve their charge sectors.

The `"eff"` path accepts `fit_block_size=2` or `3` for native block-SVD
growth. With `fit_adaptive_sweeps=N`, it uses that block size for the first
`N` full-boundary sweeps and then switches to cached one-site refinement.
Optional `fit_rtol`, `fit_min_iter`, and `fit_patience` controls are forwarded
to `FIT.run_eff`; with `fit_rtol=None`, no convergence scalar is transferred
from the backend. When enabled, stopping begins only after two completed
sweeps, using the relative retained-norm change between consecutive sweeps.
The default `RL` sequence runs each local boundary FIT left-to-right and then
right-to-left. Use `fit_sweep_sequence="LR"` when the requested local
compression should begin left-to-right.

Two-site sweeps do not rebuild a complete environment for every pair. One
side is cached once per sweep and the moving side is updated incrementally, so
environment construction remains linear in boundary length. PEPS helpers pass
the requested `chi` as the default two-site bond cap; direct `CompBdy` users
should set `fit_max_bond` when they want a lower-rank boundary to grow. New
two-site boundaries start at bond 1 and grow through these local splits.
With `fit_cutoff="auto"`, two-/three-site FIT starts each bond at its current
rank, grows only when the reported discarded weight exceeds the resolved
cutoff, and reuses learned per-bond caps on later sweeps up to `fit_max_bond`.
Reusing a boundary at a larger `chi` does not globally pad it first; lowering
`chi` still compresses existing bonds immediately.

## Public helpers

- `contract_boundary(...)`: contracts a prebuilt double-layer network with a
  supplied `BdyMPS` or `{"bdy": BdyMPS}` holder and returns
  `BoundaryContractResult`.
- `peps_normalize(...)`: normalize a PEPS in place. The generic
  `normalize(...)` spelling remains as a deprecated compatibility alias.
- `boundary_norm(...)` / `peps_norm(...)`: compute `<p|p>` without rescaling.
  Pass `return_info=True` for the structured result instead of only the scalar.
- `peps_infidelity(...)`: compute boundary-based infidelity, optionally
  reusing norm and overlap boundaries. The generic `infidelity(...)` spelling
  remains as a deprecated compatibility alias.
- `peps_fidelity(...)`: return only fidelity by default, or preserve all three
  contraction results and their FIT diagnostics with `return_info=True`.
- `contract_flat(...)`: contract one already-flattened effective PEPS-like
  layer with the `flat=True` first-slice shortcut. It is not a multilayer
  PEPS--PEPO--PEPS façade.
- `contract_layered(...)`: contract a preassembled multilayer network with
  explicit `layer_tags`, using the shared `CompBdy` engine and `flat=False`.
  This is the high-level façade for sequential BRA--PEPO--KET-style
  absorption without flattening the stack into one giant tensor network.

Use `result.cost`, `result.fidel`, and `result.fit_diagnostics` from
`BoundaryContractResult`; do not rely on tuple unpacking. Each typed boundary
FIT diagnostic reports actual iterations and convergence, with detailed sweep
timing populated only when `fit_timing=True`.

## Boundary methods

The default `method="dmrg"` uses Pepsy's `BdyMPS` plus `CompBdy` path. Other
methods route to Quimb-style contraction methods when the network exposes them:

- `method="mps"` uses `TensorNetwork.contract_boundary(...)`.
- `method="ctmrg"` uses `TensorNetwork.contract_ctmrg(...)`.
- `method="hotrg"` uses `TensorNetwork.contract_hotrg(...)`.
- `method="exact"` directly contracts the double-layer network.

When `strip_exponent=True`, helpers preserve Quimb's `(mantissa, exponent)`
representation and Pepsy applies exponent shifts explicitly.

## Editing notes

- Keep optional dependencies optional; tests for torch, JAX, CuPy, NLopt,
  SciPy, or Symmray should use `pytest.importorskip(...)` when needed.
- Preserve holder-dict behavior such as `{"bdy": ...}` because optimizers use
  it to reuse and update boundary state.
- Preserve lattice tag conventions: `X{i}`, `Y{j}`, `I...`, and physical outer
  indices conventionally named `k...` and `b...`.
- Focused validation usually starts with:

```sh
pytest -q tests/test_prepare_boundary_inputs.py
```
