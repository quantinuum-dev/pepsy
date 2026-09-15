# `pepsy.boundary.metrics`

`peps_norm`, `peps_normalize`, `peps_infidelity`, and `contract_flat` accept
the following boundary-compression modes:

- `fit_mode="direct"`, `"src"`, `"src-mps"`, `"zipup"`, `"sdc"`,
  `"sdcr"`, or `"dm"`: direct Quimb boundary compression. Quimb's
  `*-first` and `*-oversample` variants are also accepted, including
  `"srcmps"`/`"src-mps"`, `"srcmps-oversample"`/`"src-mps-oversample"`,
  `"src-oversample"`, `"sdc-oversample"`, and `"zipup-oversample"`.
  Because Quimb's randomized `sdcr` split does not support cumulative cutoff
  modes, the shared `fit_cutoff_mode="auto"` policy uses `"rel"` for `sdcr`.
- `fit_mode="eff"` (also `"dmrg"` and `"dmrg1"`): cached one-site FIT
  sweeps and the compatibility default.
- `fit_mode="two-site"`: fixed two-site FIT updates.
- `fit_mode="dmrg2"`: two-site FIT warm-up followed by one-site refinement,
  in the same spirit as a two-site-to-one-site MPS optimizer.
- `fit_mode="global"`: the full-contraction reference fit.

Direct Quimb modes on the normal `peps_norm` / `boundary_norm` path also
accept `fit_layer_mode="joint"` (the default) or
`fit_layer_mode="sequential"`. Joint mode compresses all tagged layers in a
boundary slice together. Sequential mode compresses them one at a time in the
order given by `layer_tags`; the standard double layer is automatically tagged
`KET`/`BRA`, while a custom multilayer target can use tags such as
`layer_tags=("BRA", "PEPO", "KET")`. Sequential mode is available only for
the direct Quimb modes; FIT/DMRG modes always use a joint target.

`contract_flat` has a narrower meaning: it contracts one already-flattened
effective layer and uses `flat=True` boundary initialization. It requires
`fit_layer_mode="joint"`; it is not the entry point for a stack of separate
PEPS/PEPO layers. For that case, provide the tagged network and
`BdyMPS(tn_double=network, flat=False)` to `contract_boundary`.

### Flat directional and middle-out compression

For Quimb boundary contraction, `contract_flat` accepts readable
`boundary_direction` presets in addition to the lower-level `sequence`
argument:

| `boundary_direction` | Quimb sequence | Behavior |
| --- | --- | --- |
| `"bottom-up"` | `("xmin",)` | absorb from the bottom row |
| `"top-down"` | `("xmax",)` | absorb from the top row |
| `"top-bottom"` / `"bottom-top"` | both x boundaries | alternate inward in the named order |
| `"left-to-right"` | `("ymin",)` | absorb from the left column |
| `"right-to-left"` | `("ymax",)` | absorb from the right column |
| `"left-right"` / `"right-left"` | both y boundaries | alternate inward in the named order |
| `"four-sided"` | all four boundaries | cycle top, bottom, left, and right |

These presets work with `method="mps"` and `method="ctmrg"`. They are
outside-in schedules implemented by Quimb. `compression_mode` selects the 1D
compressor for `method="mps"`; for example, this requests direct SVD boundary
compression from both time boundaries:

```python
value = pepsy.contract_flat(
    flat,
    method="mps",
    chi=64,
    compression_mode="direct",
    boundary_direction="top-bottom",
)
```

`boundary_direction="middle-out-x"` and `"middle-out-y"` are different:
they absorb the two opposing outer boundaries inward towards a selected middle
row or column. Thus `"middle-out-y"` builds compressed left and right boundary
environments around a central column, while `"middle-out-x"` does the same
from bottom and top around a central row. Pepsy uses Quimb's `around` target to
protect that middle slab and then exactly contracts the small remaining core.
This mode works with direct MPS compression and finite CTMRG; it does not start
at the center and grow outwards. The selected axis must have two open
boundaries. For a trace network that is cyclic in x/time, use
`"middle-out-y"` unless the cyclic x bond is cut explicitly.

By default, the protected target is one central slice for an odd axis or the
central pair for an even axis. `middle_slices` can instead select a contiguous
interface explicitly. For a flat time network whose fixed target rows end at
`u_last` and variational rows begin at `v_first`:

```python
value = pepsy.contract_flat(
    flat,
    method="mps",
    chi=64,
    compression_mode="direct",
    boundary_direction="middle-out-x",
    middle_slices=(u_last, v_first),
    preserve_backend=True,
)
```

The input network is copied and Pepsy does not convert the backend arrays.
Torch reverse-mode differentiation is covered explicitly by the regression
suite. Supplying both `boundary_direction` and `sequence` is an error. The
existing defaults are unchanged when neither new option is used.

`contract_layered` is the explicit façade for a preassembled multilayer
network. It requires `layer_tags`, for example
`contract_layered(network, layer_tags=("BRA", "PEPO", "KET"), chi=64)`, and
uses the same `CompBdy` engine with `flat=False`. Set
`fit_layer_mode="sequential"` to absorb those layers one at a time in the
given order, or keep the default `"joint"` policy. The façade also accepts
`fit_layer_order="auto"` for explicitly tagged layers that are mathematically
interchangeable. It does not flatten the stack into one giant tensor network.
Use `contract_boundary` when you need lower-level control over an
already-created `BdyMPS` or Quimb's native `method="mps"` path.

The layer policy applies to the package `method="dmrg"` path. Quimb's native
`method="mps"` path already handles `layer_tags` itself and rejects
`fit_layer_mode="sequential"` rather than silently ignoring the option.

## CTMRG boundary modes

With `method="ctmrg"`, `ctmrg_mode` selects the finite-boundary compressor:

- `ctmrg_mode="projector"` is the compatibility default. It computes local
  oblique projectors and preserves the previous Pepsy behavior.
- `ctmrg_mode="projector2d"` uses Quimb's explicit 2D plaquette-projector
  contraction.
- `ctmrg_mode="l2bp"` compresses each boundary with Quimb's lazy 2-norm
  belief-propagation implementation.

`ctmrg_canonize=None` preserves the existing `True` projector preconditioning
and enables the normal local gauging for `l2bp`. For `projector`, it can also
be set to `False`, `"layered"`, or `"bp"`. The `"layered"` choice gauges the
tensor layers separately; `"bp"` reruns dense D2BP while constructing each
compressed boundary. Configure those solves with `ctmrg_canonize_opts`, for
example `{"max_iterations": 10, "tol": 1e-8, "damping": 0.2}`.

`ctmrg_projector_region` selects the local projector window. `None` and
`(2, 2)` use Quimb's native two-neighbor projector path. The experimental
`(2, 3)` choice expands each central cut to three neighboring effective
boundary sites, alternating the extra site between its two sides. On a
standard norm or overlap, Quimb still absorbs the tagged `KET` and `BRA`
layers sequentially, so this is a current-boundary/incoming-layer window over
three sites rather than a periodic 2x3 unit cell. Combining `(2, 3)` with
`ctmrg_canonize="bp"` makes every such projector BP-dressed; D2BP is rerun for
each compressed boundary/layer. This first pass is dense, 2D, and
open-boundary only. It does not persist BP messages between CTMRG steps and is
not a generalized/Kikuchi BP implementation.

`ctmrg_compress_opts` is copied and forwarded to the selected compressor. For
`l2bp`, this is where iteration, convergence, damping, and update controls
belong. `ctmrg_reduce_opts` configures squared-environment factorization for
the two projector modes, while `ctmrg_gauge_smudge` applies only to
`projector`. Options that a selected mode cannot consume raise an error rather
than being silently ignored.

`projector2d`, `l2bp`, `ctmrg_canonize="bp"`, the `(2, 3)` projector region,
and projector contraction with gauging disabled are currently dense-only.
Native Symmray contraction supports the validated `projector` route with
`ctmrg_canonize=True` or `"layered"` and the native region, retaining its
established factorization safeguards. Pepsy capability-checks each selected
mode against the installed Quimb build at execution time.

```python
norm_bp_projectors = pepsy.peps_norm(
    state,
    chi=64,
    method="ctmrg",
    ctmrg_mode="projector",
    ctmrg_canonize="bp",
    ctmrg_projector_region=(2, 3),
    ctmrg_canonize_opts={"max_iterations": 8, "damping": 0.2},
)

norm_l2bp = pepsy.peps_norm(
    state,
    chi=64,
    method="ctmrg",
    ctmrg_mode="l2bp",
    ctmrg_compress_opts={"max_iterations": 20, "tol": 1e-8},
)
```

Selectors are normalized early: `"two_site"` is accepted as an alias for
`"two-site"`, `"one-site"`, `"dmrg"`, and `"dmrg1"` alias the historical
`"eff"` spelling. Unknown values fail before boundary work starts.

`fit_init_strategy="direct"` (or the compatibility alias `"auto"`) preserves
the existing boundary guess. `"guess-direct"`, `"guess-src"`, and
`"guess-sdc"` create disposable Quimb-compressed guesses from each exact
boundary target before FIT; they do not replace the target or mutate reusable
boundaries. These strategies are initialization choices only: the selected
`fit_mode` still controls the final compressor. Native Symmray boundaries
fall back to `direct` with a warning for dense Quimb guesses.
The direct Quimb `fit_mode` values are currently dense-only; use
`fit_mode="dmrg"` or `"dmrg2"` for native Symmray boundaries.
They replace each visited boundary directly and do not construct, copy, or
globally expand an unused FIT initial guess.

For `fit_mode="eff"`, set `fit_block_size=2` or `3` to use native block-SVD
growth through `FIT.run_eff`. Add `fit_adaptive_sweeps=2` to perform two
block sweeps followed by one-site refinement; omit it to keep fixed block
sweeps. `fit_rtol`, `fit_min_iter`, and `fit_patience` remain optional
convergence controls. When `fit_rtol` is enabled, stopping begins only after
two completed sweeps. The default `fit_block_size=1` path also honors the
alternating `RL` sequence.

Example:

```python
result = pepsy.peps_norm(
    state,
    chi=64,
    method="dmrg",
    fit_mode="dmrg2",
    fit_layer_mode="joint",
    fit_init_strategy="guess-src",
    fit_init_seed=7,
    fit_sweep_sequence="RL",
    cutoff="auto",
    fit_cutoff_mode="auto",
    n_iter=8,
    fit_min_iter=2,
    fit_rtol=1e-8,
    fit_patience=2,
    fit_timing=True,
    return_info=True,
)

print(result.cost)
for fit in result.fit_diagnostics:
    print(fit.boundary_key, fit.iterations, fit.convergence_reason)
```

`chi` is the default two-site SVD cap. Supply `fit_max_bond` to use a
different cap. With `fit_rtol=None`, exactly `n_iter` sweeps run. Reuse a
`BdyMPS` or `{"bdy": BdyMPS}` holder to retain fitted boundary states across
calls; pair updates retain the same fixed-plus-moving environment cache
strategy within each sweep. Newly created two-site boundaries start at bond 1
and grow locally rather than being padded globally to `chi`.

For example, an adaptive full-chain FIT boundary can be requested with:

```python
result = pepsy.peps_norm(
    state,
    chi=64,
    fit_mode="eff",
    fit_block_size=2,
    fit_adaptive_sweeps=2,
    fit_sweep_sequence="RL",
    n_iter=6,
    return_info=True,
)
```

`contract_boundary(...)` always returns `BoundaryContractResult`. The scalar
helpers `peps_norm(...)`, `boundary_norm(...)`, and `contract_flat(...)` keep
returning a scalar by default; pass `return_info=True` to receive the same
structured result. Its `fit_diagnostics` tuple has one `BoundaryFitDiagnostic`
per attempted boundary fit, with the boundary key, actual iteration count,
convergence reason, relative change, final center/direction, and reached bond
dimension. These fields are collected without per-site timing overhead.

For differentiable flat-network objectives, pass
`contract_flat(..., preserve_backend=True)`. This bypasses only Pepsy's final
Python-scalar formatting and returns the raw NumPy, Torch, or JAX scalar. With
`strip_exponent=True`, the raw `(mantissa, exponent)` pair is retained. The
contraction algorithm and its truncations are otherwise unchanged.

Set `fit_timing=True` to additionally populate each diagnostic's
`elapsed_seconds` and detailed two-site `sweep_timings`. On asynchronous
accelerators, `fit_timing_sync_device=True` adds device barriers so those
timings include completed kernels.

`peps_infidelity(...)` always returns its norm and overlap contraction results
under `norm_result`, `norm_target_result`, and `overlap_result` (a supplied
known norm has a corresponding `None`). `peps_fidelity(...)` remains scalar by
default; use `return_info=True` to receive that same dictionary plus the
computed `fidelity`. This makes `fit_timing=True` useful on both helpers.

> API details are maintained as handwritten Markdown in this page.
