# `pepsy.optimizers.mps.optimizer`

`MpsOptimizer` consumes canonical bundled gate streams of the form
`[(gate, where), ...]`. In `mode="mpo"` the stream can also contain explicit
sub-MPO events for already-factorized nonlocal operators:

```python
event = ("submpo", mpo, where)
# or
event = {"kind": "submpo", "mpo": mpo, "where": where}
```

`where` is a non-empty tuple/list of unique 1D MPS sites. The convenience
helper `MpsOptimizer.submpo_event(mpo, where)` builds the tuple form. These
events are applied with `gate_with_submpo_` and compressed to `chi`; they are
only accepted in `mode="mpo"`.

`mode="mix"` is a unitary gate-stream mode that warms up with MPO replay while
`p.max_bond() < chi`, then switches to DMRG replay once the working MPS reaches
the target bond. If a DMRG step raises or produces non-finite tensor data, the
optimizer restores the pre-step state and replays that step with MPO instead.
The mixed replay history is stored in `opt.mix_history` and summarized in
`opt.last_mix_summary`. With `progbar=True`, the progress bar shows the current
backend, cumulative MPO/DMRG/fallback counts, and `bond=current/chi`.

`mode="swap"` applies non-local two-site gates through a swap-and-split path
and swaps the sites back after each gate. `mode="perm"` uses the same
swap-and-split path but leaves the swaps in place, tracking the current
physical-site-to-logical-site ordering in `opt.qubits`. This is useful for
streams with little expected locality. The returned `opt.p` remains an MPS in
physical order; call `opt.restore_qubit_order()` when a conventional logical
site order is needed.

For repeated evolution, use `opt.apply_layout("quality")` once. This installs
the selected position-to-logical mapping in `opt.logical_order` and keeps the
MPS in that order across subsequent `run()` calls. A bond-one initial MPS is
relabelled without SVD swaps; an initially entangled MPS raises by default, or
can pay one explicit lossy reorder with `allow_lossy_reorder=True` and a caller
provided `cutoff`. The old `run(use_layout_finder=True)` path is retained only
for compatibility and performs the deprecated temporary reorder and swap-back.
Use `opt.logical_site(position)`, `opt.position(site)`,
`opt.remap_sample(config)`, and `opt.to_dense()` for logical readout.

Pauli control events use Quimb's `local_expectation_canonical` when available.
The optimizer passes its `info_c` dictionary, so canonicalization starts from
the tracked `info_c["cur_orthog"]` range, moves only as needed around the
observable support, and records the new range. A concrete tracked range avoids
an orthogonality-center scan. Older Quimb versions without the local evaluator
use a compatibility overlap contraction instead.

Normalization and norm diagnostics follow the same canonical-center contract.
`track_norm_infidelity=True` uses a one-site center norm after moving the
tracked range to the gate support; it does not build a global doubled-network
contraction. For default unitary streams, the target norm is the pre-gate
canonical norm, so no uncompressed target copy is built unless
`track_infidelity=True` also needs it. Non-unitary norm diagnostics still build
their pre-compression targets because those updates can change the norm.
Non-unitary scale control records the removed factor in `p.exponent`, while
temporary diagnostic targets never modify the live `info_c` cache.
`mode="exact"` deliberately skips canonical metadata; switching back to an MPS
mode rebuilds and canonicalizes the contracted state.

For a logical gate stream whose site order has not been chosen yet,
`MpsOptimizer.LayoutFinder(gates, L=...)` or
`MpsOptimizer.gate_stream_layout(gates, L=...)` returns a 1D layout plan with
the optimized site order, old-to-new site map, internal mapped locations, and
span statistics. The finder implementation lives in
`pepsy.optimizers.mps.layout.MpsGateStreamLayoutFinder`, while
`MpsOptimizer.LayoutFinder` keeps the attached optimizer-facing API. The finder
builds a weighted interaction graph from gate and
sub-MPO supports, scores layouts with a Tensy-like scalar objective
`weighted_total_span + weighted_cut_congestion_l2 + tail_span_penalty`, and
uses degree/BFS/spectral/recursive candidates plus adjacent-swap refinement. If
available, the refinement uses numba; `order="quality"` also tries optional
nevergrad candidates, and optional KaHyPar recursive bisection when a config is
provided with `kahypar_config_path=...` or `PEPSY_KAHYPAR_CONFIG`. Event weights
default to `weight_mode="auto"`: angle metadata when present, otherwise a cheap
operator-Schmidt proxy for small dense two-site gates, falling back to count
weights. Pass `weight_fn(payload, support, event_type)` for explicit weights.

The layout score depends on gate supports and optional gate/event weights, not
on the initial MPS tensor values. The plan does not rewrite the gate stream. To
use a layout during replay, call `opt.run(use_layout_finder=True)` or pass a
layout order such as `opt.run(use_layout_finder="quality")`; the optimizer
temporarily permutes the working MPS and restores the returned MPS to the
original site order. Layout-aware replay prints a concise report by default;
pass `layout_report=False` to silence it.

```{eval-rst}
.. automodule:: pepsy.optimizers.mps.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
```
