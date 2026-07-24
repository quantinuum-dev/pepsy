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

Streams may also include control events. `("measure", pauli, where[, outcome])`
collapses onto a Pauli eigenvalue and records `(pauli, where, outcome, prob)`.
`("reset", where[, basis])` resets each target to the `+1` eigenstate of
`basis` (`"Z"` by default, so the legacy form resets to `|0>`); the internal
measurement is not recorded. `("measure_reset", basis, where[, outcome])`
measures each target in `basis`, records the result, then resets it to the
`+1` eigenstate. The aliases `("mrx", where[, outcome])`, `("mry", ...)`, and
`("mrz", ...)` are accepted. `("cap", where, vec[, absorb])` contracts one
physical leg with `vec`, absorbs it into the selected neighbour, and shortens
the MPS by one site.

`mode="mix"` is a unitary gate-stream mode that warms up with MPO replay while
`p.max_bond() < chi`, then switches to DMRG replay once the working MPS reaches
the target bond. If a DMRG step raises or produces non-finite tensor data, the
optimizer restores the pre-step state and replays that step with MPO instead.
The mixed replay history is stored in `opt.mix_history` and summarized in
`opt.last_mix_summary`. With `progbar=True`, the progress bar shows the current
backend, cumulative MPO/DMRG/fallback counts, and `bond=current/chi`.

`mode="su"` uses simple-update evolution for imaginary-time or other
non-unitary gate streams. It keeps `opt.p` as the simple-update core and
stores the external bond factors in `opt.gauges`. After every run,
`opt.p_ungauged` is refreshed as a physical copy with those gauges inserted.
If the supplied dictionary
does not contain the current bond gauges, the optimizer initializes it with
`opt.p.gauge_all_simple_(gauges=opt.gauges, progbar=False)`, then applies each
gate through `pepsy.gate_simple(..., renorm=True)`. This mode does not
canonicalize the MPS and does not report compression infidelity. Use
`opt.p_ungauged` for the physical state and `opt.p` for continued SU updates.
If an independent physical copy is needed, use:

```python
physical = opt.p_ungauged.copy()
```

For Symmray block-sparse MPS data, `gate_simple` automatically uses Quimb's
full two-site `split` path so symmetry and fermionic fusion metadata are
preserved. Dense MPS data keeps the faster `reduce-split` path by default.

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

Normalization and infidelity use the same canonical-center contract. For a
non-unitary run with `normalize_every` enabled, after every replay step the
optimizer moves the tracked range to a one-site center, normalizes that center
tensor, and stores the removed scale in `p.exponent`. Thus `p.norm()` restores
the represented norm, while a copy with `exponent=0` exposes the normalized
working data. For DMRG, a multi-gate batch is one replay step for this purpose.
For unitary streams,
`get_infidelities()` uses the current retained norm divided by the initial
run norm, squared; this is the global retained-fidelity estimate and is
equivalent to multiplying the per-gate fidelities without requiring a
pre-gate target measurement. Local ratios remain available in detailed
samples for diagnostics. The trace is populated by default. Set
``track_infidelity=False`` in the constructor, or pass
``track_infidelity=False`` to ``run()``, to skip target-norm construction,
retained-norm calculations, samples, and progress-bar infidelity fields. For
dense two-site non-unitary gates, the target
norm is obtained from the local expectation of `G†G`, so no copied target MPS
is needed. Symmray and general sub-MPO backends use a raw target-norm fallback
where that local expectation is not available. DMRG still materializes its
target because FIT needs it, but the diagnostic uses FIT's final local norm
trace as the current retained norm.
Temporary fallback targets never modify the live `info_c` cache.
When tracking is enabled, the `mpo`, `swap`, and `svd` progress bars show the
same cumulative `infidelity` field, starting at zero before the first
compressed two-site gate.
`mode="exact"` and `mode="su"` deliberately skip canonical metadata; switching back to an MPS
mode rebuilds and canonicalizes the contracted state.

The result API is intentionally small:

```python
opt.run(non_unitary=True, normalize_every=True, track_infidelity=False)

opt.get_infidelities()        # [0.0] when tracking is disabled
opt.get_infidelity_samples()  # [] when tracking is disabled
opt.get_normalizations()      # scale events and accumulated exponents
```

When enabled, infidelity is recorded automatically whenever a compressed
two-site update occurs. `get_infidelities()` is the cheap cumulative trace for
progress and stopping criteria. Use
`get_infidelity_samples()` when the target norm, retained norm, local ratio, or
step metadata is needed.

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
