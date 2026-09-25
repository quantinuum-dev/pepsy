# Independent compression stages

Dense MPS and MPO replay accepts `run(compression_opts=...)`. PEPS boundary
workers and metrics accept the same mapping as `fit_compression_opts`, with
`method="dmrg"` and an explicit Quimb `fit_mode`. `PepsOptimizer` forwards
this policy through `boundary_kwargs`; `SweepOptimizer` accepts it directly
with `boundary_engine="dmrg"` and copies the mapping into its boundary workers.

```python
compression = {
    "max_bond_oversample": 128,
    "cutoff_oversample": 0.0,
    "cutoff_mode_oversample": "rel",
    "compress_opts_final": {
        "method": "svd",
        "cutoff": 1e-10,
        "cutoff_mode": "rsum2",
    },
}

optimizer = pepsy.MpsOptimizer(state, gates, chi=64, mode="sdc-oversample")
result = optimizer.run(compression_opts=compression)

norm = pepsy.peps_norm(
    peps, chi=64, fit_mode="sdc-oversample",
    fit_compression_opts=compression,
)
```

`max_bond_oversample` is a positive integer specifying the intermediate cap.
The other intermediate options select its cutoff and interpretation. Final
compression uses the ordinary `cutoff` and `cutoff_mode` unless explicitly
overridden in `compress_opts_final`. That dictionary accepts `method`,
`cutoff`, and `cutoff_mode`. It cannot override the final bond limit: `chi`
controls MPS/MPO output and `fit_max_bond` controls boundary compression
(falling back to the existing boundary cap when omitted).
Final decomposition may be `"svd"`, `"svd:eig"`, or explicitly requested
`"rsvd"`; the latter requires an explicit `"abs"` or `"rel"` final cutoff
mode. QR is not a final truncating decomposition and is rejected.

Each supplied option must be explicitly supported by the installed Quimb
compressor. Unsupported options raise before replay or boundary retuning;
an older compressor's `**kwargs` does not count as support. The caller's
mapping is copied. Omitting the mapping preserves existing defaults.

`sdc-oversample`, `sdcr-oversample`, and `zipup-oversample` in recent Quimb
builds support all four options. Availability for other methods depends on
their signatures. Randomized compression remains opt-in. SDCR intermediate
cutoffs must be `"abs"` or `"rel"`; its final deterministic sweep can use
`"rsum2"`. Base `sdcr` retains its relative-cutoff compatibility policy and
now warns when a cumulative cutoff is changed to `"rel"`.

These options apply to compression replay, including dense gates and MPS
sub-MPO events. They do not configure disposable DMRG/FIT guesses or change
exact fitting targets. Native Symmray MPS/MPO replay, MPO channel events, and
non-Quimb replay modes reject explicit advanced settings. Native PEPS direct
compression retains its existing restriction; use its established DMRG path.
No speedup is implied by choosing a larger intermediate cap.
