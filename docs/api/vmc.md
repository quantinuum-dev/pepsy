# `pepsy.vmc`

Optional VMC integrations live under `pepsy.vmc`. The first implementation is a
NetKet bridge for Symmray-backed fermionic PEPS amplitudes:

```python
import pepsy.vmc as pvmc

pvmc.configure_jax_for_vmc()
ansatz = pvmc.pack_fermionic_peps_ansatz(peps, lattice_shape=(2, 2))
settings = pvmc.recommend_netket_vmc_settings(
    n_params=ansatz.n_params,
    n_samples=4096,
    n_chains=32,
)
setup = pvmc.build_fermi_hubbard_vmc(
    peps,
    Lx=2,
    Ly=2,
    t=1.0,
    U=8.0,
    contraction="exact",
    n_samples=settings.n_samples,
    n_chains=settings.n_chains,
    chunk_size=settings.chunks.chunk_size,
    sampler_chunk_size=settings.chunks.sampler_chunk_size,
    seed=1,
    sampler_seed=2,
)
driver = pvmc.make_netket_vmc_driver(
    setup,
    driver=settings.driver,
    learning_rate=0.02,
    sr_mode=settings.sr_mode,
    use_ntk=settings.use_ntk,
    on_the_fly=settings.on_the_fly,
    chunk_size_bwd=settings.chunks.chunk_size_bwd,
)
```

For notebook profiling or custom NetKet models, use the same batched JAX path
directly. This mirrors Symmray's batch GPU amplitude example: pack the PEPS once,
keep `flat=True` Symmray data for JIT-friendly leaves, and evaluate many
spin-orbital occupation rows with one compiled function.

```python
batched_amp = pvmc.make_fermionic_peps_batched_amplitude_function(
    ansatz,
    setup.columns,
    contraction="hotrg",
    chi=4,
    output="mantissa_exponent",
)
mantissa, exponent = batched_amp(occupation_rows)
```

The approximate contraction choices are Quimb's finite 2D contractions:
`contraction="hotrg"` calls `contract_hotrg`, `contraction="ctmrg"` calls
`contract_ctmrg`, and `contraction="boundary"` or `"mps"` calls
`contract_boundary(mode="mps")`. Pass `contraction_opts={...}` for lower-level
Quimb options such as `sequence`, `max_separation`, `canonize`, or a non-default
boundary `mode`.

For larger lattices use an approximate contraction with `chi=...`, and treat
`chi`, `chunk_size`, `chunk_size_bwd`, `n_samples`, and the SR setting as
performance/accuracy knobs. The dense sign checks from the notebook should
remain capped to tiny systems. For large parameter counts prefer
`driver="vmc_sr"` with `sr_mode="real"` so NetKet can use its SR/minSR driver at
lower fermionic Jacobian cost, and use `make_netket_autochunk_callback(...)` on
the first GPU run to tune sampler, forward, and backward chunks after
compilation pressure is known.

```python
callback = pvmc.make_netket_autochunk_callback(
    sampler_chunk_size=settings.chunks.sampler_chunk_size,
    chunk_size=settings.chunks.chunk_size,
    chunk_size_bwd=settings.chunks.chunk_size_bwd,
)
driver.run(n_iter=100, out="vmc_run", callback=callback)
```

```{eval-rst}
.. automodule:: pepsy.vmc.netket
   :members:
   :undoc-members:
```
