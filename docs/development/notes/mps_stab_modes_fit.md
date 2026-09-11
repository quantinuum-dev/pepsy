# Stabilizer-MPS mode and FIT parity audit

## 2026-09-10

This audit covers the `MpsStabOptimizer` native compression and local FIT
paths. The ordinary `MpsOptimizer` remains the behavioral reference.

Installed versions in the active `py312` environment:

- Quimb `1.15.1.dev51+g2e99c793e`
- Autoray `0.11.1.dev3+g1b476b305`
- Cotengra `0.8.3.dev7+g1d7fd333f`
- Symmray `0.3.2.dev8+g6c6dd34b5`

API probes:

- `FIT.run_gate` accepts `cutoff_mode`, `min_iter`, `rtol`, `patience`,
  `finite_check`, adaptive block controls, and the three-site transition
  control used by the STN wrapper.
- `FIT(...)` accepts `contraction_opt`.
- Quimb `MatrixProductState.gate_with_submpo_` accepts compression options
  through `**compress_opts`; `MatrixProductState.compress` does the same.
- The installed Quimb dispatch provides `direct`, `src`, `sdc`,
  `sdc-oversample`, and `fit-projector`. The capability check is now shared
  with `MpsOptimizer`, so an unavailable method fails explicitly at replay.

Upstream sources checked:

- Quimb changelog: <https://quimb.readthedocs.io/en/latest/changelog.html>
- Autoray repository/docs: <https://github.com/jcmgray/autoray>
- Cotengra docs/changelog: <https://cotengra.readthedocs.io/en/latest/>
- Symmray Abelian-array docs: <https://symmray.readthedocs.io/en/latest/abelian_arrays.html>
- Symmray repository: <https://github.com/jcmgray/symmray>

Disposition:

- **Adopt:** ordinary MPS `cutoff="auto"`, `cutoff_mode="auto"`, explicit
  contraction options, dtype-aware `fit_rtol`, FIT iteration/patience/block
  controls, `guess-src` initialization, DMRG1/2/3 schedules, and native SDC
  capability validation.
- **Compatibility shim:** STN's constructor and `run()` retain the existing
  `fit_init_strategy` spelling and legacy mode aliases while accepting the
  ordinary MPS run controls. `fit_rtol` is the canonical FIT tolerance name;
  no separate `fit_ftol` option is introduced because FIT does not expose one.
- **Defer:** Quimb's newer `sdcr` and `sdcr-oversample` methods remain outside
  Pepsy's ordinary `_MPO_COMPRESSION_METHODS` set, so the STN API does not
  advertise modes that `MpsOptimizer` cannot select.

Focused validation: `tests/test_stabilizer_tn.py` (336 passed), the selected
MPS optimizer DMRG/compression suite (138 passed), and Ruff on the modified
source and stabilizer tests.
