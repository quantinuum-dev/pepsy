# VMC module map

The Torch VMC package keeps the public surface in
`pepsy.vmc.torch.__init__`. The implementation is organized by responsibility:

- `_common.py`: shared configuration normalization, shape validation, device
  resolution, edge helpers, and the opt-in cheap-kernel compiler fallback
- `_graded.py`: physically-owned optional Symmray graded-contraction helpers
- `amplitude.py`: physically-owned PEPS amplitude models, boundary reuse, and
  connected-configuration batching
- `sr.py`: physically-owned log derivatives and stochastic reconfiguration
- `metadata.py`: physically-owned PEPS/Fermion layout, basis, and sector
  inference
- `connections.py`: physically-owned operator-to-connection compilation
- `proposals.py`: symmetry-preserving proposal kernels, Metropolis sweeps, and
  move diagnostics
- `sampler.py`: stateful Metropolis and BP sampler state
- `local_energy.py`: physically-owned Hamiltonian connection builders,
  local-energy estimators, weighted statistics, and adaptive diagnostics
- `results.py`: physically-owned result records and progress helpers
- `driver.py`: native Torch VMC measurement and optimization driver
- `fermion.py`: fermion-specific metadata, initialization, and setup façade
- `importance.py`: sampler-agnostic proposal normalization and importance
  measurement

The package retains `_core.py` as a compatibility and cross-workflow dispatch
module while the responsibility paths are stabilized. Result records,
metadata inference, operator compilation, local-energy/statistics helpers,
PEPS amplitude models, graded/Symmray contraction helpers, stochastic
reconfiguration, proposal/sampler state, the native driver, and fermion setup
now have physically-owned modules. `_common.py` owns the shared leaf helpers;
`_core.py` keeps compatibility imports and cross-workflow dispatch. New code
should use the public symbols from `pepsy.vmc` or `pepsy.vmc.torch`, and the
proposal entry point `TorchVMCDriver.measure_from_proposal(...)`.
