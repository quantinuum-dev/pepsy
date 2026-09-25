# Shared numerical contracts

Read the relevant section before changing Torch linear algebra, CTMRG, or a
downstream numerical comparison. These are maintained contracts, not benchmark
recommendations. Domain-specific MPS, tree, and fermion rules remain in their
own skills. Follow the environment selected by repository and local instructions.

## Torch SVD/QR policy

Keep Torch linear-algebra registration behind the single public
`pepsy.TorchLinalgConfig` class. Its `register()` operation configures the
Autoray SVD and QR rules together and can additionally configure Quimb's raw
Symmray split drivers. Do not add a new optimizer call that independently
combines `reg_*_torch` helpers or `register_torch_linalg(...)` arguments.

- `stabilized=False` is the native default for ordinary simulation.
  `stabilized=True` is the autodiff policy: it installs Pepsy's
  relative-regularized SVD VJP and the configured rank-aware QR behavior for
  difficult or rank-deficient tensor-network splits.
- `svd_driver="gesvdj"` and `"gesvd"`, plus CPU Torch/Scipy `gesdd`/`gesvd`,
  are non-approximate choices. `gesvda` is approximate and must remain behind
  `allow_approximate=True`.
- `quimb_split_drivers=True` is required when raw Torch Symmray blocks enter
  Quimb's `svd_truncated` or `qr_stabilized` paths, because those blocks bypass
  ordinary Autoray dispatch. PEPS optimizers should enable it automatically
  when Symmray data is detected.
- `PepsOptimizer` accepts `torch_linalg_config=TorchLinalgConfig(...)` and
  should use that policy for global Torch cleanup. The legacy
  `register_torch_svd` switch remains only for compatibility.
- Preserve dtype/device behavior: driver selection changes the underlying
  decomposition, not `complex64`/`complex128` promotion. Add a focused
  reconstruction or gradient test when changing a backend route.

## Cyclic CTMRG compatibility

- Use `pepsy.boundary.quimb_ctmrg_projector_compat` around Quimb CTMRG calls
  for cyclic PEPS/PEPO networks whose effective bond dimensions can vary,
  especially native U(1) term-by-term replays.
- This is a scoped compatibility context: it redirects projector insertion to
  the current network and restores Quimb's method on exit. It does not modify
  installed `site-packages`, alter boundary-MPS contractions, or replace
  CTMRG with MPS.
- Keep the workaround at the Pepsy boundary API. Do not copy Quimb's CTMRG
  implementation into Pepsy or edit the installed Quimb source. Add a focused
  regression test when changing the compatibility behavior.
- The shared CTMRG entry points already apply this context for
  `contract_flat(..., method="ctmrg")`, native Torch PEPS VMC models with
  `contraction="ctmrg"`, and the NetKet/JAX PEPS amplitude validation path.
  Keep exact, HOTRG, and boundary-MPS routes independent of this workaround.

## Gaugy downstream joint Pauli-cluster validation

The example
`gaugy_examples/gauge_mps/mpsg/join_pauliexpansion.ipynb` optionally uses
Pepsy through Gaugy's HRPS/MPS comparison. In that notebook,
`mps_mode="exact"` means an exact state-vector contraction for the selected
sample and target state; it is not an exact global operator trace. Do not
compare that state cost directly with Gaugy's normalized joint Pauli-cluster
operator cost. The Pauli-cluster target and its spatial-cutoff convergence are
Gaugy concerns and do not require changes to Pepsy's tensor-network code.
