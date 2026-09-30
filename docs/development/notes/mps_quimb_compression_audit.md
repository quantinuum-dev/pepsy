# MPS Quimb compression compatibility audit

## 2026-09-11

This audit covers `MpsOptimizer` and the shared dense `MpoOptimizer` Quimb
1D-compression surface, with particular attention to `src`, `src-oversample`,
`srcmps`, `sdc`, and the randomized `sdcr` variants.

Installed versions in the active `py312` environment:

- Quimb `1.15.1.dev51+g2e99c793e`
- Autoray `0.11.1.dev3+g1b476b305`
- Cotengra `0.8.3.dev7+g1d7fd333f`
- Symmray `0.3.2.dev8+g6c6dd34b5`

API probes:

- Quimb's `_TN1D_COMPRESS_METHODS` registry contains `src`, `src-oversample`,
  `srcmps`, `srcmps-oversample`, `sdc`, `sdc-oversample`, `sdcr`, and
  `sdcr-oversample` in addition to the existing direct, DM, zipup, and FIT
  methods.
- The installed `gate_nonlocal_` and `gate_with_submpo_` APIs forward
  compression options through `**compress_opts`; the concrete `src` and
  `srcmps` compressors explicitly accept `seed`, while `sdcr` does not expose
  a top-level seed parameter.
- The online Quimb API adds separate `cutoff_mode_oversample` and
  `compress_opts_final` controls for oversampled methods. The installed dev
  build does not name those options yet, so Pepsy does not forward them
  unconditionally.
- Autoray dispatch exposes backend-specific `random.array`, `linalg.svd`, and
  `linalg.qr` rules for NumPy, Torch, JAX, and CuPy. No Pepsy-side backend
  registration change is required for these MPS methods.

Upstream sources checked:

- Quimb changelog: <https://quimb.readthedocs.io/en/latest/changelog.html>
- Quimb 1D compression API:
  <https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tn1d/compress/index.html>
- Autoray repository: <https://github.com/jcmgray/autoray>
- Cotengra docs/changelog: <https://cotengra.readthedocs.io/en/latest/>
  and <https://cotengra.readthedocs.io/en/latest/changelog.html>
- Symmray Abelian-array docs/repository:
  <https://symmray.readthedocs.io/en/latest/abelian_arrays.html> and
  <https://github.com/jcmgray/symmray>

Disposition:

- **Adopt:** `sdcr` and `sdcr-oversample` as opt-in MPS/MPO mode aliases and
  FIT warm-start methods, guarded by the installed Quimb dispatcher.
- **Compatibility shim:** base `sdcr` ignores singular-value cutoff for its
  rank-controlled randomized environment and changes cumulative `sum*` /
  `rsum*` modes to `rel`; `None` remains untouched so native Quimb defaults
  continue to work. Existing `src`, `src-oversample`, and `srcmps` seed and
  cutoff behavior remains unchanged.
- **Defer:** public MPS run-level exposure of Quimb's newer
  `max_bond_oversample`, `cutoff_oversample`, `cutoff_mode_oversample`, and
  `compress_opts_final` controls. Adding them requires threading capability
  filtering through dense gates, sub-MPO events, FIT guesses, and copies;
  Quimb's native defaults remain available through the selected oversample
  methods.

## 2026-09-16 scalar conversion and roughening instrumentation follow-up

The active `py312` environment was rechecked before changing backend scalar
reads: Python 3.12.13, Quimb `1.15.1.dev51+g2e99c793e`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, and Symmray
`0.3.2.dev8+g6c6dd34b5`. Autoray's public `to_numpy` and `do` signatures were
probed directly. The MPS, MPO, PEPS, and trajectory scalar conversion paths
now use Pepsy's shared Autoray-backed `to_float` helper rather than extracting
backend values with a backend object's `.item()` method.

Disposition:

- **Adopt:** the public Autoray conversion path for scalar norm and trajectory
  probability reads; no installed dependency was modified.
- **Adopt:** roughening's disabled memory-report path is now inert, and its
  wrapper synchronization is opt-in with memory capture or explicit device
  timing synchronization.
- **Defer:** removing unavoidable scalar transfers from correctness diagnostics
  such as unitary norm stabilization; those reads preserve the existing norm
  ledger and require a separate asynchronous diagnostic design.
