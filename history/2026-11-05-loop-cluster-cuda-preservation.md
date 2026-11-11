# 2026-11-05 — Preserve actual CUDA arrays in cluster gate streams

- Scope: user-requested GPU profiling of local PEPO cluster updates.
- Branch/baseline: `develop`, `ee22198`, with the existing local-cluster
  working-tree changes. Uncommitted; no staging or publication.
- Original GPU-labeled benchmark prefix was actually NumPy/CPU: all 100
  checkpoint tensors and 180 gauges were NumPy. The generic `to_dense()`
  feature check in `gate_loop_cluster` converted dense Torch arrays too.

Removed that conversion path. Native symmetry support is already checked
explicitly; dense tensors, gauges and gate payloads retain their backends.
Four new PEPS/PEPO CPU/CUDA stream cases assert actual tensor/gauge devices
after mixed one-/two-site gates. The downstream benchmark additionally
asserts runtime devices and serialized checkpoint tensor devices.

Validation: the complete reduced-update, cluster-patch and local-gauge suites,
three native reduced cases and two SymPEPS wrappers passed **93 tests**
(11.70 s), with CUDA cases executed. Four downstream middle-out tests passed
with explicit checkpoint device assertions. Earlier wider-suite baseline
fermionic readout failures remain outside this fix. Full Ruff and whitespace
checks passed; the complete repository suite was not run.

The maintained BP skill and prior upstream audit were reused with unchanged
dependencies. No installed library or unrelated running job was modified.
The original first prefix and failed profile artifacts remain preserved;
its queued full run was canceled and a corrected frozen GPU trial started.

The [downstream profiling record](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/2026-11-05-eli-loop-cluster-gpu-profile.md)
contains the phase timings, frozen-source provenance and fixed-environment
ALS ladder. Cached CUDA pair updates averaged 108.7 ms on twelve early
interior lower-leg pairs. PSD eigendecomposition accounted for 30.9% of the
synchronized warm sum, and all ALS work for 23.3%. Current tol1e-8 fits stopped
after one sweep; tol1e-12 needed up to seventeen sweeps. These are local,
early-state measurements rather than a late-time convergence certificate.

The corrected downstream prefix completed 1,056 updates in 111.60 s. Its
serialized checkpoint independently contains 100 Torch tensors and 180 Torch
gauges on cuda:0. The full corrected trial then launched with the original
tol1e-8/cap10 settings; no final-time convergence result is claimed.
