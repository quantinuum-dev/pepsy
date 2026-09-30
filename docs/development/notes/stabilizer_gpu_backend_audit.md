# Stabilizer coefficient backend and GPU synchronization audit

Audit date: 2026-09-23. Public owners are `StabilizerMpsSimulator` and
`StabilizerTreeSimulator`; `MpsStabOptimizer` and `TreeStabOptimizer` remain
compatibility aliases. This change preserves the `C|nu>` representation,
Stim tableau, exact-cooling policy, canonical metadata, compression methods,
cutoffs, measurement probabilities, and segmented norm-survival contract.

## Upstream audit

Installed: Autoray `0.11.1.dev3+g1b476b305`, Quimb
`1.15.1.dev55+gd0591eb70`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.3.2.dev8+g6c6dd34b5`. Checked the
[Autoray current docs](https://autoray.readthedocs.io/en/latest/),
[dispatch/namespace guide](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html),
[compilation guide](https://autoray.readthedocs.io/en/latest/compilation.html),
[API reference](https://autoray.readthedocs.io/en/latest/autoapi/autoray/autoray/index.html),
and [repository](https://github.com/jcmgray/autoray), plus the
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
Autoray's linked random/changelog pages and Symmray's `abelian_arrays.html`
were unavailable through the browser. Installed callable signatures and
implementations were inspected instead; documentation version banners alone
were not used as capability evidence.

- **Adopt:** `get_namespace(like=array)` for cached early dispatch; array
  creation with explicit `like`; backend `stack`, `einsum`, reductions, thin
  SVD, `log10`, `exp`, `expm1`, `where`, and `stop_gradient`; backend
  `random.default_rng` plus `random.array` for FIT noise.
- **Compatibility shim:** when `get_namespace` is absent, use Autoray's late
  dispatch namespace with explicit `like` on creation calls. Retain the
  existing old-Autoray NumPy random fallback only when random APIs are absent.
- **Defer:** compilation/fusion of dynamic replay and Quimb rank selection.
  Autoray dispatch does not by itself eliminate scalar-driven control flow or
  make a Stim tableau executable on a GPU. No upstream numerical algorithm
  or global linalg registration is replaced.
- **Prototype:** none.

Probes verified `get_namespace(like=None, device=None, dtype=None,
submodule=None)`, `random.default_rng(seed=None, **kwargs)`, and
`random.array(shape, dist, loc, scale, dtype, device, rng, backend)`, plus
Torch dispatch for SVD, stack, where, log10, and stop-gradient. Complex64
namespace creation and random output preserved the example array's dtype and
device. Torch's `minimum(tensor, scalar)` is not NumPy-compatible here, so
upper clipping uses Autoray `clip(..., None, 0)`.

## Changes

1. Coefficient Pauli sub-MPOs were assembled in NumPy and uploaded repeatedly.
   Optional `like=` builders now assemble their existing bond layouts with
   backend stacks and diagonal channel contractions. Internal callers supply
   the live coefficient array. Local rotations/projectors combine cached
   backend constants; cache refresh follows the live backend/device/dtype.
   Real-valued coefficient states cache `-iY` for real Y rotations, avoiding
   the incorrect loss of Y when casting its imaginary matrix separately.
2. MPS stabilizer FIT passed a NumPy RNG to backend random generation. Its
   seeded generator now matches the coefficient array, including its device.
3. Ordinary MPS compressed-unitary norm reads and log-survival accumulation
   used Python scalars. They now stay detached on the backend, with public
   getters converting at readout. Invalid events carry a backend mask and are
   excluded at readout without a per-update host branch. Copy updates remain
   functional. Float32 display underflow does not prematurely erase the
   logarithmic cumulative ledger. Nonzero extracted exponents and explicit
   norm stabilization retain their validated host range and decisions.
4. Tree stabilizer inherits the ordinary TreeOptimizer backend norm ledger
   and Pauli construction changes. Its exact-cooling candidate previously
   downloaded an entire leaf matrix and ran two CPU SVDs. It now runs one
   backend thin SVD, reads one rank Boolean, and retains the candidate vector
   on-device. Both stabilizer engines compute Bloch values on the backend and
   read only three classical values to select a Stim tableau update.
5. Both stabilizer MPI routes default rank diagnostics to false. Existing
   `timing=False`, finite/overlap checks, and tree profiling defaults are
   preserved; no new replay clocks were added.

## Deliberate host boundaries

Stim/tableau decisions, bounded arbitrary dense gate classification and Pauli
decomposition, Born probabilities and selected-branch normalization, FIT
convergence/reporting, upstream decomposition rank selection, explicit
exponent/stabilization bookkeeping, and public readout still use the host.
Exact cooling remains on by default and needs its small classical decisions;
it is not removed to claim synchronization-free execution. Explicit injection
and reporting runner APIs retain their existing timing reports. Native
Symmray charge/fermion handling and QR safeguards are unchanged.

The backend scalar path uses float64 on Torch/CuPy where supported and native
precision on JAX/Metal. Array execution is backend-neutral through Autoray;
backend checks are limited to supported scalar policy and dtype capability.
No GPU speedup is claimed without hardware measurements.

## Validation

`tests/test_stabilizer_gpu_backend.py` covers Torch/JAX replay versus NumPy,
backend/dtype-preserving Pauli builders, seeded random FIT initialization,
cooling without coefficient matrix downloads, zero/nonfinite/underflow norm
policies, copy and measurement boundaries, and extreme exponent transitions.
Torch ordinary named replay also runs with scalar reads, NumPy downloads, and
profiling clocks forbidden when exact cooling is explicitly disabled to
isolate its separate classical decision boundary. CUDA/CuPy variants skip
when hardware/dependencies are unavailable on this CPU-only test machine.

Closest domain/integration checks include stabilizer MPS/tree, the stabilizer
sampler, trajectory/noise, MPI, simulator planning, Quimb compatibility, and
public API/package layout.

Final results: 699 domain/integration tests passed (23 skipped), including
gradient-detached histories, the namespace fallback, and real-valued Y
rotation coverage. The default smoke suite passed 130 tests; repository Ruff
and `git diff --check` passed.

The full suite was attempted. A macOS GUI Matplotlib abort in an unrelated
lattice-drawing test was avoided by rerunning with `MPLBACKEND=Agg`. That run
was stopped after 1,706 passes, 57 skips, and 87 failures concentrated in BP,
MPO, and ordinary MPS paths. Six representative failures reproduced against
an isolated archive of committed `HEAD`: the native reduced-loop BP test,
native MPO DMRG boundary test, three MPS metadata/cache tests, and native
spinful MPS DMRG2/U1. The native MPS case raises Symmray's incompatibility
between cumulative cutoff modes and eager max-bond selection. These failures
are outside the stabilizer change; a complete clean full-suite result is not
claimed, and the remaining failure instances were not individually rerun on
the baseline.
