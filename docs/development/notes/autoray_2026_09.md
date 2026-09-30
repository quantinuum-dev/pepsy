# Autoray opportunity and compatibility audit

Audit date: 2026-09-24. Initial assessment followed by the authorized
compatibility fixes recorded below. Dependencies were not upgraded.

## Sources and environment

Reviewed the [Autoray documentation](https://autoray.readthedocs.io/en/latest/index.html),
[release notes](https://github.com/jcmgray/autoray/releases),
[API reference](https://autoray.readthedocs.io/en/latest/autoapi/autoray/autoray/index.html),
[lazy computation guide](https://autoray.readthedocs.io/en/latest/lazy_computation.html),
[compilation guide](https://autoray.readthedocs.io/en/latest/compilation.html),
and [gradient helper reference](https://autoray.readthedocs.io/en/latest/autoapi/autoray/grad/index.html).
Pages returned different documentation revisions, so installed source and
behavior probes take precedence for the current environment. Some changelog
and random-guide URLs failed to load; release notes supplied those details.
The related Quimb/Cotengra/Symmray audit is recorded in
[`cotengra_2026_09.md`](cotengra_2026_09.md).

Installed: Autoray `0.11.1.dev3+g1b476b305`, Quimb
`1.15.1.dev55+gd0591eb70`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.3.2.dev8+g6c6dd34b5`, Torch `2.9.1`, NumPy `2.5.2`. Probes used the
device-local `genpy` environment. CUDA and MPS were unavailable to this
process; device-protocol observations are not GPU execution validation.

Confirmed signatures include `get_namespace(like=None, device=None,
dtype=None, submodule=None)`, `infer_backend_device_dtype(like, device=None,
dtype=None)`, `to(tree, like=None, *, backend=None, dtype=None, device=None)`,
`DoFunc(fn)`, and `autojit(fn=None, *, backend=None, compiler_opts=None)`.
`random.array` is available through dispatch with `(shape, dist='normal',
loc=0.0, scale=1.0, dtype=None, device=None, rng=None, backend=None)`;
`autoray.random_array` is not a top-level callable in this build.

## Findings and priorities

### 1. Adopt upstream device handling; narrow the legacy shim

Autoray 0.9 fixed namespace cache keys for unhashable device objects.
Pepsy's `backends/config.py::_patch_unhashable_device_namespace_key`, called
by `backend_cupy`, still unconditionally replaces the global Autoray and
Quimb namespace entry points. It transforms the actual device argument into
`"device:<id>"`, rather than only normalizing a cache key.

A CPU-only mock backend with an unhashable device confirmed that current
Autoray passes the original device object to a registered creation function.
After installing Pepsy's shim in the isolated process, the same creation
function receives the string `"device:7"`. This establishes a device-payload
change, not a demonstrated CUDA failure. A future fix should capability-test
upstream behavior, skip the shim when unnecessary, and preserve genuine
device objects on the old-version path. Actual CuPy creation needs GPU
regression coverage before claiming full compatibility.

### 2. Compatibility shim: align complex random scale across versions

Pepsy already adopts Autoray 0.10's `random.array` in
`_internal/random.py`, used by MPS/MPO/tree/stabilizer FIT and MPS transfer
calculations. This is useful for backend/device-native initialization and
backend-specific generators; it is not an unimplemented feature.

The legacy fallback samples independent standard-normal real and imaginary
parts without dividing by sqrt(2). Autoray's complex normal convention has
unit total variance. With 200,000 complex128 samples, seed 123, and scale 1:

| Route | Mean squared magnitude |
| --- | ---: |
| Current Autoray route | 0.999864 |
| Simulated missing-API fallback | 1.998706 |

The standard deviation therefore differs by sqrt(2). Harmonize the documented
meaning of `scale` and add a focused statistical/reconstruction regression;
do not claim identical random sequences across backends. The fallback probe
simulated the missing operation in current Autoray, not an entire old stack.

### 3. Prototype selective conversion-helper reuse

`to(...)` supports nested collections, repeated-reference preservation, and
integer-index dtype preservation. A NumPy-to-Torch probe confirmed shared
output identity and retained int64 indices. A same-backend Torch dtype
conversion retained gradients.

It is not a drop-in replacement for `backends/convert.py`: converting
`1+2j` with a float64 Torch template via `ar.to` returned `1` with a warning;
Pepsy's converter preserved `1+2j`. Installed `ar.to` routes cross-backend
conversion through `to_numpy`/`from_numpy`, so it does not promise zero-copy
GPU transfer or cross-framework autograd. Preserve Pepsy's complex-promotion
policy and native Symmray block metadata. Start with dense preparation-time
conversions, not global substitution.

### 4. Adopt existing namespace/registration support; profile further use

Cached namespaces already appear in scalar bookkeeping, tensor conversions,
and tree compression. `DoFunc` provides call-time backend dispatch for a
fixed operation; a bound namespace resolves functions earlier. Profile
remaining Python-heavy small kernels before adding more caching. This does
not by itself accelerate a large SVD or eliminate a host scalar read.

Release 0.9.1 notes describe old held namespaces retaining resolved functions.
The installed development source now invalidates function/submodule caches
in live namespaces in place: an isolated registration probe returned the
new function through both the held and freshly requested namespaces. Saved
function objects are a separate concern; configure Pepsy's Torch linalg
policy before binding/compiling numerical kernels. No change to the existing
rank-aware QR/SVD policies is recommended.

### 5. Prototype fixed-shape compilation; retain existing compiled paths

Installed compiler dispatch distinguishes `backend='torch'` (JIT tracing)
from `backend='torch:compile'` (Dynamo). A tiny complex squared-norm kernel
using `ar.do` failed on a cold `fullgraph=True` compilation because dispatch
attempted an import. After an eager warmup, both the original kernel and a
variant with pre-resolved namespace functions passed through
`autojit(..., backend='torch:compile', compiler_opts={'torch:compile':
{'backend': 'eager', 'fullgraph': True}})`. Value 30 and gradient
`[2+4j, 6+8j]` matched the analytical result.

This is graph/gradient validation, not an Inductor or speed benchmark.
Candidate applications are repeated fixed-shape numerical kernels. Adaptive
truncation, data-dependent convergence, and changing tensor topology need
separate treatment. Pepsy's Torch VMC already has an opt-in export/vmap/compile
route; do not replace it just to introduce another wrapper.

Autoray lazy graphs can fold constants and share intermediates. Pepsy's qMERA
compiler already delegates constant-bearing expressions to Cotengra and
explicitly preserves native Symmray dispatch. Sharing across multiple
observable expressions is a possible later prototype, with parameter updates
remaining explicit inputs and measured cache/memory tradeoffs.

### 6. Defer generic stop-gradient replacement and new backend promises

Upstream Torch `stop_gradient` detaches without copying. Pepsy intentionally
registers `detach().clone()` for an independent snapshot. Removing copies
could help selected large, immutable bookkeeping arrays, but changes aliasing
semantics; require a call-site mutation audit first. No generic replacement
is proposed. Likewise, Autoray MLX support alone does not establish Pepsy
complex-linalg, decomposition, or native-Symmray support.

Pepsy still declares `autoray>=0.6`. Any new adoption should be capability
gated or accompanied by a tested minimum-version change; this assessment
does not establish full compatibility at that floor.

## Validation and scope

- `python -m pytest -q -o addopts='' tests/test_backends.py`: **64 passed**,
  with expected rank-deficient QR warnings. Includes relevant Torch/native
  Symmray split-driver checks; not a full package or GPU run.
- Isolated behavior probes: complex random scale, device payloads,
  conversion policy, same-backend gradient preservation, pytree aliasing,
  registration invalidation, and compilation cold/warm behavior.
- Temporary probe: `/tmp/pepsy_autoray_audit.py`. Registrations and simulated
  compatibility changes were confined to that process. No installed files
  or runtime package sources were edited; no commits or pushes.

## Implemented follow-up: device compatibility and random initialization

The initial assessment above describes the pre-fix behavior. The authorized
follow-up changes `backends/config.py` and `_internal/random.py`:

- Namespace compatibility now probes the actual upstream getter with an
  unhashable device argument without allocating arrays. A working getter is
  left untouched. On affected versions, the wrapper first tries the normal
  getter and only handles a TypeError accompanied by an unhashable device.
  That case constructs an uncached public `AutoNamespace`, preserving the
  original/inferred device, dtype, and submodule. Hashable-device errors
  propagate normally. Repeated installation is idempotent.
- The legacy complex normal draw is divided by sqrt(2), matching modern
  Autoray's total-variance convention. Real draws retain their scale.
- Regression tests also uncovered explicit `dtype=None` suppressing
  Autoray's template inference. The helper now resolves an omitted dtype
  from `like` before dispatch. Fallback conversion passes its resolved dtype
  explicitly, so a real template cannot discard an explicit complex override.
  NumPy/Torch dtype objects are normalized for old dispatch APIs, and the
  staging dtype does not replace requested float16 output precision.

Upstream audit sources were rechecked before edits. Installed versions are
unchanged from the table above. The Symmray documentation URL again failed
to load; its repository was available.

Validation:

- Backend, Quimb compatibility, and contraction dependency suites:
  **83 passed, 1 skipped** (CuPy unavailable), with expected QR warnings.
  This includes native Symmray/Torch split-driver regressions.
- Existing random/SRC regressions across MPS, tree, stabilizer, and
  successive tree compression: **32 passed, 12 skipped** (unavailable
  optional backends), 165 deselected.
- Isolated namespace tests exercise current and simulated legacy behavior,
  inferred and explicit device identity, dtype injection, idempotence,
  NumPy passthrough, and propagation of an unrelated TypeError.
- Downloaded actual Autoray **0.8.11** without dependencies to
  `/tmp/pepsy-autoray-0811`; the namespace regression passed using its real
  legacy cache and namespace implementation. NumPy and Torch fallback draws
  retained complex64 and had mean squared magnitude **0.03999455** for
  `scale=0.2` (expected 0.04). The active environment was not modified.
- CUDA creation has an executable optional regression but could not run on
  this machine. Protocol tests are not evidence of GPU execution.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Full package suite not run; validation targeted the changed compatibility
  boundaries and their numerical consumers. No commit or push.
