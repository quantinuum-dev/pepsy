# Native JAX MPS sampling — 2026-11-05

Dense `MpsSampler` now draws and scores configurations with JAX arrays on the
source device. The [API guide](../../api/sampling/samplers.md#native-jax-mps-sampling)
owns supported layouts, keys, gradients and conversion behavior.

## Implementation and decisions

Classification: **adopt** public JAX JIT, categorical sampling and PRNG APIs.
No upstream shims, package changes or vendored implementations are required.
`_mps_jax.py` is imported only for JAX inputs. Shared array detection recognizes
JAX without importing it; NumPy, Torch, CuPy and Symmray routes remain distinct.

Preparation rescales each site and right environment by positive scalars.
Conditional prefixes are similarly rescaled, with probabilities accumulated
in logarithms. Normalized amplitudes retain complex phase and subtract the
scaled state's log norm. Environments keep bra-row/ket-column ordering; two
ordered contractions avoid a quartic bond-dimension intermediate.
Source tensors and Quimb canonical metadata are unchanged.

Compiled functions accept numerical leaves as runtime arguments. Instance
caches retain the current numerical environments; `refresh()` discards them
without discarding equal-shape compiled code. Inference stops gradients by
default; query probabilities and opt-in amplitudes/sampled probabilities admit
autodiff. Discrete categorical decisions are not differentiable. Failed traced
validity checks use a failure-only debug callback, following the existing MPS
normalization pattern; ordinary eager checks synchronize one validity scalar.
Callbacks recheck the actual predicate because outer `vmap` can execute both
conditional branches; valid batched members must not spuriously raise.

Categorical draws use `mode="high"` to improve rare-event resolution for
float32. Explicit real-dtype fallback rows avoid accidental float64 promotion
when JAX x64 is enabled. The sampler never changes global JAX configuration.

## Environment and upstream audit

Activated Python 3.12 environment: JAX/jaxlib 0.8.2, NumPy 2.5.2,
Quimb 1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a,
Cotengra 0.8.3.dev8+g8954240f2 and Symmray 0.4.1.dev15+g0374aaa3c.
Installed Pepsy metadata reports 0.5.0; checkout `pyproject.toml` declares 0.6.0.
Inspected public Quimb tensor transposition, JAX key/key-data, device placement,
JIT and categorical signatures, plus Autoray backend/conversion dispatch.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray repository](https://github.com/jcmgray/symmray), and
[JAX categorical documentation](https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html).
The Symmray abelian-array documentation returned an internal fetch error;
official source and installed capabilities supplied the fallback audit.

## Validation

- New native-JAX suite: **26 passed**, using two logical CPU devices. Covers
  all four supported dtypes, nonuniform physical/bond sizes, independent dense
  probability/amplitude oracles, empirical draws, integer/typed/legacy keys,
  source immutability, refresh, invalid inputs, global scales, JIT/vmap, complex
  gradients, opt-in sampled-probability derivatives, inference gradient stops,
  explicit host conversions and nondefault-device placement.
- Existing sampler, public API, layout and import suites: **184 passed,
  10 skipped, 1 failed**. The sole failure is
  `test_package_version_matches_installed_distribution`, caused by the
  installed/check-out version mismatch above. Packages were not reinstalled.
- Full `python -m ruff check src tests` passed. No full numerical-suite or
  GPU validation is claimed. CuPy/CUDA-dependent checks skipped as unavailable.

The initial test development found a real float32 fallback promotion and a
fixture reversing a size-one bond instead of its physical axis; both were
corrected before the passing run.

## CPU sampling probe

Matched real float64 open chains: Gaussian arrays (NumPy seed 8), physical
dimension 5, bond 8, 128 shots (seed 9), seven warm repeats. Each call constructs
a fresh sampler and returns host arrays; the comparison route additionally
copies JAX tensors to NumPy before native NumPy sampling. BLAS/OMP threads were
limited to one. Device: one local JAX CPU, no accelerator.

| Sites | Native JAX first call | Native JAX warm median | NumPy snapshot warm median |
| --- | --- | --- | --- |
| 8 | 727 ms | 1.28 ms | 0.92 ms |
| 40 | 1703 ms | 4.63 ms | 4.73 ms |

Native JAX removes tensor host transfer; these small CPU timings establish
neither a general speedup nor GPU throughput. Compilation is a material cold
cost. Multi-device sharding, JAX-backed Symmray, and flip-ratio acceleration
remain unsupported; scale conditioning still has finite-precision limits.
