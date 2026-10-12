# CuPy vector-sampling precision correction

The Torch correction in 095cc7e is valid but does not cover CuPy. Installed
CuPy 14.0.1 `RandomState.choice` builds `p.cumsum()` in the input dtype.
For complex64 vectors, the float32 CDF loses small Born intervals. This is
a sampler defect, not evidence that time evolution or entropy is incorrect.

On an A100, public VecSampler with a uniform 25-qubit complex64 state,
8192 shots, seed 62, and chunk size 1200 returned lowest-three-bit one
fractions 0.245605, 0.244629, 0.254150 before correction. Expected: 0.5.
The corrected path returns 0.503784, 0.494263, 0.508545. Maximum per-bit
deviation is 0.010010 across uniform 16-, 20-, 25-, and 30-qubit tests.
This does not establish the size of bias in each production observable.

Use public `cupy.cumsum(probabilities, dtype=cupy.float64)`, normalize the
whole CDF by a copied final scalar, draw float64 uniforms, and search with
`side="right"`. Copying the scalar avoids aliasing during in-place division.
Returned weights retain their original dtype; no full promoted probability
copy or persistent CDF cache is added. The 30-qubit standalone sampling
probe used a 32.002 GiB CuPy memory pool, including cached allocations;
this is not a production-evolution peak-memory guarantee.

## Dependency audit

Installed versions: CuPy 14.0.1, Torch 2.11.0,
Quimb 1.15.1.dev75+g4112e304a, Autoray 0.11.1.dev9+g1291702f9,
Cotengra 0.8.3.dev7+g1d7fd333f, Symmray 0.4.1.dev11+g1a3481803.
Inspected installed cumsum, random_sample, and searchsorted signatures.
Classification: adopt supported public CuPy APIs; defer unrelated upstream
changes. No installed-library edits or dependency upgrades.

Consulted the official [CuPy cumsum API](https://docs.cupy.dev/en/stable/reference/generated/cupy.cumsum.html),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray).
The Symmray abelian-array documentation returned an error; no Symmray
behavior is changed.

## Validation and recomputation

Deterministic CuPy narrow-interval test checks correct outcome, source
immutability, probability dtype, and device. Statistical regressions cover
16/20/25 qubits; a standalone public-API probe additionally checks 30 qubits.
Sampler/public-API/layout selection before and after synchronization: 168 passed,
one unrelated installed-package version mismatch (metadata 0.4.0 versus
project 0.5.0). Ruff is absent from the active environment. Full suite not run.

The user authorized recomputing all 14 previous exact sweeps in fresh roots,
preserving old results and each run's entropy policy. Their archives do not
retain full vectors for resampling. The running old 5x6 dt=0.05 exact sweep
was stopped for replacement; the MPS job on GPU3 was preserved. Runtime
evidence and the campaign manifest are under `/tmp/pepsy_examples_runs/`.
