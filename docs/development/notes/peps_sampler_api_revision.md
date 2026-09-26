# 2026-09-26 — PEPS sampler public API revision

Status: implemented in the working tree on `develop` / `80f451a`.
Scope: the user's requested additional review and API fix, following the
[final sweep audit](peps_sampler_final_sweep_audit.md). No commits, publication,
dependency changes, sibling edits, or production-job changes were made.

## API decisions

- **Adopt:** preferred constructor names `chi` for the cached future
  double-layer boundary and `chi_prime` for the conditioned ket boundary.
  Existing `marginal_chi` and `sample_chi` keywords remain supported. Both
  spellings are validated, then non-None duplicates must agree. An explicit
  None leaves resolution to the other spelling; zero chi means identity
  future caps. Read-only properties expose the resolved canonical names.
- **Adopt:** automatic engine selection. `PepsSampler(peps)` remains exact.
  Supplying positive caps selects DMRG future preparation by default, fixing
  the previous constructor/docstring disagreement for `sample_chi` alone.
  `boundary_engine=None` follows auto selection. Explicit exact mode still
  rejects positive caps. A positive chi with the default compressor requires
  chi_prime; the two caps are never implicitly tied together.
- **Adopt:** explicit `ket_compression=None` allows an omitted chi_prime in
  boundary mode, using the existing uncompressed path. No contraction or
  normalization algorithm was rewritten in this revision.
- **Adopt:** `PEPSSampleResult.log_probabilities`, `log_abs_amplitudes`, and
  `log_weights` compute natural logarithms from the existing scaled pairs.
  They return host NumPy arrays, supporting native backend scalar amplitudes
  from the shared BP result container as well. They do not materialize powers
  of ten or normalize the importance weights. Existing fields are preserved.
- Updated [API documentation](../../api/sampling/samplers.md), changelog,
  owning/root export checks, and a runnable
  [small example](../../../examples/peps_sampling.py). Removed a reference to
  an absent development example.

The upstream audit from the preceding same-task review was reused in the
unchanged environment. This revision adds no upstream compatibility shim;
backend conversion uses the existing module helper. Future-boundary and ket
compression providers are unchanged. No performance claim is made.

## Validation from this revision

Environment: project py312 interpreter, checkout via `PYTHONPATH=src`, one
BLAS/OpenMP thread. JAX restricted to CPU, with two devices in focused tests.

- Targeted API checks: **25 passed**, 76 deselected. These compare canonical
  and legacy seeded samples and likelihoods, uncompressed boundaries versus
  exact Born probabilities, conflicting/invalid options, extreme scaled
  logarithms, zero amplitudes, empty results, and native scalar conversion.
- Complete focused PEPS suite: **101 passed** in 116.44 s, with 13
  existing Quimb mode-to-method compatibility warnings.
- Existing sampler/public API/package-layout selection: **162 passed, 1 failed**.
- Default repository smoke: **153 passed, 1 failed**.
- Both broader failures are the pre-existing
  `test_package_version_matches_installed_distribution`: installed/runtime
  version 0.4.0 versus project metadata 0.5.0. No environment metadata was
  changed and no full-suite success is claimed.
- Small native Torch CUDA complex128 check: canonical caps, automatic backend
  inference, batch log probabilities versus queried probabilities, original
  PEPS log weights, unchanged ket arrays/device, and CUDA scalar result
  conversion all passed.
- Runnable 2×3 D=2 example: largest sampled log-probability difference versus
  exact was **8.9e-16** with chi=8 and chi_prime=4.
- Ruff over src/tests and the example passed. Final whitespace/link checks
  are recorded in the session handoff.

Temporary logs: `/tmp/peps_sampler_api_new_tests.log`,
`peps_sampler_api_focused.log`, `peps_sampler_api_broader.log`, and
`peps_sampler_api_smoke.log`. The implementation snapshot before this API
revision is `/tmp/peps_sampler_before_api_revision.py`.

## Remaining limits

The [earlier numerical limits](peps_sampler_final_sweep_audit.md) still apply:
truncation can remove proposal support; logarithmic outputs cannot fix absent
support or every ill-scaled raw contraction; native Symmray and JAX GPU are
unverified/unsupported as described there. The installed Quimb provider still
emits its existing mode-to-method compatibility warning. Log-result access
copies backend scalar data to the host; it is not a differentiable native
batch-output interface.
