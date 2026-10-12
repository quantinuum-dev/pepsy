# 2026-09-26 — PEPS sampler API revision

- Scope: user's requested additional sampler review and API fixes.
- Branch / baseline: `develop` / `80f451a`, ahead 8 at start.
- Commit status: working-tree edits only; nothing staged, committed, or
  published. Preserved existing sampler and unrelated edits.

## Implemented

- Preferred `chi` (cached future double layer) and `chi_prime` (conditioned
  single ket) options, compatible legacy aliases, conflict validation, and
  resolved read-only properties.
- Default auto engine selects exact without caps, DMRG with positive caps.
  Explicit exact rejects caps. Uncompressed boundary ket mode can omit χ′.
- Shared PEPS result natural-log probability, absolute amplitude, and
  unnormalized importance-weight properties, computed from scaled values.
- Public docs, changelog, runnable small example, numerical API regressions,
  and owning/root export checks. Algorithms/backend dispatch were reused.

## Validation

- Targeted API selection: 25 passed.
- Full focused PEPS suite: 101 passed, 13 existing Quimb compatibility
  warnings (116.44 s).
- Sampler/public API/layout: 162 passed, 1 known metadata failure.
- Smoke: 153 passed, the same failure: installed/runtime 0.4.0 versus
  pyproject 0.5.0 in `test_package_version_matches_installed_distribution`.
- Small Torch CUDA probe passed, including native scalar log-result access.
- Example agrees with exact sampled log probabilities to 8.9e-16.
- Ruff (src/tests/example), `git diff --check`, and local documentation links
  pass. No full-suite success claimed.

See [detailed evidence and decisions](../docs/development/notes/peps_sampler_api_revision.md)
and [public API](../docs/api/sampling/samplers.md).
The upstream audit from the preceding same-task review was reused with no
library changes. The sandbox/patch helper again failed before file access;
authorized edits used checked replacements through reviewed shell escalation.

## Remaining limits

Existing truncation-support, dense/OBC, raw-contraction scaling, and JAX GPU
limitations still apply. The installed Quimb boundary provider emits its
existing mode-to-method warning. Log result views copy scalar data to host
NumPy arrays. No production jobs or shared environments were modified.
