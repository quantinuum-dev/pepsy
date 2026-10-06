# 2026-10-05 — Public paired-factor registration

Branch: develop. This record accompanies the local commit; the user explicitly
requested committing and pushing the associated Gaugy gradient correction.

Exposed the existing `register_projector_split` through `pepsy.backends` so
Gaugy can use the owning public namespace. No decomposition, derivative,
global registration policy or existing boundary default changes. The API guide
documents fixed-rank/gapped and gauge-invariant first-order use.

Validation in the existing py312 environment: **84 passed**, two deprecation
warnings, with `python -m pytest -q -o addopts='' tests/test_projector_split.py
tests/test_public_api.py tests/test_package_layout.py`. Full `python -m ruff
check src tests` passed. No full numerical suite or new native/JAX claim;
this change only adds an export and a public NumPy registration regression.

The upstream/version audit is unchanged from the active Gaugy task. Pepsy's
[original paired-factor derivation](../docs/development/notes/2026-09-26-projector-boundary-gradients.md)
and implementation remain the source of its numerical contract. The unrelated
untracked latest-commit review is preserved and excluded from this commit.
