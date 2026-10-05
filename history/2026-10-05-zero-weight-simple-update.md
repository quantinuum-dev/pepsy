# 2026-10-05 — Exact-zero weights in dense simple update

- Scope: dependency fix for Gaugy's nonfinite zero-cutoff SU target regression.
- Branch / baseline: `develop`, `d7d509b`.
- Publication: this record accompanies the focused dependency commit; Git
  records its final hash. Pre-existing solver, MPO, sampling and MPI edits
  remain outside this commit, including unrelated changelog/status hunks.

## Correction

Quimb's SU temporary outer-gauge removal takes a reciprocal. With dense
`cutoff=smudge=0`, exact-zero retained weights caused NaNs on a later gate.
Pepsy now scopes an exact support projection around each adjacent update,
including routed SWAPs, using public tensor selection and diagonal operations.
For `safe = where(g == 0, 1, g)` and `P = diag(g != 0)`, `safe * P = g`.
Project before insertion and after ungauging, then restore the original
external gauge objects in `finally`. The updated inner gauge remains live.
No nonzero values, cutoffs, smudges, bond dimensions or global registrations
are changed. Gaugy uses the existing public `pepsy.gate_simple` entry point.

This dense adapter does not change native Symmray support. Derivatives are
tested for fixed null support; rank-changing derivatives are not established.
Contractions that split a zero weight into two square roots have a separate
singular derivative. The gradient regressions absorb each full weight once.

## Validation

- Gate, automatic cutoff, scale, public API and package layout selection:
  **241 passed**, three existing warnings (6.18 s).
- New cases include NumPy/Torch dense references, adjacent/routed updates,
  all-zero and tiny nonzero weights, input ownership, exception restoration,
  and Torch finite differences with bond cap 1 (actual truncation) and 4.
- Package-wide Ruff passed. Full Pepsy suite was not run.
- Downstream Gaugy validation is recorded in its matching dated handoff.

## Dependency audit

Installed versions: Quimb `1.15.1.dev79+gb5e316200`, Autoray
`0.11.1.dev9+g1291702f9`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev11+g1a3481803`. Inspected installed `TensorNetwork.select`
(`virtual=True`), `gauge_simple_temp` (smudge/power/ungauge controls), and
Quimb/Pepsy `gate_simple` signatures. The installed outer-gauge removal has
no zero-safe inverse callback.

Checked the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), plus
the [Symmray repository](https://github.com/jcmgray/symmray).
The requested [Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
and its API-index fallback were unavailable through the browser; used the
official repository and installed implementation instead.

Decision: adopt public tensor/Autoray operations; scope the compatibility
adapter to dense, unsmeared external gauges. Defer native support and
rank-changing derivatives. No dependency upgrade, installed-library edit,
global monkeypatch or copied Quimb gate implementation is involved.
