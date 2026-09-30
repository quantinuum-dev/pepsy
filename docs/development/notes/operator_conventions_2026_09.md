# Operator convention corrections — 2026-09-26

Classification: **adopt** Quimb's native operator semantics. No installed
library changes or compatibility monkey patches are involved.

## Evidence and implementation

Dense `build_mpo_from_gates` and `build_pepo_from_gates` previously transposed
every gate. For a stream `[A, B]` this produced `B.T @ A.T`, rather than the
intended `B @ A`. The old `tns_align` then treated `k...` as operator inputs,
which transposed the result but still left general noncommuting streams in
the wrong order. A Gaugy XY target with a nonzero Y field reproduced an
operator error of about 0.183 before the correction.

Builders now apply gates directly to upper/output indices; `tns_align`
connects the state to lower/input indices. Explicit `transpose=True` in
`tns_align` supports an intentionally transposed representation. It is not a
general migration for saved streams: rebuild those from their source gates.
Native Symmray builders already used direct gates and retain that behavior.

`gate_with_submpo` previously forwarded `inplace_mpo` as Quimb's target
`inplace` flag, then discarded the returned copy when false. The target
working copy is now always updated; `inplace_mpo=False` independently copies
the applied MPO. Upper and lower lazy application retain Quimb's respective
left and right multiplication semantics.

## Upstream check

Checked in the unchanged local environment on 2026-09-26: Python 3.12,
Quimb `1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`,
Cotengra `0.8.3.dev7+g1d7fd333f`, and Torch `2.9.1`.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray source](https://github.com/jcmgray/symmray). The Symmray array
documentation was unavailable through the web reader; installed source and
native regressions supplied the relevant evidence. Installed Quimb lazy
application signatures and implementations confirmed that `inplace` refers
to the target and that `b...` is the default operator input family.

## Validation scope

`tests/test_operator_conventions.py` uses independent dense oracles for
complex noncommuting MPO/PEPO streams, matrix and rank-four gate layouts,
lazy state action, input preservation, both operator legs, transpose, and
copy options. The existing gate suite separately covers native U1U1, U1,
and Z2 paths. Downstream Gaugy regressions cover the XY target, lazy and
flattened PEPS application, lower-leg tensor gates, and zero-overlap losses.
See `history/2026-09-26-operator-convention-fixes.md` in the repository for
final validation results and publication status.
