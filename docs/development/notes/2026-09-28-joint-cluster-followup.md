# 2026-09-28 — Follow-up review of joint cluster implementations

This review rechecked the working-tree joint MPO/PEPO changes on
`develop` baseline `4e398e4`, after the earlier
[joint audit](2026-09-28-joint-cluster-audit.md). No production code or
dependency was changed. The checks below used the existing Python 3.12
environment, CPU only and one BLAS/OpenMP thread.

## Independent probes

| Case | Reference | Largest observed matrix-entry error |
| --- | --- | ---: |
| Cyclic 2×2 joint PEPO, directed XY/ZZ then YX/XX terms, order four; reuse off/on and declared translation | Independently embedded full Hamiltonians, then dense ordered exponentials | `4.45e-16` |
| Mixed uniform/located joint PEPO, order four, trainable onsite/edge and factor coefficients | Dense ordered Torch exponentials | `3.33e-16` |
| Joint graph MPO, order four, same trainable-factor model | Dense ordered Torch exponentials | `1.12e-16` |
| Joint graph MPO, streaming and recursive numerical assembly, orders two through four, `assembly_chi=64` | Independent connected set-partition expansion | `3.76e-14` |

The cyclic PEPO probe retained both length-two parallel bond occurrences.
Automatic reuse reduced local target evaluations from 13 to five; a
validated one-row translation declaration gave the same operator and the
same five evaluations. The mixed-binding Torch probe used independent
trainable term values `h` and `j`, trainable factor weights `alpha`
and `beta`, and trainable time. Across nonzero values and a point with zero term
coefficients and time, the maximum gradient differences from dense ordered
products were `1.11e-16` for PEPO and `6.94e-18` for MPO.

A durable
[periodic joint regression](../../../tests/test_cluster_correctness_review.py)
now checks directed and parallel bonds with reuse off/on and an explicitly
declared translation against a dense ordered product. The full
correctness-review file passed **20 tests** after this addition. The
broader fixed/symmetry/recursive/JIT selection passed **93 tests**
before the final regression was added. Ruff, whitespace, and affected
relative-link checks passed.

## Conclusion and limits

No additional numerical mismatch was found in these finite cases.
The existing 5×6 order-four graph MPO measurement remains inaccurate at
assembly bond caps one and two; complete collection planning does not
establish compression convergence. The review does not certify large-system
operator error, GPU/native-sector behavior or full Torch builder graph
capture. The temporary numerical probes were not promoted to a large
benchmark or a new performance claim.
