# 2026-09-29 — Exact channel options for the 1D notebooks

- Scope: answer whether analytical reductions can avoid the large fixed MPO
  before numerical SVD, using the current NN + NNN notebook model.
- Baseline: Pepsy `develop` at `c1ff0f8`; the experimental channel implementation
  was already present as uncommitted work. No implementation or notebook edits
  were made for this investigation. This handoff is uncommitted; nothing staged
  or published.

## Findings and measured checks

The [channel API](../docs/api/operators/cluster_channels.md) supports exact
frontier preparation plus formal transition sharing. With `max_bond=None`, it
retains the finite cluster sum without reference-based QR projection.
Frontier preparation bypasses full collection enumeration and the original
MPO assemblers; prepared evaluation contracts directly into final tensors.
The notebook currently uses recursive numerical SVD during assembly instead.

Executed a temporary NumPy probe with six OBC sites, J1=1, J2=0.5, h=1,
time=0.06, and the notebook joint XX/YY/ZZ matrix order. A fixed recursive
source supplied the same geometry, cluster size and reflection symmetry.
Compared each result against the independent `ExplicitClusterSum` matrix.

| Target / size | Original fixed bonds | Frontier bonds before sharing | Exact shared bonds | Relative matrix error |
| --- | --- | --- | --- | --- |
| Single / 3 | 172,256,265,256,172 | 25,125,165,125,25 | 5,33,65,81,13 | 9.06e-15 |
| Single / 5 | 250,502,649,502,250 | 73,321,533,321,73 | 5,33,209,129,13 | 2.38e-14 |
| Joint / 3 | 172,256,265,256,172 | 25,125,165,125,25 | 5,33,65,81,13 | 1.22e-14 |

During exact plan preparation and evaluation, patched the source `exp`,
original MPO assemblers, graph collection planner, NumPy SVD and SciPy QR to
raise if called. All three probes passed. Reports showed zero full collections
enumerated and zero replay history blocks. Temporary reproduction script:
`/tmp/probe_notebook_exact_channels.py`; results:
`/tmp/notebook_exact_channels_probe.json`.

Environment: existing Python 3.12, NumPy 2.5.2, SciPy 1.17.1, Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Pepsy 0.5.0 from this checkout; single BLAS/OpenMP thread.

## Limits

These are stored dimensions, not measured peak-memory reductions. Preparation
still constructs frontier transitions and maps that can exceed final output
dimensions. Formal residual entries remain independent; this is not global
symbolic linear-dependency minimization. No arbitrary small exact bond bound
is promised. A smaller optional QR cap is approximate, and cluster truncation
remains separate. No new PBC, gradient or full-suite run was performed here;
earlier implementation validation is in the
[frontier handoff](2026-09-29-frontier-cluster-channels.md).

Reviewed the primary papers by
[Crosswhite and Bacon](https://arxiv.org/abs/0708.1221) and
[Hubig et al.](https://arxiv.org/abs/1611.02498) to distinguish finite-state
construction from SVD, deparallelisation and delinearisation. General
parameter-family symbolic minimization remains unimplemented in this path.
