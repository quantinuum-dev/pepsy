# 2026-09-29 — Six-site cluster MPO dimension audit

The user requested verification of the dimensions in
[cluster_1d.ipynb](../../../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb)
and an explanation of fixed versus compact channels. This audit changes
notebook diagnostics and explanations, not package implementations or
compression settings. Measurements use the current working trees, including
pre-existing package changes; they are not clean-baseline measurements.

## Meaning and measured dimensions

Both representations target the same cluster operator `C_p` for the six-site
ITF plus NNN model, with `J1=1`, `J2=0.5`, `h=1`. Fixed factorization uses
shape-based identity splits and preserves redundant virtual directions without
SVD. Compact assembly uses automatic local factorization and recursive SVD
compression. A channel is a virtual-index value; these labels describe its
representation, not different Hamiltonians or cluster approximations.

The plot reports the maximum **stored** bond dimension. At `t=0.06`:

| p | OBC fixed | OBC compressed | PBC fixed | PBC compressed |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 1 | 1 |
| 2 | 92 | 23 | 384 | 50 |
| 3 | 265 | 35 | 1065 | 57 |
| 4 | 553 | 37 | 1641 | 56 |
| 5 | 649 | 35 | 1737 | 56 |

At p=5, the full OBC bond profiles are `(250,502,649,502,250)` and
`(4,16,35,16,4)`; PBC profiles are `(519,1266,1737,1266,519)` and
`(4,16,56,16,4)`. Both are open-chain MPOs in physical site order 0 through 5;
PBC describes the interaction graph. The notebook asserts this storage layout
and checks that `max_bond()` equals the maximum of actual `bond_sizes()`.

An exact rank-minimal six-qubit MPO has bonds at most `(4,16,64,16,4)`, from
the dimensions of each operator Schmidt unfolding. Larger stored bonds are
valid when they retain redundancy. Compressed dimensions depend on the
singular-value spectra and cutoff and need not increase monotonically with p.

Recursive assembly retains the existing cap 64 and cutoff `1e-26` with explicit
`rsum2` mode. This bounds locally discarded relative squared singular-value
weight, not the final global error. The notebook independently computes
numerical Schmidt ranks at relative singular-value threshold `1e-12`; these
are diagnostics, not exact algebraic ranks or the assembly compression rule.
At p=5 they are `(4,16,30,16,4)` for OBC and `(4,16,50,16,4)` for PBC.
The compressed representation is not guaranteed to have minimum dimension.

## Validation measured in this session

- Full notebook execution passed with all assertions and saved regenerated
  MPO build, dimension-audit, and bond-plot outputs. Other saved outputs were
  preserved. The new cell compares all five stored bonds at `t=0.06` and checks
  discarded Schmidt-tail lower bounds against measured compression error.
- Maximum relative Frobenius errors against independent explicit `C_p`:
  fixed `5.9127e-14` across 10 matrices at `t=0.06`; compressed `3.0876e-13`
  across 30 matrices at times `0.03,0.06,0.12`. Both cover p=1 through 5 and
  OBC/PBC. Maximum direct fixed/compressed difference is `2.4268e-13` across
  the 10 comparisons at `t=0.06`.
- An independent unfolding sanity check gave identity ranks `(1,1,1,1,1)`
  and a SWAP across sites 2 and 3 ranks `(1,1,4,1,1)`. Left input and output
  indices must be grouped together; the full operator's matrix rank is not
  its MPO bond rank.
- Six focused tests passed: `tests/test_mpo_cluster_compression.py`,
  `test_recursive_compression_converges_and_reports_rank_reduction` in
  `tests/test_mpo_cluster_recursive.py`, and
  `test_fixed_mpo_zero_crossing_values_gradients_and_shapes` plus
  `test_fixed_mpo_preserves_disjoint_collections_and_factor_order` in
  `tests/test_cluster_fixed_factorization.py`. These included NumPy and Torch
  checks without skips. Fixed shapes persist through zero crossings without
  SVD, and gradients and full matrices were checked.
- Notebook Ruff and strict KaTeX rendering of all 106 math expressions passed.
  The regenerated bond plot was visually inspected.

Agreement with `C_p` does not remove its finite-cluster error against the full
exponential. No full package suite, larger system, or native Symmray numerical
validation was performed in this session.

## Environment and upstream inspection

Used the device's existing Python 3.12 environment with single-threaded CPU
execution. Installed versions: NumPy 2.5.2, SciPy 1.17.1,
Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865.

Inspected installed signatures and implementation for
`MPOBasis.compile_graph_cluster_expansion`, `MatrixProductOperator.bond_sizes`,
`compress`, and `from_dense`, together with Pepsy's fixed split and recursive
accumulator compression. Public `bond_sizes()` includes the closing bond only
for cyclic MPO storage. These notebook MPOs are not cyclic.

- **Adopt:** use public bond inspection and preserve the explicit cutoff mode.
  The [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
  describes a default cutoff-mode change; the notebook already specifies
  `rsum2`, so its compression policy was not changed.
- **Defer:** dependency and backend changes; none is needed for this audit.
  Reviewed the [Autoray repository](https://github.com/jcmgray/autoray),
  [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
  [Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
  and [Symmray repository](https://github.com/jcmgray/symmray).
  The [Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
  could not be retrieved; the official repository and installed source were
  available instead.
- No compatibility shim or prototype was introduced; installed libraries
  were not modified.
