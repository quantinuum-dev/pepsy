# 2026-09-29 — Direct Pepsy call for notebook MPO construction

- Scope: make the 1D notebook's MPO workflow explicitly use Pepsy's public
  terms + geometry + cluster-size constructor throughout.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: working-tree edits only, nothing staged, committed, or published.
  Preserved pre-existing changes in both repositories.

## Changes and API finding

The previous notebook already used Pepsy through
`MPOBasis.compile_graph_cluster_expansion(...).exp(...)`. Replaced both the
fixed and compressed construction paths with direct
`pepsy.operators.exp_mpo_cluster(terms, step, shape=N, graph=geometry,
cluster_size=p, ...)` calls. No notebook wrapper or cluster algorithm was added.

Input preparation builds the local X/ZZ terms and NN + NNN graph. Pepsy owns
local exponentials, residuals, complete disjoint-cluster products, MPO assembly,
and compression. Independent explicit references now live in a separate
validation cell; their residuals and matrices are not construction inputs.

Kept the Hamiltonian, fixed/compact policies, spatial symmetry declarations,
and all dimension/accuracy checks. The one-call API creates a fresh plan on
each call; spatial reuse still operates within each call. Removed the claim
that the notebook reuses compiled plans across times.

An installed-source probe confirmed that Pepsy rejects two-site interactions
at `cluster_size=1`. The notebook retains its existing explicit onsite-only
special case, giving `C1`; for every p>=2 it supplies the full ITF + NNN
Hamiltonian. This limitation is documented rather than changing package policy.

Files: [notebook](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb) and
[guide](../../gaugy_examples/pauli_gaugy/README.md). No package implementation
was changed. The source probe resolved the function to this local Pepsy
checkout's `src/pepsy/operators/mpo_product.py`. Compression and environment
are unchanged from the [dimension audit](2026-09-29-cluster-mpo-dimension-audit.md).

## Validation

- Full notebook execution passed, including 10 fixed and 30 compressed MPO
  comparisons, direct fixed/compact comparisons, and per-cut rank checks.
- `tests/test_mpo_cluster_recursive.py` and the term-centric facade and
  coordinate-graph/ordered-factor tests in `tests/test_mpo_cluster.py`:
  **30 passed** in this session.
- Notebook Ruff, notebook schema, all **110** strict KaTeX expressions,
  documentation links, and whitespace checks passed.
- Saved refreshed outputs for affected MPO cells and preserved other cells
  and figures. Source review confirms that the construction cell contains
  neither explicit cluster residual construction nor the old compiled API.

No full package suite or new backend validation was run; the small-system
evidence applies to the notebook's six-site defaults. The p=1 input restriction
and fresh-plan behavior above remain properties of the chosen API.
