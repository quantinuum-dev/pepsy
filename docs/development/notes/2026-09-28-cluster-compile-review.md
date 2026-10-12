# 2026-09-28 — Review fixed-structure PEPO compilation

Scope: user requested another review and improvements toward fixing Hamiltonian
term structure, system size and lattice geometry once, compiling clusters, and
reusing them to construct PEPOs. Pepsy only; branch `develop`, baseline
`4e398e4` plus the preceding uncommitted geometry-cache changes. Nothing staged,
committed or published.

## Findings and implementation

Previously the located `compile_exp()` path enumerated cluster records but
left static operator maps and deterministic spanning trees to evaluation.
It keyed local maps by global site and edge ids, duplicating identical matrices
for translated clusters. Homogeneous orders above four left generic geometry
and source maps lazy; mixed uniform/located products prepared uniform factors
for the homogeneous route even though evaluation uses the located route.

- Prepare located maps and deterministic tree metadata during compilation.
- Share static NumPy maps using site count and the ordered local endpoint
  pairs. Keep repeated endpoint pairs and their order; global site/bond indices
  still select independent dynamic coefficients. Tree metadata uses the full
  directed geometry, including lattice directions.
- Prepare homogeneous higher-order source records and maps during compilation.
- Prepare every mixed-product factor for its actual evaluation route.
- Make preparation idempotent per route. Expose prepared modes and numbers of
  located embedding/tree plans in `cache_info`.
- Document the fixed-H and parameterized-H compile/reuse workflows, including
  rebuilding a basis after structural changes. Existing numerical subtraction,
  contraction, factorization and backend conversion policies are unchanged.

Compilation retains static maps and geometry, not parameter-dependent
exponentials, residuals, SVD factors or backend autodiff graphs. Dynamic block
assembly and numerical residual contractions still run at each evaluation.
Full PEPO construction is not claimed to be fully precomputed. When H and the
step are both fixed and no new gradient graph is needed, the caller can reuse
the constructed operator directly.

## New validation

- Cluster domain, public API and layout selection: **122 passed**, two existing
  compatibility warnings, 37.08 seconds, with `OPENBLAS_NUM_THREADS=1` and
  `OMP_NUM_THREADS=1` in the existing Python 3.12 environment.
- Added two evaluations with different coefficients and time steps, comparing
  both operator values and Torch gradients against independent dense matrix
  exponentials on a full three-site cluster. Guards reject static embedding or
  tree recomputation after compilation.
- Added mixed uniform/located factor reconstruction and compile-route checks.
- Strengthened the five-site chain test to require its static maps to exist
  before evaluation while retaining its dense reference comparison.
- Full Ruff and `git diff --check`: passed. Full repository suite not run.
- An initial test guard rejected all embedding calls, including necessary
  coefficient-dependent residual embeddings. Restricted the guard to missing
  static Hamiltonian maps; numerical tolerances and reference checks unchanged.

## Static preparation measurement

5x6 OBC, order 4, a located X slot, median of five runs. Compare the pre-review
embedding method with the new compiler in one process. Before: enumerate
records, prepare every embedding map and deterministic tree once, as their
first evaluation would. After: `compile_exp()` prepares that static work.
Basis construction is excluded from both timings. Existing simulations were
running concurrently.

| Quantity | Before review | After review |
| --- | ---: | ---: |
| Actual finite connected clusters, sizes 1–4 | 492 | 492 |
| Stored local operator maps | 492 | 15 |
| Operator-map arrays, bytes | 84,268,544 | 2,828,544 |
| Static preparation, seconds | 1.23382 | 0.04167 |

Maps sharing local endpoint structure can coincide even for distinct geometric
orientations; their PEPO tree directions and placement coefficients remain
separate. Memory figures count the static map arrays only. These are roughly
30-fold reductions in static preparation time and map memory, not measured
end-to-end PEPO build speedups. The prior compiler deferred work, so the timing
comparison includes that formerly deferred work rather than comparing bare
`compile_exp()` wall times. Compilation now incurs its static preparation cost
upfront, and static-map storage still grows exponentially with cluster order.

Raw local measurement: `/tmp/pepsy-compile-review-benchmark.json`. The prior
method was extracted from `/tmp/pepsy-basis-before-compile-review.py`, a snapshot
before this review's edits. No installed dependencies or running jobs changed.
