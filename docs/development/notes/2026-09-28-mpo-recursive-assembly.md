# 2026-09-28 — Complete graph MPO assembly through shared subproblems

## Status and scope

Implemented in the working tree on `develop`, baseline `4e398e4`; nothing
staged, committed or published. The user requested adding MPO contributions
and compressing while retaining the complete chosen spatial cluster expansion.
This change is confined to the graph MPO assembly/API subsystem. Earlier
PEPO/spatial reuse and independent sampler edits remain present.

The owning guide is [MPO cluster expansion](../../api/operators/mpo_cluster.md#complete-recursive-assembly-add-mpo-branches-and-compress).

## Construction

`assembly="recursive"` compiles a remaining-site DAG. At the first available
site, either insert its singleton target or select a connected residual that
contains that site and whose support remains available. Every compatible
collection has exactly one branch sequence. Subproblems are shared by their
remaining-site bitmask; integer order is a nonrecursive topological order.
The count of nonempty collections is evaluated on the same DAG with Python
integers. Compilation caches structure only; every call regenerates backend
values and gradients. Numeric subproblem references are released after last
use; autodiff can retain its own backward buffers.

Each branch multiplies a pure residual MPO into its child on disjoint support.
Residual MPO spans include identity gaps, preserving crossing/nested clusters
under a square-lattice snake map. A direct sum adds each branch (or batch),
followed by optional canonicalized compression. No inverse singleton operator
or global dense operator is needed. `collection_budget` is unused in this
mode; `assembly_state_budget` caps distinct structural subproblems and raises
without dropping contributions. It counts the empty base as well.

`assembly_chi=None` skips intermediate compression. Otherwise the default
batch contains one addition; `None` also means one, consistently with streaming.
Local residual `cutoff`/`max_bond` and optional final `chi` remain separate.
Charge/fermionic recursive assembly is rejected explicitly.

## Corrections found during reference checks

- Existing streaming on a direct graph plan added only individual residual
  paths, losing products of separated disjoint residuals. It now uses the
  complete recurrence on that path. Explicit exact/bounded collection plans
  still use the existing streaming collection-path assembler.
- Truncating raw direct-sum/core-product gauges can lose large operator terms.
  Both assembly paths now prepare the opposite orthonormal environment before
  truncation. The existing semantic left TT-SVD is reused with reversed core
  ordering to prepare a right-canonical environment and handle right-directed
  output. The existing semantic right sweep absorbs singular values into its
  site cores, so it cannot itself supply this canonical environment. No global
  semantic compression behavior or installed dependency was changed.
- Stable `api_info.truncated` now includes assembly bond reductions and graph
  collection omissions as well as local residual truncation. The assembly
  flag conservatively records dimension reduction, including redundant zero
  channels; adaptive discarded weights are local sweep diagnostics, not a
  global accumulated error bound.

## Measurements

Single CPU-only runs, one BLAS/OpenMP thread; timings are scoped observations,
not a benchmark guarantee. Open square lattice, snake order, uniform onsite
X and nearest-neighbour ZZ, shared coefficient identities.

| Shape / p | Nonempty collections | DAG states | Compile seconds | Numerical evaluation |
| --- | ---: | ---: | ---: | --- |
| 3x3 / 2 | 130 | 27 | 0.014 | chi=8: 0.096 s, 29 compressions, 8 peak cached states, peak bond 40 |
| 5x6 / 2 | 65,805,402 | 703 | 0.011 | chi=4: 8.31 s, 941 compressions, 68 peak cached states, peak bond 20 |
| 5x6 / 4 | 249,479,463,512 | 33,514 | 0.326 | Compile only; numerical runtime/accuracy unverified |

Evaluations used step=0.03, h=0.2, J=0.4. The 3x3 result's relative Frobenius
error against independent explicit complete collection assembly was 4.08e-7.
No dense 30-site operator was allocated; the 5x6 numerical accuracy was not
measured. A 40,000-state guard was used for these runs. The default 4,096 guard
requires an explicit increase for the 5x6 p=4 plan.

A separate 2x3 example (onsite X and three disjoint vertical ZZ bonds) has
absolute Frobenius errors approximately 2.00e-3, 2.00e-6, 2.77e-8 and 8.97e-14
at chi=4,8,16,64. No intermediate compression gives 1.22e-15. This is scoped
convergence evidence, not a general monotonic convergence guarantee.

## Upstream audit / classification

**Adopt:** existing Pepsy semantic left TT-SVD, path addition and Autoray core
operations. No new dependency, registration, compatibility shim or vendored
implementation. Installed versions: Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124.
Inspected `compress_fixed_rank(max_bond, *, form, return_report)`,
`compress_adaptive(max_bond, *, cutoff, cutoff_mode, form, return_report)` and
`_add_path_cores_batch(paths)`; checked NumPy/Torch Autoray reshape, transpose,
tensordot and SVD dispatch. Existing scoped Torch SVD policy is preserved.

Consulted [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The [Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an internal error; official source and installed code were used.

## Validation and limits

Final focused gate recorded in the session handoff. Initial checks exposed
and corrected gauge preparation; the six-site exact reconstruction check uses
chi=64 (chi=8 deliberately truncates). A separate low-chi error regression
protects canonicalization in both directions and assembly modes. Small exact
checks cover disjoint, crossing, nested and cyclic clusters, ordered products,
explicit enumeration equivalence, independent dense exponentials, state-budget
failures, compile-cache reuse, coefficient/time gradients and singleton graphs.

The earlier 5,199-pass full-suite result predates these changes. These changes
have no new full-suite claim. All new numerical checks are CPU-only; active
user GPU workloads were left alone. Native sectors/fermions and large-p
numerical performance are unverified/unsupported as documented. The DAG can
still grow rapidly with graph width, and its state cap is not a byte bound.
