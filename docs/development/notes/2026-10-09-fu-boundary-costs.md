# 2026-10-09 — FU boundary modes, CuPy and operation counts

Audit of the current two-site full-update path, following the
[scalar-objective implementation](2026-10-09-cached-pair-tn.md). The initial
audit made no production changes. The subsequent user-requested policy change
makes direct compressors sequential by default in `PepsOptimizer`; standalone
boundary helpers retain their defaults. The environment and upstream inspection
from that active task were reused.

## Boundary modes and GPU execution

`PepsOptimizer(..., mode='full-update', fit_mode=...)` forwards the selected
mode to boundary compression, including normalization boundaries:

- `dmrg` (default), `eff`, and `one-site`: one-site variational FIT.
- `dmrg2`: two-site warm-up followed by one-site refinement when the sweep
  budget allows; `two-site` keeps all sweeps two-site.
- `direct`: Quimb site contraction, canonicalization, then SVD compression.
  Other exposed Quimb compression methods have their own behavior and costs;
  they were not all tested in this audit.

Expanded `tests/test_peps_full_update_cupy.py` to exercise `direct`, `eff`,
`dmrg`, and `dmrg2` with both complex64/128 and both reduced ALS solvers
(`quimb`, `qr`). These real-GPU tests cover local QR/SVD initialization,
boundary construction, reduced norm/PSD construction, ALS, reconstruction,
and normalization. They reject non-scalar CuPy transfers through both
`autoray.to_numpy` and `cupy.asnumpy`, check output dtype/device, and compare
reported fidelity to the exact small-state overlap. Existing tests additionally
compare CuPy with Torch, exercise adaptive boundaries and cache mutation.

Full-tensor L-BFGS accepts CuPy input/output, but its autodiff computation uses
Torch on the same CUDA device. SciPy still exchanges parameter/gradient vectors
with host memory. Reduced ALS uses native CuPy numerical arrays; scalar
validation, convergence decisions, and cache checks can synchronize the GPU.
These are bounded correctness checks, not GPU speed benchmarks.

## Costs for a bulk square-lattice pair

Let D be the PEPS virtual dimension, chi the *actual* boundary rank, d the
fixed physical dimension, and r <= dD the reduced exterior rank. Assume
dense arrays, enough lattice width to attain those ranks, and suitable
contraction paths. A cap larger than the realized rank does not demonstrate
asymptotic scaling. Costs below are per local contraction/decomposition;
boundary sweeps, ALS iterations, objective evaluations, and gates multiply
the corresponding costs.

| Step | Leading work, fixed d | With chi proportional to D squared |
| --- | --- | --- |
| Local QR/LQ reduction and reconstruction | D^5 | D^5 |
| Reduced local SVD warm start / balancing | D^3 | D^3 |
| One-site boundary FIT projection or strip prefix/suffix advance | chi^3 D^4 + chi^2 D^6 | D^10 |
| Boundary one-site canonical QR | chi^3 D^2 | D^8 |
| Reduced pair norm construction | chi^3 D^4 + chi^2 D^6 | D^10 |
| Reduced norm PSD eigendecomposition / dense ALS solve | D^6 | D^6 |
| Full-pair scalar TN norm, overlap and reverse-mode gradient | chi^3 D^4 + chi^2 D^6 | D^10 |
| Two-site boundary projection and dense SVD | chi^3 D^6 + chi^2 D^6 | D^12 |
| Standard joint `direct` boundary canonicalization | chi^3 D^8 | D^14 |
| Sequential bra/ket `direct` canonicalization, same grouped sites | chi^3 D^5 | D^11 |

The final row is an important distinction: joint direct contraction produces
an untruncated boundary with bond R = chi D^2 and local dimension q = D^2.
Its initial canonicalization can QR a `(Rq) x R` bulk matrix, costing
q R^3 = chi^3 D^8. It can also store chi^2 D^6 elements in one expanded
boundary tensor. Small or short boundaries have smaller ranks and can be
fast despite this worst-case bulk scaling. These counts describe the standard
joint/canonicalizing direct route, not sequential-layer, randomized, or
custom compression variants.

Sequential bra/ket absorption is already available with
`fit_mode='direct', fit_layer_mode='sequential', layer_tags=('BRA', 'KET')`.
It compresses after each layer; reversing the tags gives ket-first absorption.
It is distinct from `fit_layer_mode='joint'`. After the user's follow-up,
sequential absorption is the `PepsOptimizer` default for direct compressors;
DMRG/FIT still defaults to a joint, uncontracted target. Here the expanded
bond is only R = chi D. However, a grouped boundary site still has local
dimension d D^2 after the first absorption (the remaining incoming leg, the
new outgoing leg, and the physical leg), and D^2 after the second. Standard
direct canonicalization therefore costs O(d chi^3 D^5), or O(D^11) for
fixed d and chi ~ D^2. The D^14 row does **not** describe this route.
Further splitting the grouped boundary site or using another compression
algorithm requires a separate count; it is not implied by sequential layers.

An eight-site bulk target with d=2 was instrumented through the installed
Quimb compressor. At D=3, chi=9, joint absorption QR'd a 729-by-81 matrix,
while sequential bra-first absorption QR'd 486-by-27 and 243-by-27 matrices.
At D=2, chi=4, the corresponding shapes were 64-by-16, 64-by-8, and 32-by-8.
These match qR-by-R and the powers above; they are shape checks, not timing
fits. Reproduction: `/tmp/pepsy_sequential_shapes.py` and its `.log`.

Two-site FIT produces a wavefunction with chi^2 D^4 entries and explicitly
SVDs a `(chi D^2) x (chi D^2)` matrix before truncation. Capping the retained
singular values does not make this dense SVD rank-truncated computationally.
One-site refinement after warm-up avoids repeating that particular step.
Instrumentation of the production FIT block split confirmed 16-by-16 SVDs
at D=2, chi=4 and 81-by-81 SVDs at D=3, chi=9, despite retaining only chi
singular values (`/tmp/pepsy_dmrg2_shapes.py` and its `.log`).

Thus one-site DMRG has D^8 canonicalization and D^10 leading contractions
at chi ~ D^2. Two-site DMRG is not generally limited to that range in the
current grouped-site implementation. Sequential-layer mode is currently
restricted to direct Quimb compressors, not the variational FIT modes.

The reduced norm is the dense n-by-n matrix for the two reduced exterior
legs, with n = r_A r_B <= d^2 D^2. After construction the implementation
Hermitianizes it, N_H = (N + N.H)/2, at O(n^2) work; diagonalizes N_H and
clips negative eigenvalues to zero; and reconstructs its PSD metric through
a square-root factor. Diagonalization and reconstruction cost O(n^3), with
O(n^2) matrix storage. Thus conditioning costs O(d^6 D^6) work and
O(d^4 D^4) storage, independently of boundary chi once the matrix exists.
It enforces positive **semidefiniteness**, not strict positive definiteness,
and remains enabled when optional environment gauge fixing is off.

For one-site FIT and the full scalar-TN path, representative largest
contraction intermediates have chi^2 D^4 elements: **D^8 memory**, alongside
**D^10 work** when chi ~ D^2. This is not total process memory: saved autodiff
intermediates, cached cuts, CuPy mutation snapshots, solver workspace, and
L-BFGS history add storage. A boundary tensor/prefix has chi^2 D^2 elements;
the reduced norm has r^4 elements. CuPy snapshot validation is linear in the
arrays compared, and adds synchronization/scan overhead independently of the
contraction work.

Caching avoids repeating unchanged contractions and amortizes sweep work; it
does not change the D/chi powers of a necessary rebuild. Grouped strip
traversal can keep the main contraction count proportional to lattice size
per pass. Interleaved gates, extra norm/acceptance evaluations and dependent
cut invalidations add work. No universal O(N D^8) end-to-end claim follows.
Heuristic Cotengra planning is not a proof that every run attains an optimal
path. Path search and Python/cache bookkeeping are excluded from these
dense arithmetic counts.

## Dimension-only checks and literature

Cotengra `optimal` searches on explicit index graphs, without tensor
allocation, were run at `(D,chi) = (4,16), (8,128), (16,512)`:

- A one-site boundary projection and a prefix advance each gave
  `2 chi^3 D^4 + 2 d chi^2 D^6` multiplication units.
- A two-site boundary projection gave
  `2 chi^3 D^4 + 4 d chi^2 D^6 + chi^3 D^6`, before its SVD.
- A six-tensor ring around the reduced pair gave
  `2 chi^3 D^4 + 2 chi^2 D^5 r + 2 chi^2 D^4 r^2 +
  2 chi^3 D^2 r^2 + chi^2 r^4`.
- The full-pair norm reproduced
  `4 chi^3 D^4 + 4 d chi^2 D^6 + d D^4`. Some environment-only
  contractions are constant-folded per solve; variable-dependent terms
  retain the leading powers. Explicit fixed-size gates do not change them.

These model graphs match the bulk index structure, not a timing profile of
every dynamically selected production path. Temporary reproductions:
`/tmp/pepsy_boundary_cost_audit.py`, `/tmp/pepsy_pair_scaling.py`, and
`/tmp/pepsy_boundary_cost_audit.json`.

The standard boundary cost and reduced-pair costs agree with
[Lubasch et al., sections III.1 and III.2.1](https://arxiv.org/html/1405.3259).
The user's [1711.07584, page 5](https://arxiv.org/pdf/1711.07584) studies a
different multi-tensor update, quoting O(D^6 chi^3) norm construction and
O(D^12) ALS; those are not the present two-site reduced ALS costs.

## Validation

The CuPy full-update file plus the full-pair CuPy/Torch L-BFGS comparison:
**28 passed**, one existing Quimb SVD-default warning, 9.83 seconds.
Log: `/tmp/pepsy-boundary-modes-cupy.log`. No production changes or full-suite
rerun for this audit. Broader previous checks remain recorded separately.

Follow-up for sequential absorption: both bra-first and ket-first direct
compression were added to the CuPy FU regression, across both precisions
and both reduced ALS solvers. The expanded file passed **35 tests**, one
existing SVD-default warning, 8.69 seconds. Log:
`/tmp/pepsy-sequential-cupy.log`. This check preceded the policy change below.

## Implemented follow-up — PepsOptimizer layer defaults

At the user's request, `PepsOptimizer` now resolves an omitted layer policy
to sequential BRA then KET for all registered direct Quimb boundary modes,
including zipup, SRC and SDC variants. Variational FIT modes resolve to joint.
An explicit policy or layer order remains authoritative. The resolved policy
is shared by normalization, infidelity evaluation, full-update environments,
and delegated sweeps. This changes the optimizer's defaults, not standalone
`CompBdy` or boundary-helper defaults. It retains SRC boundary initialization
(`fit_init_strategy='guess-src'`) before joint one-site DMRG fitting.
The automatic layer policy is resolved for the actual metric route, so exact
normalization/overlap overrides and the separate native Quimb MPS engine keep
their existing semantics. Explicit incompatible layer policies still fail
validation rather than being silently changed.

Tests inspect actual direct/zipup compressor inputs, checking alternating
single BRA and KET layers. For DMRG and DMRG2 they check that FIT retains
the original separate layer arrays: FIT retags/reindexes the target but
does not materialize a combined row tensor. GPU tests cover zipup as well
as direct compression under the new defaults.

Final policy validation: 233 passing metric/driver/safeguard cases plus 130
passing FU/cache/CuPy/pair/warmstart cases (the latter also rechecks the one
initial-loss case fixed after the first selection). All 363 selected cases
are covered with no outstanding failures. Ruff and diff checks pass. The
full suite was not rerun for this optimizer policy change.
