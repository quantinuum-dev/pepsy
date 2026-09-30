# 2026-09-29 — Exact algebraic edge reduction for Pauli PEPOs

Implemented in the local working tree on `develop`, baseline `c1ff0f8`.
This follows [identical-slice sharing](2026-09-29-symbolic-pepo-channels.md).
The new opt-in mode is `prepare_cluster_channels(...,
preparation="algebraic", max_bond=None)`. The default remains `"reference"`.
Pepsy owns all algorithms; Gaugy's existing adapter forwards the new value.

## Exactness and implementation

For every selected connected cluster, multiply the declared Pauli words
modulo phase to obtain a closed span. The inventory includes every contained
term in every ordered factor, independently of its coefficient or reference
value. Each local generator is in this algebra, hence so are its exponential,
the ordered product, and the products/subtractions defining connected
residuals. This proves an admissible residual span without evaluating or
dividing by live parameters. It does not commute or reorder factors.

Use real Weyl words `X^x Z^z`, with phases such as `Y=iXZ` absorbed in live
coefficients. Their matrix entries are 0 and ±1. A formal independent atom
per allowed word makes edge slices linear expressions with rational
coefficients. Full spans use independent matrix entries to avoid expanding
an already unrestricted basis.

For one endpoint, stack each channel's **entire** slice: physical indices,
all other virtual indices and formal atoms. Sparse rational elimination
selects original independent slices and proves a constant transfer matrix
`W_old = W_selected @ T.T`. Multiply `T` into the other endpoint, using the
ordinary transpose appropriate to bilinear tensor contraction. Protect the
singleton rail. Sweep both endpoints until no edge dimension decreases.
This is a local tensor identity on any network, including branches and
loops. It is stronger than identical-slice sharing, but not a globally
minimal PEPO construction.

Graph reduction precedes routing; routed square edges receive the same
elimination. Reference/tangent samples evaluate a reusable linear program.
Numeric expanded PEPO reference builders are not called. Fixed maps are
folded into the existing fused assembly recipe. Constant rational maps are
converted to floating point, so numerical operators retain normal roundoff.
Oversized allocations and nonfinite transfer maps raise.

Replay includes a fixed projection onto each proved Pauli span. Its Walsh
transforms use reshaping, permutations and fixed 2-by-2 matrix contractions,
without SVD, QR, rank decisions or parameter division. This projection is
the identity on the declared family, including its derivatives at zero.
It also explicitly defines `bind_assembler` on arbitrary input matrices
outside that family. `pack_residuals` keeps the original dense residuals;
local matrix exponentials are still evaluated by the existing source.

Implementation files: `_cluster_channel_pauli.py`,
`_cluster_channel_linear.py`, `_cluster_channel_pepo.py`, integration in
`_cluster_channel_assembly.py` and `cluster_channels.py`. No dependency or
installed-library changes. The same-task upstream audit is reused; the
environment is unchanged. An Autoray keyword-translation wrapper failed
Torch full-graph capture in the first probe; replacing that stack operation
with fixed contractions resolved the failure without an upstream shim.

The API requires fixed finite NumPy local matrices exactly proportional to
I/X/Y/Z, without string operators. It rejects other local dimensions and
non-Pauli or live local-operator arrays. Scalar parameters remain live.
Public evaluation checks the declared operator inventory; compiled kernels
assume that inventory is fixed. Native/fermionic PEPOs are outside scope.

## Measurements

All 72 rows are in [JSONL](2026-09-29-algebraic-pepo-channels.jsonl), generated
with the shared Python 3.12 environment and one BLAS/OpenMP thread:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  python examples/symbolic_pepo_channels_benchmark.py --repeats 3 \
  --preparations reference symbolic algebraic
```

Four-site square controls cover branches, loops, a gapped higher-body term,
and independent crossing wires on periodic geometry. Times are medians after
warming residual geometry, with no concurrent test run. The table uses six
reference samples and **full channels**. Memory is preparation tracemalloc
peak in decimal MB, not process RSS or device/backward memory. Replay timing
includes fresh Torch residuals, PEPO assembly, dense contraction, scalar
objective and backward; it is not a compiled throughput benchmark.

| Case | Preparation | Exact output bonds | Site entries | Prep ms | Prep MB | Forward/backward ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Branch | reference | 289,17,1,17 | 39440 | 34.03 | 4.246 | 25.10 |
| Branch | symbolic | 25,5,1,5 | 1040 | 29.01 | 0.303 | 20.26 |
| Branch | algebraic | 25,5,1,5 | 1040 | 105.02 | 1.847 | 26.71 |
| Loop | reference | 13,13,13,13 | 2704 | 24.19 | 0.191 | 23.98 |
| Loop | symbolic | 9,9,9,9 | 1296 | 27.84 | 0.210 | 17.65 |
| Loop | algebraic | 7,7,7,7 | 784 | 39.25 | 0.489 | 24.25 |
| Higher body | reference | 81,1,1,45 | 15088 | 16.65 | 0.533 | 12.85 |
| Higher body | symbolic | 25,1,1,5 | 624 | 14.14 | 0.177 | 11.37 |
| Higher body | algebraic | 13,1,1,5 | 336 | 25.00 | 0.211 | 12.87 |
| Crossing | reference | 5,1,5,1,1,5,1,5 | 1040 | 6.67 | 0.091 | 6.60 |
| Crossing | symbolic | 5,1,5,1,1,5,1,5 | 1040 | 7.01 | 0.119 | 7.00 |
| Crossing | algebraic | 5,1,5,1,1,5,1,5 | 1040 | 11.19 | 0.126 | 9.67 |

All full-channel operator relative errors are at most `1.85e-16` and gradient
relative errors at most `2.61e-15`, against reference preparation of the
**same selected cluster target**. These do not bound the cluster cutoff's
error against the full-lattice ordered exponential product.

The mode gives smaller exact tensors for loop and higher-body cases, but
preparation is slower than simple sharing in these controls. It gives no
further chi reduction for branch/crossing cases. Rational symbolic work and
the fixed residual projection have costs; use explicit measurements when
choosing a mode. There is no general speedup claim.

Further `max_bond` caps remain approximate, reference-dependent QR choices
outside autodiff. For cap 4 and six references:

| Case | reference operator / gradient relative error | symbolic operator / gradient relative error | algebraic operator / gradient relative error |
| --- | --- | --- | --- |
| Loop | .01810 / .07737 | .04900 / .63063 | .001580 / .22720 |
| Higher body | .02192 / .11794 | .02192 / .11794 | .000527 / .000314 |

Smaller operator error does not imply smaller objective-gradient error.
Exact gauge changes can improve or worsen tight caps; always check both.

## Scope and unfinished work

- No SVD or QR is used with full algebraic channels, including preparation.
  A smaller optional numerical cap still uses the existing reference QR.
- Constant linear relations are certified locally. There is no global
  symbolic minimization, environment-optimal truncation or small-chi guarantee.
- Pauli closure, local fixed factors and remaining routed blocks are still
  enumerated once. The closure can fill the whole operator space. Rational
  elimination can suffer fill-in and large constants. Allocation guards are
  not whole-process memory or rational bit-complexity bounds.
- Dense local exponentials and connected subtraction remain; evaluating
  directly in a reduced operator algebra is separate future work.
- Torch `aot_eager` full-graph capture verifies the assembly boundary, not
  optimized runtime or complete Torch exponential-evaluator capture.
- MPO frontier construction and the separate numerical NumPy
  `delinearize_mpo` routine are unchanged. The latter's QR-based current-value
  relations are not used as parameter-family proofs here.

The motivating distinction from numerical delinearisation is consistent
with [Hubig et al., Appendix C](https://arxiv.org/abs/1611.02498): this path
proves constant formal identities instead of detecting rank at one parameter
value. Connectivity is handled explicitly as required for
[cluster tensor-network operators](https://arxiv.org/abs/2112.01507); no
one-dimensional continuation rule is transplanted onto loops.

## Validation record

The focused algebra/PEPO suite passed **49 tests**, including independent
ordered-exponential references, arbitrary in-algebra residuals and adjoints,
zero coefficients, independent bindings, single precision, Torch CPU/CUDA,
JAX trace JIT and Torch full-graph assembly capture/backward. Full-channel
tests forbid NumPy/SciPy/Torch SVD/QR calls and the expanded numerical PEPO
builder. A synthetic nonidentical-column test checks an exact 1/3 transfer
and a protected duplicate rail.

The broader Pepsy selection passed **274 tests**, two existing compatibility
warnings, 88.90 s. Gaugy passed **167 tests**, six existing Quimb warnings,
45.14 s, including actual-PEPO trace gradients for all three preparations.
After concurrent MPO edits to shared helpers, **134 overlapping channel
tests passed again**, 60.56 s. Counts are not additive. Ruff, touched Gaugy
Pyflakes, local documentation links and whitespace checks pass. Full package
suites were not run. Exact commands, scope, logs and publication status are
in the [handoff](../../../history/2026-09-29-algebraic-pepo-channels.md).
