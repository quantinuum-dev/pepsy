# 2026-09-29 — Operator-aware cluster MPO automaton

## Scope and implementation

The user requested an efficient explicit automaton path for the cluster MPO
and comparison in the single/joint 1D notebooks. Added opt-in
`prepare_cluster_channels(..., preparation="automaton")` for dense qubit
Pauli product terms. Existing defaults and the general `MPOAutomaton` class
are unchanged. The new mode compiles the **selected cluster expansion C_p**;
finite p still differs from the full exponential/product.

**Adopt:** weighted-state/path representation, as described by
[Crosswhite and Bacon](https://arxiv.org/abs/0708.1221), and the state-transfer
principle of [Hubig et al.](https://arxiv.org/abs/1611.02498). Reuse the current
frontier planner instead of enumerating complete collections. Reuse the
existing Pauli closure and exact rational slice helpers shared with PEPOs.
Each residual's declared Pauli span is closed under local products, so it
contains ordered exponentials and connected subtraction. Symbolic atom
identities hold for the whole declared family, including zero coefficients.

**Implemented experimental path:** prefix/suffix sweeps select independent
transition slices with rational arithmetic and transfer the other slices to
the opposite endpoint. The singleton rail remains separate. Frozen maps enter
the existing fused local assembler; a fixed Walsh projection is the identity
on the declared Pauli family. Arbitrary supplied residuals outside that span
are projected. Full channels require no SVD/QR in preparation or replay;
smaller caps retain reference QR approximation. Operator inventory changes
require a new plan. Native charge sectors use the existing frontier mode.

**Defer:** global minimal weighted automata, parameter-dependent polynomial
state reduction and broader operator-algebra certificates. Constant local
state relations do not guarantee minimum chi or improved runtime. Local
dense residuals, sparse frontier preparation and its maps remain; memory and
state budgets are guards, not asymptotic memory guarantees.

## Measurements

Four-site nearest-neighbor XX, YY or ZZ commuting generators, complete
cluster order p=4: frontier bonds **(5,25,9)** become **(3,9,5)** in automaton
mode. Independent full matrix exponentials agree at zero and nonzero
couplings. This demonstrates reduction beyond identical-path sharing,
including negative Pauli signs; it does not establish global minimality.

In the six-site notebooks (J1=1, J2=0.5, h=1, t=0.06), the automaton and
frontier profiles **coincide** for the checked single/joint models and p=1..5.
At OBC p=5 their maximum bond is 209, compared with raw fixed 649. Numerical
delinearisation/SVD give 50/35 for the single target and 64/64 for the joint
target. The notebook plots and printed messages explicitly show the overlap.
No extra bond reduction is claimed for these NN + NNN models. PBC p=5 has
maximum bond 329 for both automaton and frontier, versus raw fixed 1737;
PBC describes the interaction graph, with open MPO storage.

Both complete notebooks passed OBC and PBC execution independently. Maximum
relative automaton error against independent explicit C_p matrices: single
OBC 2.38e-14, joint OBC 2.86e-14, single PBC 5.92e-14, joint PBC 6.17e-14.
Fresh OBC outputs are saved; the PBC runs remain temporary validation copies.

A forty-site disjoint-crossing regression has four active cluster sets at a
cut and more than one million possible nonempty complete collections. The
automaton prepares with collection_budget=1 while complete-collection/source
assembly methods are patched to raise. Its actual-MPO normalized trace agrees
with an independent analytical product. No global dense operator is formed.

## Validation

- Affected Pepsy selection: **217 passed, 10 skipped** (CUDA hidden), two
  existing deprecation warnings. Files: `test_cluster_channel_automaton`,
  `test_cluster_channel_algebra`, `test_cluster_channel_frontier`,
  `test_cluster_channel_structure`, `test_cluster_channels`,
  `test_cluster_channel_assembly`, `test_cluster_channel_pepo`,
  `test_mpo_delinearize`, `test_public_api`, `test_package_layout`.
- The new file contributes 12 tests: independent matrix exponentials,
  projected arbitrary residuals/adjoints, coupling and time derivatives at
  zero, independent coefficient slots, complex64, Torch full-graph assembly,
  JAX trace JIT, family/reference independence, guards and forty-site scaling.
  No-decomposition tests patch NumPy/SciPy/Torch SVD and QR to raise.
- Downstream Gaugy channel/binding/materialization checks: **31 passed**.
- The first combined Pepsy run had 214 passing cases and two Torch
  `RecompileLimitExceeded` failures in the existing PEPO compilation tests.
  New independent compilation cases had consumed Dynamo's shared eight-entry
  frame cache. The new tests now reset their own compiler cache before and
  after each compilation case; the complete selection above then passed.
  No numerical acceptance tolerance or production compiler limit changed.
- Full notebook and presentation validation is recorded in the
  [handoff](../../../history/2026-09-29-mpo-automaton.md).
- No full-repository suite or optimized compiler throughput measurement.

The active Python 3.12 environment is unchanged: NumPy 2.5.2, SciPy 1.17.1,
Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, Symmray
0.4.1.dev7+g83fb22865. Rechecked preparation and slice-elimination signatures.
Reused the same-task [upstream compatibility audit](2026-09-29-mpo-delinearisation.md#validation-and-environment);
no installed library or shared environment was modified. Torch tests use
`aot_eager` full-graph capture and public `torch.compiler.reset` for isolation.
