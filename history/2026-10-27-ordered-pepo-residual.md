# 2026-10-27 — Ordered cluster PEPO residual

- Scope: Construct an OBC/PBC PEPO for an ordered exponential product minus identity, with a chosen spatial cluster size and PEPO bond cap, for Gaugy's gauged-residual insertion work.
- Branch / baseline commit: `develop` at `310361e`.
- Commit status: locally committed with the same-session Gaugy trace companion;
  neither repository was pushed.

## What changed

- Added `residual_exp` to `PEPOClusterProductExpansion` and its compiled evaluator. It reuses the existing joint connected-cluster product, subtracts an identity PEPO, then optionally applies Quimb SVD compression to bond cap `chi`.
- Fused parallel virtual bonds before subtraction, which permits the periodic 2×2 lattice as well as ordinary OBC/PBC square lattices.
- Documented the factor order and independent spatial-cluster, operator-bond and later boundary-contraction cutoffs.

## Validation

- Focused and neighboring PEPO regressions: 100 passed initially. After adding explicit four-factor defect checks for OBC/PBC, the focused file passed all 7 tests. These include independent dense ordered-product-minus-identity checks, a real bond cap, and a Torch parameter derivative.
- `ruff check` on changed Python files passed.

## Limits

- The new method builds one square PEPO residual `Q_j A† - I` when the caller supplies ordered factors. The same-session Gaugy `ClusterEngine.pepo_residual_traces` companion maps fitted V/G and target factors into this builder and measures Z1/Z2/Z3 through gate-wise simple-update transport and boundary contraction.
- Directly replacing each connected-cluster local product by `P_C-I_C` in the existing product assembler is not equivalent: even two disconnected sites need both singleton defects and their product. The current route uses the mature product PEPO assembly and subtracts only its global vacuum contribution before compression.
- Cluster truncation, post-subtraction SVD compression and boundary contraction remain separate approximations. No full-suite or large-lattice accuracy claim follows from the focused checks.
