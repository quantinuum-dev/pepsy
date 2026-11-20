---
name: mpo-dmt
description: 'Explain, implement, review, and profile Pepsy density-matrix truncation (DMT), ordinary or Pauli MPO replay, larger protected regions, and verification controls. Use for local-density preservation, DMT accuracy, numerical stability, and optimality questions; compose tensor-fitting for explicitly requested variational fitting.'
---

# MPO density-matrix truncation

Maintain DMT's local-density guarantees while distinguishing the full
constrained approximation problem from the residual SVD actually implemented.
Follow repository scope and environment instructions in
[`AGENTS.md`](../../../AGENTS.md).

For API changes, verification costs, larger windows, backend failures or
benchmarks, read [operations and validation](references/operations.md).
It distinguishes fast replay checks, full sweep verification and actual
trajectory accuracy. Use the mathematical reference below when changing
the protected construction itself.

## Mathematical contract

Read [the mathematics reference](references/mathematics.md) for derivations,
paper comparisons, dimension questions, or changes to the splitting rule.
Use its notation consistently:

$$
M\in\mathbb C^{(4\chi)\times(4\chi)},\quad
M'\in\mathbb C^{\chi\times\chi},\quad
{\color{purple}A=U_cM'V_c^\dagger},\quad
U_c,V_c\in\mathbb C^{(4\chi)\times\chi}.
$$

Here $M$ is the post-gate canonical bond matrix; $A$ is the compressed
operator in the **original** virtual coordinates. The compressed MPO uses
the small $M'$ after absorbing $U_c,V_c^\dagger$. The $4\chi$ dimension is
the maximal adjacent-qubit example, not a promise that every gate saturates it.
Use the actual canonical support when smaller. Do not subtract $M'-M$.

$$
\boxed{\min_{\operatorname{rank}A\le\chi}\|A-M\|_F^2
\quad\text{s.t.}\quad
{\color{blue}F_L(A-M)=0},\quad
{\color{blue}(A-M)F_R^T=0}.}
$$

- A constraint-satisfying result is **feasible**, not automatically optimal.
  Canonical outer isometries make this matrix objective the physical
  Hilbert–Schmidt error at one cut. Do not assert it for arbitrary gauges.
- In computational coordinates, $F_L$ has ket and bra legs on the retained
  boundary site and one virtual leg: $2\times2\times4\chi$, or a fused
  $4\times4\chi$ matrix. $F_R$ has the corresponding three legs. Partial
  tracing is linear: use $F_R^T$, and the protected right range is
  $\operatorname{range}F_R^T$, not $\operatorname{range}F_R^\dagger$.
- Default 1D protection preserves $[b,N-1]$ and $[0,b+1]$ at cut $b|b+1$;
  these contain every contiguous three-site window. Selected regions refer
  to the current **pre-compression target**, including any preceding gate.
  Explicit inventories can require more than four directions per side;
  retain trace directions on both sides and use the actual protected ranks.
- Default density DMT truncates the **free connected residual**. For protected
  ranks $p,q$, the conservative free-rank budget is $\chi-p-q$. Its SVD is
  optimal at the retained residual rank, not over all feasible total-rank
  matrices in the box. Extra cutoffs may retain fewer residual directions.
  A conservative budget rejection does not prove mathematical infeasibility.
- Optional `truncation="optimized"` keeps a feasible paper seed, tests a
  compatible-border Schur SVD, then uses guarded constrained alternating
  factor updates. Accept only feasible one-cut error improvements. General
  convergence is local; there is no global MPO optimum certificate.
- `region_selection="min_cover"` minimizes physical-string count across
  crossing regions. Each product observable must have one selected endpoint;
  retain trace on both sides. This does not minimize numerical virtual rank.
- Ordinary matrix units and normalized Pauli operators are equivalent
  orthonormal coordinates. Raw Pauli coefficients have different trace and
  norm factors. Keep those conventions and diagnostic units explicit.

## Find the owning operation

| Request | Owning API / implementation | Closest validation |
| --- | --- | --- |
| Ordinary ket/bra MPO compression | [API](../../../docs/api/operators/mpo_dmt.md), [adapter](../../../src/pepsy/operators/mpo_dmt.py) | [ordinary MPO tests](../../../tests/test_mpo_dmt.py) |
| Pauli MPO compression or shared protected split | [API](../../../docs/api/operators/pauli_mpo.md), [shared kernel](../../../src/pepsy/operators/_pauli_dmt.py) | [block-reference tests](../../../tests/test_pauli_dmt.py), [cutoffs](../../../tests/test_pauli_dmt_cutoff.py) |
| Persistent adjacent-gate evolution | [API](../../../docs/api/optimizers/pauli_dmt.md), [driver](../../../src/pepsy/optimizers/pauli_dmt.py) | [workspace tests](../../../tests/test_pauli_dmt_optimizations.py) |
| Ordinary MPO gate-stream replay | [MPO optimizer API](../../../docs/api/optimizers/mpo.md) | [replay tests](../../../tests/test_mpo_dmt_replay.py) |
| Mapped 2D regions / larger squares | [region contract](../../../docs/api/optimizers/pauli_dmt.md#2d-regions-and-physically-local-gates) | [lattice tests](../../../tests/test_pauli_dmt_lattice.py), [large cases](../../../tests/test_pauli_dmt_large.py) |

The ordinary adapter and Pauli APIs share `_split_scaled`. Identify its
initial exact support SVD, the lossy residual SVD, and final factorization
separately. Reuse the kernel instead of copying it into replay drivers.
Check current source and callers when API defaults or historical notes differ.

## Preserve implementation invariants

- Keep `variant="density"` (connected baseline, resolved nonzero trace)
  separate from `variant="operator"` (raw free block, including traceless
  operators). Neither generally guarantees positivity or global Hermiticity.
- Retain protected data independently of the physics cutoff. Apply cutoff
  policies only to the free spectrum. Preserve backend, dtype, scale/exponent,
  tags, index order, centre metadata and transactional writes. Keep numerical
  feasibility guards; select optional verification through the public controls.
- Keep physical region inventories explicit for mapped 2D systems. They
  replace default chain protection unless both are supplied. This is an MPO
  construction, not a proof of canonical isometries for a loopy PEPO.
- Compare preservation with each split's current target in independent
  validation. Runtime replay defaults to bond/trace checks; full-region
  scans are opt-in. Strict standalone compression checks the original and
  final windows once, avoiding redundant full scans after every bond.
  A sweep's error is not generally the sum of squared one-bond tails.
- The current adaptive compressor uses host decisions. Do not describe it
  as fully JIT compiled or smoothly differentiable through all rank changes.
  Follow the public guide's existing gradient/precision policies when relevant.

## Validation and explanations

For mathematical explanations, start with the six-site cut $012|345$:
trace $01$ to define $F_L$, substitute it into $\Delta\rho_{2345}$, then
mirror the derivation for $F_R$ and $\Delta\rho_{0123}$. Show the four-window
cover and distinguish matrix **shape** from **rank** before discussing SVD.
Keep the local-preservation proof separate from any optimum claim.

The [deterministic mathematical probe](scripts/check_math.py) checks complex
trace contractions, embedded versus compressed dimensions, basis conversion,
and the gap between the full objective and conservative residual truncation:

```bash
python .github/skills/mpo-dmt/scripts/check_math.py
```

Add `--check-kernel` to exercise the conservative-budget counterexample
against the current Pepsy installation; the probe prints its source path.

For implementation changes, run the closest suites above after the repository's
required environment and upstream checks. Use independent dense partial traces,
actual lossy truncation, bond caps, and physical error units. Add complex-valued
cases that distinguish a transpose from an adjoint. Do not infer global
optimality, positivity, or full-suite success from preservation tests.

Wider protection and larger chi need not improve trajectory accuracy
monotonically. Keep the protected budget, chosen truncation and unphysical
outliers visible. Energy agreement alone does not validate local dynamics.

Compose [tensor-fitting](../tensor-fitting/SKILL.md) for changes to FIT-based
variational sweeps. The optional direct bond refiner lives with DMT and uses
hard linear constraints, not a finite marginal penalty. Global constrained
MPO fitting remains separate from its local alternating matrix solve.
