# Belief propagation for tensor networks

## The idea in one paragraph

Belief propagation (BP) is a message-passing algorithm that approximately
contracts a tensor network by assuming the network is locally tree-like (i.e.
ignoring closed loops). For each edge `e` shared by tensors `a` and `b`, BP
keeps a pair of *messages* — vectors `m_{a→b}` and `m_{b→a}`. Messages are
updated by contracting a tensor with all of its *incoming* messages except the
one on the target edge, and iterated to a fixed point. At the fixed point, the
contraction factorizes into per-tensor scalars whose product is the BP estimate
of the full contraction; the corresponding free energy is the **Bethe free
energy**. BP is exact on trees and a good approximation when loop correlations
are weak.

Two facts make BP attractive for PEPS:
- **BP ≈ simple update** (Alkabetz & Arad 2021): the BP fixed point gives the
  same environments as the "simple update" gauge.
- **BP gauges a TN** (Tindall & Fishman 2023): the fixed-point messages define a
  canonical/Vidal-like gauge that is cheap to compute and a great initializer.

## Making BP controlled: loop / cluster expansions

BP's weakness is that it is *uncontrolled* — you cannot systematically improve it
by growing a bond dimension. The two papers driving this workstream fix that.

**Loop series expansion** (vqks-cr6x, *Phys. Rev. Research* 8, 013245):
- Resolve each edge of dimension `d` into a rank-1 projector `P` onto the
  message subspace plus its complement `Q` (rank `d-1`):  `I = P + Q`.
- Expanding every edge over `{P, Q}` writes the whole network as a sum over
  "configurations". A configuration's **degree** = number of edges set to `Q`
  ("excited").
- **Key simplification:** any configuration with a *dangling* excitation (a
  tensor touched by exactly one excited edge) has weight zero. So only
  excitations that form *closed loops* survive. This is the loop series.
- **Weights factorize:** separated excitations multiply; after normalizing each
  tensor so its BP-vacuum contribution is 1, only *connected* loop clusters
  matter.
- **Exponential suppression:** for spatially-homogeneous networks where BP is
  already decent, the weight of a degree-`k` excitation decays ~`exp(-c·k)`
  (justified via the transfer-matrix spectral gap, App. A). So a few low-degree
  loops recover most of the correction.
- Deliverables in the paper: corrections to the **free-energy density**
  (App. B: single-excitation, self-consistent completion, multi-excitation
  counting), the **transfer matrix**, and the **2-site density matrix**.

**Loop cluster expansion** (arXiv:2510.05647): the same machinery analyzed as a
systematic correction to BP for general quantum many-body problems — ground-state
observables and energies in 2D & 3D, open & periodic boundaries, spin & fermion
systems; contraction error converges ~exponentially with cluster size.

## Why this is worth it

The dominant (genus-1) loop corrections cost ~`O(D^…)` polynomial in the PEPS
bond dimension `D` and are feasible at bond dimensions far beyond where boundary
MPS / CTMRG become impractical, while improving on raw BP by several orders of
magnitude. So BP + low-degree loops is a cheap, controllable contraction route
that complements pepsy's existing boundary-MPS path.

## How it maps onto pepsy

- **Input network:** reuse `pepsy.build_bra_ket(ket, bra?)` to form the closed
  double-layer network with the existing `KET`/`BRA` and `X*/Y*/I*` tags. BP
  operates on that same object.
- **Result type:** return a `BPResult` dataclass mirroring
  `BoundaryContractResult` (`.cost`, `.fidel`-style fields), so downstream code
  treats BP and boundary-MPS uniformly.
- **Validation:** the papers (and pepsy) use large-`chi` boundary-MPS as the
  numerically-exact reference. So every BP/loop test compares against
  `pepsy.contract_boundary` at large `chi`.
- **Placement:** new subpackage (peer to `boundary/`), e.g.
  `pepsy.contraction` / `pepsy.bp` with `bp.py`, `loops.py`, `expansion.py`,
  `environments.py` (see `PLAN.md` §1).

## Failure modes to document

- No / non-unique BP fixed point (common for ground states of local
  Hamiltonians).
- (Near-)degenerate fixed points → loop series does **not** converge; canonical
  example is a GHZ-like PEPS where the "other branch" is a maximal-degree
  excitation with equal weight.
- Square lattices: number of distinct loops grows fast with degree → prune and
  cap cost; reuse `cotengra` for the sub-network contractions.

## References

- Evenbly, Pancotti, Milsted, Gray, Chan — *Loop series expansions for tensor
  networks*, Phys. Rev. Research 8, 013245 (2026), DOI `10.1103/vqks-cr6x`.
- Gray, Park, Evenbly, Pancotti, Kjønstad, Chan — *Tensor Network Loop Cluster
  Expansions for Quantum Many-Body Problems*, arXiv:2510.05647.
- Alkabetz & Arad, Phys. Rev. Research 3, 023073 (2021).
- Tindall & Fishman, SciPost Phys. 15, 222 (2023).
- Chertkov & Chernyak, Phys. Rev. E 73, 065102(R) (2006).
