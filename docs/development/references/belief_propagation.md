# Belief-propagation references (`pepsy.bp`)

Annotated bibliography for the `pepsy.bp` subpackage (BP contraction, loop /
cluster corrections, convergence robustness) and the sibling **tensy** BP-decoder
work. Kept next to the code so the reference trail is not lost. Last updated
2026-07-15.

Legend: **[pepsy]** implemented/wrapped here · **[quimb]** available upstream in
`quimb.tensor.belief_propagation` · **[tensy]** consumed by the tensy decoder
side · **[roots]** foundational / prior art.

---

## 0. Implemented / wrapped in `pepsy.bp`

| API | what | paper |
| --- | --- | --- |
| `one_norm_bp`, `two_norm_bp`, `relay_bp`, `RelayBPResult` | plain D1BP / D2BP plus disordered-memory / relay-BP convergence robustness | Müller et al. 2506.01779 |
| `BPState`, `BPUpdateResult` | cached D1BP messages and a bounded/local warm-update light cone for value-only topology-preserving changes | Midha et al. 2604.21919 (algorithmic-locality motivation; Pepsy scope documented in API) |
| `gauge_all`, `simple_update_core_and_gauges_from_messages`, `run_d1bp_from_simple_update_gauges`, `gauge_all_simple` | nonnegative simple-update ↔ D1BP vector-message bridge; ordinary/relay/DIIS simple-update gauging with exact core compensation | Alkabetz & Arad (2021); Tindall & Fishman (2023); Müller et al. 2506.01779 |
| `d2bp_from_simple_update_gauges`, `run_d2bp_from_simple_update_gauges`, `simple_update_core_and_gauges_from_d2bp`, `gauge_all(norm="2norm")` | physical PEPS simple-update/Vidal ↔ D2BP density-message bridge: `diag(lambda)` initialization and PSD metric gauge conversion | Tindall & Fishman (2023); Gray et al. 2510.05647 |
| `loop_cluster_expand`, `LoopClusterResult` | loop **cluster** expansion (thin wrapper over quimb `contract_gloop_expand`) | Gray et al. 2510.05647 |
| `loop_series_expand`, `LoopSeriesTerm`, `LoopSeriesCache` | edge-resolved `P + Q` loop series for D1BP and D2BP; retains excited-bond degree and distinct embeddings/chord subsets | Evenbly et al. 2409.03108 |
| `partial_trace_loop_series_expand`, `compute_local_expectation_loop_series` | D2BP local reduced-density-matrix and scalar `P + Q` loop series; keeps physical output legs open and uses native Symmray virtual projectors | Evenbly et al. 2409.03108; quimb local loop-series API |
| `partial_trace_edge_loop_series_expand`, `compute_local_expectation_edge_loop_series` | D2BP local RDM and graded scalar observable expansion over canonical explicit Q-edge terms; does not reinterpret Quimb's local-region cutoff | Evenbly et al. 2409.03108; Pepsy API |
| `partial_trace_open_loop_series_expand`, `partial_trace_open_loop_series_sweep` | Lazy/path-first D2BP rho configuration sum over open Q paths, closed loops, and attached or disconnected path-plus-loop terms; `edge_cutoff` counts excited Q edges, bounded discovery uses `max_terms` / enumeration time / approximate geometry memory limits, and regional contraction paths are reused; `corridor_width` activates bounded weighted-shortest-path discovery, sampled connected loop decorations, and optional compressed boundary contraction; cyclic native fermionic graphs select the graded cluster route before edge discovery and use `cluster_size` | Evenbly et al. 2409.03108; Pepsy API |
| `compute_local_expectation_open_loop_series`, `diagnose_open_loop_series` | Scalar companion for long-range gates plus a non-contracting diagnostic pass: accepts native Fermion operators, reuses open-loop geometry, selects exact/corridor/graded-cluster routes, reports path/loop families and Cotengra FLOP/peak-memory estimates, and can reuse cached diagnostics before inserting the gate through the graded open-bond projector route | Evenbly et al. 2409.03108; Pepsy API |
| `partial_trace_loop_cluster_expand`, `compute_local_expectation_loop_cluster` | D2BP local reduced-density-matrix and scalar generalized-loop cluster expansion; combines BP-closed regions with inclusion--exclusion counts | Gray et al. 2510.05647; quimb local cluster API |
| `loop_expand` | explicit selector between the edge loop series and region loop-cluster expansion; preserves each method's cutoff and result metadata | Pepsy API |
| `partitioned_expand`, `pne_expand`, `PNEExpansionResult` | linear and combinatorial partitioned network expansions for D1BP/D2BP, with optional residue, explicit projectors, open outputs, and fixed recursive schedules | Evenbly, Gray & Chan 2512.10910 |
| `weight_pass`, `WeightPassingResult` | Appendix-C positive-weight passing on closed pairwise networks and rank-`r` projectors in the returned gauge | Evenbly, Gray & Chan 2512.10910 |
| `linked_cluster_expand`, `LinkedClusterCache`, `select_bp_candidate` | connected-loop/Ursell expansion of `log Z`, cached graph enumeration, and cluster-tail fixed-point selection | Midha & Zhang 2510.02290 |

---

## 1. Beyond-BP loop / cluster corrections (the 2024–2026 wave)

Four complementary ways to systematically correct BP's loop error. **This is the
active research frontier.**

- **Evenbly, Pancotti, Milsted, Gray, Chan**, *Loop series expansions for tensor
  networks*, Phys. Rev. Research **8**, 013245 (2026), arXiv:2409.03108.
  The loop **series**: resolve each edge as `|m><m|/I + P⊥`, sum generalized-loop
  excitations. **[quimb** `contract_loop_series_expansion`; **pepsy**
  `loop_series_expand` **]**. Pepsy keeps the actual excited bond set rather
  than using only a tensor-region length, so chorded supports and separate
  lattice embeddings remain distinct. Caveat: the series **diverges in the
  thermodynamic limit** (combinatorial disconnected loops) — the problem the
  cluster expansions below fix. Inserts `I − m⊗m` projectors →
  **fixed-point-sensitive**.
- **Park, Gray, Chan**, *Simulating quantum dynamics in 2D lattices with tensor
  network influence functional belief propagation*, Phys. Rev. B **112**, 174310
  (2025). Influence-functional BP + linked-cluster expansion; the direct
  predecessor of the loop cluster expansion.
- **Gray, Park, Evenbly, Pancotti, Kjønstad, Chan**, *Tensor Network Loop Cluster
  Expansions for Quantum Many-Body Problems*, Phys. Rev. B **113**, 235135
  (2026), arXiv:2510.05647. Loop **cluster** expansion (NLCE-style): product /
  sum of **exact contractions of growing loopy regions**, inclusion-exclusion
  counting numbers. *Simplifies GBP by avoiding generalized messages.*
  **Convergence-robust** in the finite-cluster sense: messages only close each
  region's boundary, so a system-covering cluster is exact and independent of
  the boundary messages.  The formal BP loop-cluster cancellations and
  tree/dangling reductions, however, assume fixed-point BP messages; arbitrary
  or unconverged messages should be interpreted as a boundary-closure cluster
  approximation and checked by sweeping cluster size. **[quimb**
  `contract_gloop_expand`; **pepsy** `loop_cluster_expand` **]**.
- **Midha, Zhang**, *Beyond Belief Propagation: Cluster-Corrected Tensor Network
  Contraction with Exponential Convergence*, arXiv:2510.02290 (2025). Expands
  **log Z (free energy)**, not Z → only **connected** clusters contribute
  (Mayer / Ursell / abstract polymer model). **Rigorous** exponential-convergence
  theorem + error bound via Kotecký–Preiss: `|log Z − F̃_m| ≤ n e^{−d(m+1)}` when
  `|Z_l| ≤ e^{−c|l|}`, `d = c − log(2(Δ−1)) − 1/2`. Size-independent. Built
  **around** the BP fixed point (fixed-point-sensitive). Forthcoming decoding
  paper (degeneracy / qLDPC threshold). ITensor.
- **Midha, Zhang, and collaborators**, *Algorithmic Locality via Provable
  Convergence in Quantum Tensor Networks*, arXiv:2604.21919 (2026). In its
  strong-injectivity regime, a unique convergent BP-like fixed point has
  exponentially decaying response to a local tensor perturbation. This
  motivates Pepsy's `BPState.update_local`: it is a scheduler-level warm start
  for D1BP, not a universal proof that arbitrary Pepsy tensors satisfy those
  hypotheses. A finite `radius` is therefore labelled a bounded local
  approximation; `radius=None` runs until D1BP's local queue is empty.
- **Sim, Park, Kim, Zang, Zou, Kim, Yang, … (Myung / UNIST group)**, *Stochastic
  Loop Corrections to Belief Propagation for Tensor Network Contraction*,
  arXiv:2603.08427 (2026). **Monte-Carlo** loop correction: exact factorization
  `Z = Z_BP × (loop-correction factor)`, MCMC-sample the loop-configuration sum
  with loop-constraint-preserving moves + umbrella sampling. **Unbiased**,
  controllable **statistical** (not truncation) error, any parameter regime.
  Pairwise Markov random field / symmetric edge potentials; 2D Ising.
- **Tindall, Sommers, Kappen**, *Contracting Tensor Networks with Generalized
  Belief Propagation*, arXiv:2604.24760 (2026). The **GBP / region-message**
  branch (Kikuchi cluster-variation method): pass messages within a hierarchy of
  **overlapping regions**; plain BP is the simplest-region corner. The powerful
  but harder-to-converge alternative that the cluster expansions deliberately
  avoid. quimb-adjacent (Tindall).

**Landscape.** Beyond BP = correct the loops, four ways: (a) loop **series**
[quimb, divergent], (b) loop **cluster** / NLCE [quimb, robust], (c) rigorous
**free-energy** cluster expansion [Mayer/KP bound], (d) **stochastic** MCMC loop
sampling [unbiased]. Plus (e) **GBP** region messages as the message-passing
realization of the same "regions beyond BP" idea.

---

## 2. BP ↔ tensor-network foundations & gauging

- **Alkabetz, Arad**, *Tensor networks contraction and the belief propagation
  algorithm*, Phys. Rev. Research **3**, 023073 (2021). BP ≡ simple-update /
  super-orthogonal / Vidal-gauge fixed point.
- **Tindall, Fishman**, *Gauging tensor networks with belief propagation*,
  SciPost Phys. **15**, 222 (2023). BP messages ↔ TN gauge; the practical
  BP-gauging recipe.
- **Implementation note.** quimb also exposes simple-update gauge cluster
  helpers such as `TensorNetworkGenVector.norm_gloop_expand(gauges=...)`.
  Those helpers use supplied SU/simple gauges as cluster-boundary data and do
  **not** run BP inside the cluster call.  As Gray emphasized, once simple
  gauges are converged for the same closed scalar/norm tensor network, they are
  BP/super-orthogonal fixed-point gauges up to message-gauge normalization, so
  tree-like regions are trivial and dangling/tree reductions are valid.  If the
  gauges are unconverged or borrowed from a different projected network, read
  the result as a gauge-boundary approximation and check by BP residual or
  cluster-size sweep.  The quimb issue found here was instead an `nfactor`
  normalization bug: `normalize_simple(gauges)` already returns the norm
  scaling, so `norm_gloop_expand` should not square-root that factor again.
- **Relay-SU implementation note (2026-07-14).**
  `gauge_all_simple(..., relay=RelayGaugeOptions(...))` layers per-bond,
  nonnegative disordered memory over
  Quimb's simple update. After mixing an external gauge it inversely rescales
  the two incident core tensor legs, preserving the represented TN exactly.
  Optional DIIS is likewise projected to a positive normalized gauge before
  compensation. The CPU NumPy parallel route uses edge-coloured, disjoint bond
  batches, not concurrent writes to neighbouring tensors; stable external bond
  ids are consequently required and multibond fusion is disabled.
- **Guo, Poletti, Arad**, *Block belief propagation algorithm for two-dimensional
  tensor networks*, Phys. Rev. B **108**, 125111 (2023).
- **Pancotti, Gray**, *One-step replica symmetry breaking in the language of
  tensor networks*, arXiv:2306.15004 (2023).
- **Sahu, Swingle**, *Efficient tensor network simulation of quantum many-body
  physics on sparse graphs*, arXiv:2206.04701 (2022).
- **Leifer, Poulin**, *Quantum graphical models and belief propagation*, Ann.
  Phys. **323**, 1899 (2008); **Robeva, Seigal**, *Duality of graphical models
  and tensor networks*, Info. & Inference **8**, 273 (2019).

---

## 3. Generalized BP / region graphs (Kikuchi / CVM)

- **Yedidia, Freeman, Weiss**, *Constructing free-energy approximations and
  generalized belief propagation algorithms*, IEEE Trans. Inf. Theory **51**,
  2282 (2005); *Understanding belief propagation and its generalizations* (2003).
  **[roots]** — GBP / region-graph / counting-number origin. **[quimb**
  `RegionGraph`, `gen_region_counts` — scaffolding, not a solver **]**.
- **Welling, Gelfand, Ihler**, *A cluster-cumulant expansion at the fixed points
  of belief propagation*, UAI 2012, arXiv:1210.4916. The cluster-cumulant
  expansion the loop cluster expansion rediscovers.
- **Kirkley, Cantwell, Newman**, *Belief propagation for networks with loops*,
  Science Advances **7**, eabf1211 (2021).

---

## 4. Loop-correction roots (statistical inference)

- **Chertkov, Chernyak**, *Loop calculus in statistical physics and information
  science*, Phys. Rev. E **73**, 065102 (2006); *Loop series for discrete
  statistical models on graphs*, JSTAT P06009 (2006). **[roots]** — the original
  loop series.
- **Montanari, Rizzo**, *How to compute loop corrections to the Bethe
  approximation*, JSTAT P10011 (2005).
- **Parisi, Slanina**, *Loop expansion around the Bethe–Peierls approximation*,
  JSTAT L02003 (2006).
- **Mooij, Kappen**, *Sufficient conditions for convergence of loopy belief
  propagation*, arXiv:1207.1405 (2012); **Gómez, Mooij, Kappen**, *Truncating the
  loop series expansion for belief propagation* (2007).

---

## 5. Convergence robustness / relay-BP

- **Müller, Alexander, Beverland, Bühler, Johnson, Maurer, Vandeth**, *Improved
  belief propagation is sufficient for real-time decoding of quantum memory*,
  arXiv:2506.01779 (2025). **Relay-BP** = disordered-memory (bias-term memory,
  incl. negative) + relayed warm-started legs. **[pepsy** `relay_bp`; **tensy**
  `RelayBpDecoder` **]**. FPGA follow-up arXiv:2510.21600. Code:
  github.com/trmue/relay.
- **Pakhunov**, *Belief Propagation Convergence Prediction for Bivariate Bicycle
  QEC Codes*, arXiv:2604.07995 (2026). Per-shot convergence prediction =
  confidence-gating signal.

---

## 6. BP decoding (QEC) — see the tensy `bp-decoding` skill

- **Poulin, Chung**, *On the iterative decoding of sparse quantum codes*,
  arXiv:0801.1241 (2008). The **degeneracy problem** (naive BP has no qLDPC
  threshold).
- **Panteleev, Kalachev**, *Degenerate quantum LDPC codes with good finite length
  performance*, Quantum **5**, 585 (2021). BP + **OSD**.
- **Roffe, White, Burton, Campbell**, *Decoding across the quantum LDPC code
  landscape*, Phys. Rev. Research **2**, 043423 (2020). **BP-OSD**.
- **Old, Rispler**, *Generalized belief propagation algorithms for decoding of
  surface codes*, Quantum **7**, 1037 (2023). **GBP** decoding.
- **Kaufmann, Arad**, *A blockBP decoder for the surface code* (2024).
- **Yao, Abu Laban, Häger, Graell i Amat, Pfister**, *Belief propagation decoding
  of quantum LDPC codes with guided decimation*, ISIT 2024.
- **Koutsioumpas, Sayginel, Webster, Browne**, *Automorphism ensemble decoding of
  quantum LDPC codes*, arXiv:2503.01738 (2025).
- **Wang, Zhang, Pan, Zhang**, *Tensor Network Message Passing*, Phys. Rev. Lett.
  **132**, 117401 (2024), arXiv:2305.01874. **TNMP** — exact short-loop clusters
  + BP on the sparse remainder; the structural bridge to BP-OSD.

---

## 7. BP-based classical simulation of quantum experiments

- **Tindall, Fishman, Stoudenmire, Sels**, *Efficient tensor network simulation
  of IBM's Eagle kicked Ising experiment*, PRX Quantum **5**, 010308 (2024).
- **Begušić, Gray, Chan**, *Fast and converged classical simulations of evidence
  for the utility of quantum computing before fault tolerance*, Sci. Adv. **10**,
  eadk4321 (2024).
- **Patra, Jahromi, Singh, Orús**, *Efficient tensor network simulation of IBM's
  largest quantum processors*, Phys. Rev. Research **6**, 013326 (2024).
- **Tindall, Mello, Fishman, Stoudenmire, Sels**, *Dynamics of disordered quantum
  systems with 2D and 3D tensor networks*, arXiv:2503.05693 (2025).

---

## 8. Classic BP roots

- **Pearl**, *Probabilistic Reasoning in Intelligent Systems* (1988).
- **Bethe**, *Statistical theory of superlattices*, Proc. R. Soc. A **150**, 552
  (1935).
- **Yedidia, Freeman, Weiss**, *Understanding belief propagation and its
  generalizations* (2003).
