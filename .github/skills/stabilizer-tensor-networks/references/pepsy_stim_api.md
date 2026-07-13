# pepsy + stim call surface for STN

Concrete public APIs for the two STN substrates. Verify signatures against the installed
versions; keep optional deps optional (`pytest.importorskip("stim")`).

## Environment
```bash
source ~/envs/py312/bin/activate
python -c "import stim, quimb, pepsy; print(stim.__version__, quimb.__version__, pepsy.__version__)"
```
If stim is missing: `python -m pip install stim`.

## Coefficient state |p> (paper: |nu>) — pepsy / quimb MPS

The public simulator types are exported at top level (`import pepsy`); see
`src/pepsy/__init__.py`. Helpers such as `pauli_combo_submpo` remain internal.

- **Create a simulator**: `sim = pepsy.MpsStabOptimizer(n, gates=None, chi=None,
  cutoff=1e-12, operator_tol=None, max_pauli_decomposition_qubits=2,
  track_infidelity=False, seed=None, to_backend=None)`. It owns an
  `STNState`; `sim.p` / `sim.state.p` is the coefficient MPS and `sim.nu` is an alias.
  `STNState` constructs the initial product MPS with `pepsy.ps_to_mps(n)`.
- **Exact/approximate control**:
  - `chi=None` is exact evolution; keep a small `cutoff` to remove numerically redundant
    Schmidt values introduced by bond-dimension-2 operators.
  - `chi=cap` bounds the coefficient bond. Add `track_infidelity=True` to compare each
    capped update with its uncapped local target. Read `.infidelities` and `.bond_history`.
- **Named physical stream entries**:
  - Clifford: `("h"|"s"|"sdg"|"x"|"y"|"z", q)`,
    `("cnot"|"cx"|"cy"|"cz"|"swap", a, b)`.
  - Rotation: `("rx"|"ry"|"rz", theta, q)`,
    `("rxx"|"ryy"|"rzz", theta, a, b)`,
    `("rot", theta, "XZ...", where)`, `("t", q)`, `("tdg", q)`.
  - Measurement/reset: `("measure", pauli, where[, outcome[, absorb_basis]])`,
    `("reset", where)`.
  - Explicit physical matrix: `(matrix, where)`. Clifford matrices update the tableau;
    non-Clifford 1q unitaries use ZYZ; all other few-qubit matrices use a frame-mapped
    Pauli decomposition, whose `4**k` cost makes it a few-qubit path. Matrix inputs may be
    backend arrays: they are converted to NumPy for classification and then coefficient
    operations return to `to_backend`. The fallback is limited to two qubits by default;
    increase `max_pauli_decomposition_qubits` only as an explicit cost opt-in.
  - Coefficient-frame MPO: `("submpo", mpo, where)` or the matching mapping event. It acts
    directly on `p`; do not use it for a physical-frame operator.
- **Internal coefficient updates**:
  - One-site frame images use `p.gate_(matrix, site, contract=True)`.
  - Multi-site two-branch operators use `pauli_combo_submpo(...)` followed by
    `p.gate_with_submpo_(..., where=true_window, max_bond=chi, cutoff=cutoff, info=state.info)`.
  - General Pauli sums copy/apply/weight branches, then combine them with a balanced,
    streaming MPS reduction and compress after additions.
- **Observables and collapse**: use `sim.expectation`, `sim.expectation_pauli_sum`,
  `sim.sample` (no collapse), and `sim.measure` (collapse). `absorb_basis=True` localizes
  the frame Pauli, updates both basis and MPS, and disentangles the projected pivot.
- **Computational-basis probabilities**: `sim.probability_bits(bits)` and
  `sim.sample_bits(shots, seed=...)` use conditional measurements on copies and do not form
  the dense statevector. `sim.amplitude` / `sim.probability` reconstruct densely and are for
  small systems only.
- **Magic-state injection**: `prepare_magic`, `inject_rz` (pi/4 multiples), `inject_t`,
  `inject_tdg`, `run_with_injection`, and `with_injection`. Ancillas must start in `|0>`;
  `reset` restores that precondition for reuse.
- **Norm and diagnostics**: `sim.norm()`, `sim.state.max_bond()`,
  `sim.pseudo_stabilizer_rank()` (dense/small-n), `.measurements`, `.infidelities`, and
  `.bond_history`. General non-unitary matrix entries intentionally change the norm.
- **Canonical centre**: pass `sim.state.info` through quimb operations that can move the
  centre. Use the simulator's canonicalization/renormalization helpers when extending
  projective paths; do not replace them with a blind full-MPS normalization.
- **Site ordering**: stim qubit `q`, MPS site `q`, and dense big-endian bit position `q`
  must agree. Physical supports can become dynamically spread after frame mapping, so a
  static physical-stream layout is not generally valid for `p`.

## Basis B(S,D) — stim tableau

- **State**: `sim = stim.TableauSimulator()` (tracks the full stabilizer + destabilizer
  tableau). Clifford ops: `sim.h(q)`, `sim.s(q)`, `sim.cnot(a,b)`, `sim.x/y/z(q)`, etc.
- **Read the tableau**: `t = sim.current_inverse_tableau()` → `stim.Tableau`. A
  `stim.Tableau` maps Paulis under the Clifford frame; `t.x_output(q)`, `t.z_output(q)`
  return `stim.PauliString`s. Use these to express a physical Pauli in the generator basis.
- **Pauli objects**: `stim.PauliString("X_ZY...")`, support `*` (multiply),
  `.commutes(other)`, `.sign`, `[i]` (0=I,1=X,2=Y,3=Z), `.to_numpy()` → `(xs, zs)` bit
  arrays. Use `.commutes` for the symplectic products
  $\hat d_i=\langle s_i,P\rangle,\ \hat s_i=\langle d_i,P\rangle$.
- **Direct tableau**: `stim.Tableau(n)` (identity), compose with
  `tab.append(gate, targets)` or `tab.then(...)`. `tab.to_numpy()` gives the boolean
  tableau blocks if you need raw $x,z,r$ entries.
- **Clifford Pauli rotations**: for $\exp(-ik\pi P/4)$, map each non-identity
  factor of $P$ to `Z`, gather its parity onto one support qubit with CNOTs,
  apply `S`/`Z`/`S_DAG` for `k mod 4`, and uncompute. Compile that linear-size
  circuit with `stim.Tableau.from_circuit`; do not build a dense unitary.
- **STN measurement**: do not call stim measurement as an independent second simulation.
  `MpsStabOptimizer.measure` computes the Born rule from the coefficient frame. The default
  fixed-basis path leaves the tableau unchanged; `absorb_basis=True` applies a localizing
  Clifford to `p`, absorbs its inverse into the basis, and projects one coefficient site.

## Actual module layout (`src/pepsy/optimizers/stabilizer_tn/`)
```
src/pepsy/optimizers/stabilizer_tn/
  __init__.py        # public API: STNState, MpsStabOptimizer, MPO builders
  stn_state.py       # STNState: stim tableau + MPS |p>, frame_pauli, dense reconstruction
  mps_stab_optimizer.py  # routing, rotations, measurement, injection, sampling, backends
  operators.py       # Pauli decomposition and full/windowed two-branch MPO builders
  paulis.py          # stim Pauli helpers (single_pauli, pauli_string, hermitian_pauli_terms)
  PLAN.md            # implementation record and research roadmap
tests/test_stabilizer_tn.py          # focused behavior and dense/stim validation
tests/test_stabilizer_tn_stress.py   # random/deep/long-range/general-matrix validation
```
If you add public symbols, follow repo Public API Rules: update the owning subpackage
`__all__`, top-level `src/pepsy/__init__.py`, `docs/api/`, and `tests/test_public_api.py`.

## Validation helpers
- Dense reference: build the statevector by expanding
  $\sum_i\nu_i\hat d_{\hat i}|\psi_{\mathcal S}\rangle$ (dense `p` × basis states from the
  tableau) and compare to `stim.TableauSimulator(...).state_vector()` for Clifford parts, or
  to a direct dense circuit for the full `{CNOT,RX,RY,RZ}` case.
- **Reconstruction identity (validated):** $\hat d_{\hat i}|\psi_{\mathcal S}\rangle=C|\hat i\rangle$,
  so $|\psi\rangle=C|p\rangle$ where $C$ is the tableau's Clifford unitary. Compute it as
  `sim.current_inverse_tableau().inverse().to_unitary_matrix(endian="big")` and the dense
  `|p>` as `p.to_dense().reshape(-1)` — both **big-endian**, so `C @ p_dense` is the
  statevector. `to_unitary_matrix` is **single precision**, so compare states up to global
  phase with a fidelity tolerance ~1e-6, not 1e-9.
- **Exact mode must still compress losslessly.** The bond-dim-2 rotation/projector MPO
  multiplies the `|p>` bond by 2 on every application. Use `chi=None` and a cutoff
  (`1e-12`) so redundant bonds are trimmed back to the true Schmidt rank —
  otherwise bonds grow as `2^(#rotations)` and blow up memory (observed: 512 GiB alloc).
- **Windowed `gate_with_submpo_` (fast path) — build the sub-MPO ON its target sites.**
  quimb's `gate_with_submpo_` aligns the sub-MPO to the MPS by the MPO's OWN site labels
  (via `gate_with_op_lazy_`); `where` only steers which region is canonicalized/compressed.
  So a windowed MPO labelled `0..w-1` acts on MPS sites `0..w-1` (WRONG — this caused the
  earlier fidelity 0.939). Build it on the real sites:
  `qtn.MatrixProductOperator(arrays, sites=(lo..hi), L=n)`, then
  `p.gate_with_submpo_(mpo, where=(lo..hi), max_bond=chi, cutoff=1e-12)`. This only
  canonicalizes/compresses `[lo,hi]`, so cost is O(depth·window) — **independent of `n`**
  for local circuits (verified: n=16/24/32 at fixed depth all ~0.28s). `pauli_combo_submpo`
  builds it. `|p>` stays a proper MPS: use `swap+split` for the measurement localizer's
  two-qubit gates; never merge two sites with an un-split `contract=True` update.
- **Track the canonical centre.** `STNState.info["cur_orthog"]` is initialized unknown,
  established once, and threaded through sub-MPO/swap+split operations. Local measurement
  expectations and normalization then touch only the centre tensor. Operator-sum rebuilds
  reset the tracker to the rebuilt MPS centre.
- Keep tests tiny/deterministic (fixed seeds, small $n$), per repo Examples guidance.

## Reference implementation mapping (bsc-quantic/stabilizer-TN)
- `v1.1` = single `stabilizers.py`; `v1.2` (latest) = packaged `src/` + disentangling
  experiments. Main class `gen_clifford` **inherits Qiskit's `Clifford`** (tableau) and holds
  a **quimb MPS** coefficient vector; `gen_clifford.compose(U)` accepts non-Clifford `U` and
  decomposes it with their methods. Example: `stabilizers_example.ipynb`.
- Our mapping: Qiskit `Clifford` → **stim tableau**; their MPS + `.compose` decomposition →
  **`MpsStabOptimizer`** frame mapping plus direct quimb local/sub-MPO/branch-sum updates.
  Use their notebook to cross-check Pauli decomposition and signs; do not vendor their code.
- Disentangling (their extra feature, optional for us): exact method arXiv:2412.17209 +
  sweeping disentangling arXiv:2407.01692 — candidates if you later want to actively reduce
  $\chi$ of `|nu>` by moving entanglement back into the basis.
