# pepsy + stim call surface for STN

Concrete public APIs for the two STN substrates. Verify signatures against the installed
versions; keep optional deps optional (`pytest.importorskip("stim")`).

## Environment
```bash
source ~/envs/py312/bin/activate
python -c "import stim, quimb, pepsy; print(stim.__version__, quimb.__version__, pepsy.__version__)"
```
If stim is missing: `python -m pip install stim`.

## Coefficient state |nu> — pepsy / quimb MPS

All exported at top level (`import pepsy`); see `src/pepsy/__init__.py`.

- **Build initial |0...0> MPS**: `pepsy.ps_to_mps(n)` (product-state MPS, χ=1; default
  `theta=0` ⇒ every site `[1,0]` = `|0>`). Returns a `quimb` `MatrixProductState`; use
  `.to_dense().reshape(-1)` for the big-endian dense vector. Also `pepsy.ps_to_peps`,
  `pepsy.hrps_to_mps` for other geometries.
- **Apply gate streams (main engine)**: `pepsy.MpsOptimizer(p, gates, chi=..., mode=...,
  inplace=..., track_infidelity=...)` where `gates` is the canonical bundled stream
  `((gate, where), ...)`. This is how you apply the CNOT-cascade rotations to `|nu>`.
  - `mode="exact"` → no truncation (ground truth). Compressed modes
    `{"dmrg","svd","mpo","swap","mix"}` + `chi` → bounded-$\chi$ approximate mode.
  - After `.run()`, read `.infidelities` / `.losses` for the accuracy trace, and
    `.normalizations` for norm bookkeeping (`p.exponent`). `ind_id="k{}"` for 1D sites.
  - Single one-off/exact gate application (no optimizer needed):
    `pepsy.gate(tn, gates, where=..., which=...)` / `pepsy.gate_simple(...)`.
- **Gate matrices** (build the stream entries):
  - single-qubit: `pepsy.x/y/z/h/hadamard/s/sdg/t/tdg`, `pepsy.rx(theta)`,
    `pepsy.ry(theta)`, `pepsy.rz(theta)`, `pepsy.phase`, `pepsy.u1/u2/u3`, `pepsy.pauli`.
  - two-qubit: `pepsy.cnot/cx/cy/cz/cphase/swap/iswap`, `pepsy.rxx/ryy/rzz`,
    `pepsy.crx/cry/crz`, `pepsy.fsim`, `pepsy.su4`.
  - Gate stream form is canonical `[(gate, where), ...]` (see repo `AGENTS.md`).
  - User gate tensors are NOT auto-coerced to the TN backend — keep dtypes/backends aligned.
  - The measurement central op (Eq. 44) is a non-unitary $2\times2$ tensor — pass it as a
    custom gate matrix in the stream, then project + renormalize.
- **Observables / expectations**: `pepsy.expec_mpo`, `pepsy.measure_obs`,
  `pepsy.id_to_mpo`, `pepsy.ps_to_mpo`. For $\langle\nu|X_{\hat n}Z_{\hat m}|\nu\rangle$
  build the Pauli string as an MPO and contract.
- **Norm / fidelity**: `pepsy.tn_norm(nu)` (assert ≈ 1 after each step),
  `pepsy.tn_fidelity`.
- **Sampling**: `pepsy.MpsSampler`, `pepsy.VecSampler` (+ `MpsSampleResult`,
  `MpsBatchSampleResult`).
- **Bond dimension / truncation**: use the underlying quimb MPS methods on the tensor
  network (e.g. `.max_bond()`, `.compress(max_bond=...)`) for the approximate-mode cap.
- **Contraction optimizer**: `pepsy.build_optimizer(progbar=False)` or
  `contraction_opt="auto-hq"`; `pepsy.build_compressed_optimizer` for compressed contraction.
- **Site ordering**: fix a single `pepsy.OneDMap` convention and reuse it so the MPS site
  order matches the stim qubit order exactly.

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
- **Measurement**: `sim.measure(q)` / `sim.measure_kickback(...)`; for STN you drive the
  outcome from $\langle\mathcal O\rangle$ computed on `nu`, then apply the matching stim
  tableau update so basis and coefficient state stay consistent.

## Actual module layout (`src/pepsy/optimizers/stabilizer_tn/`)
```
src/pepsy/optimizers/stabilizer_tn/
  __init__.py        # public API: STNState, MpsStabOptimizer, MPO builders
  stn_state.py       # STNState: stim tableau + pepsy MPS |nu>, init |0..0>, nu_frame_pauli
  mps_stab_optimizer.py  # MpsStabOptimizer: gate-stream simulator (Clifford/rotation/measure/submpo)
  operators.py       # pauli_combo_mpo / pauli_rotation_mpo / single_qubit_* / pauli_matrix
  paulis.py          # stim Pauli helpers (single_pauli, pauli_string, hermitian_pauli_terms)
  PLAN.md            # roadmap (magic-state injection, disentangling sweep, ...)
tests/test_stabilizer_tn.py   # importorskip("stim"); validate vs dense/stim
```
If you add public symbols, follow repo Public API Rules: update the owning subpackage
`__all__`, top-level `src/pepsy/__init__.py`, `docs/api/`, and `tests/test_public_api.py`.

## Validation helpers
- Dense reference: build the statevector by expanding
  $\sum_i\nu_i\hat d_{\hat i}|\psi_{\mathcal S}\rangle$ (dense `nu` × basis states from the
  tableau) and compare to `stim.TableauSimulator(...).state_vector()` for Clifford parts, or
  to a direct dense circuit for the full `{CNOT,RX,RY,RZ}` case.
- **Reconstruction identity (validated):** $\hat d_{\hat i}|\psi_{\mathcal S}\rangle=C|\hat i\rangle$,
  so $|\psi\rangle=C|\nu\rangle$ where $C$ is the tableau's Clifford unitary. Compute it as
  `sim.current_inverse_tableau().inverse().to_unitary_matrix(endian="big")` and the dense
  `|nu>` as `nu.to_dense().reshape(-1)` — both **big-endian**, so `C @ nu_dense` is the
  statevector. `to_unitary_matrix` is **single precision**, so compare states up to global
  phase with a fidelity tolerance ~1e-6, not 1e-9.
- **Exact mode must still compress losslessly.** The bond-dim-2 rotation/projector MPO
  multiplies the `|nu>` bond by 2 on every application. Use a cutoff (`1e-12`) with no
  `max_bond` (exact) so the redundant bond is trimmed back to the true Schmidt rank —
  otherwise bonds grow as `2^(#rotations)` and blow up memory (observed: 512 GiB alloc).
- **Windowed `gate_with_submpo_` (fast path) — build the sub-MPO ON its target sites.**
  quimb's `gate_with_submpo_` aligns the sub-MPO to the MPS by the MPO's OWN site labels
  (via `gate_with_op_lazy_`); `where` only steers which region is canonicalized/compressed.
  So a windowed MPO labelled `0..w-1` acts on MPS sites `0..w-1` (WRONG — this caused the
  earlier fidelity 0.939). Build it on the real sites:
  `qtn.MatrixProductOperator(arrays, sites=(lo..hi), L=n)`, then
  `nu.gate_with_submpo_(mpo, where=(lo..hi), max_bond=chi, cutoff=1e-12)`. This only
  canonicalizes/compresses `[lo,hi]`, so cost is O(depth·window) — **independent of `n`**
  for local circuits (verified: n=16/24/32 at fixed depth all ~0.28s). `pauli_combo_submpo`
  builds it. NB `|nu>` is always a proper MPS (only 1-qubit gates + sub-MPOs touch it);
  never apply a 2-qubit gate to `|nu>` with `contract=True` (it merges sites and breaks it).
- Keep tests tiny/deterministic (fixed seeds, small $n$), per repo Examples guidance.

## Reference implementation mapping (bsc-quantic/stabilizer-TN)
- `v1.1` = single `stabilizers.py`; `v1.2` (latest) = packaged `src/` + disentangling
  experiments. Main class `gen_clifford` **inherits Qiskit's `Clifford`** (tableau) and holds
  a **quimb MPS** coefficient vector; `gen_clifford.compose(U)` accepts non-Clifford `U` and
  decomposes it with their methods. Example: `stabilizers_example.ipynb`.
- Our mapping: Qiskit `Clifford` → **stim tableau**; their MPS + `.compose` decomposition →
  **`MpsOptimizer`** stream (exact) or bounded-$\chi$ (approximate). Use their notebook to
  cross-check the Pauli→basis decomposition and update signs; do not vendor their code.
- Disentangling (their extra feature, optional for us): exact method arXiv:2412.17209 +
  sweeping disentangling arXiv:2407.01692 — candidates if you later want to actively reduce
  $\chi$ of `|nu>` by moving entanglement back into the basis.
