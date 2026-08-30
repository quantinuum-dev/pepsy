# Native Symmray implementation helper

Read this when changing `Fermion`, native `SymMPS`/`SymPEPS`, fermionic
MPO/PEPO construction, MPS FIT, PEPS energy, or the SymDMRG2 bridge. It is a
small implementation map, not a replacement for the current source or the
[upstream Symmray repository](https://github.com/jcmgray/symmray).

## Upstream watch

Before changing native Symmray code, check the [latest
docs](https://symmray.readthedocs.io/en/latest/index.html), the [official
changelog](https://symmray.readthedocs.io/en/latest/changelog.html), and the
installed `symmray.__version__`. If the docs and installed package disagree,
use the installed source and a focused regression as the compatibility
boundary. Read the upstream `README.md` and `pyproject.toml` when a feature or
dependency appears to have changed. Record the audit date, installed version,
and adopted or rejected features in
[`docs/development/notes/symmray.md`](../../../../docs/development/notes/symmray.md).

Last audited: **2026-08-24**. The shared Python 3.12 environment reports
`symmray 0.2.2.dev36+g7d1aa0c69`, so development `main` and the published docs
must not be treated as interchangeable without checking the actual API.

## Mental model

- `fermionic=True` selects Symmray's graded/Grassmann algebra. It does not
  mean “add Jordan--Wigner parity strings”. A JW/bosonic route is a separate
  compatibility representation.
- Charge sectors, fermionic parity, dummy modes, and tensor labels are
  different pieces of metadata. Preserve all of them through copies,
  `from_blocks`, reshapes, splits, and conversions.
- `build_local_fermionic_elements` and
  `build_local_fermionic_dense` return raw graded tensor data. That data is
  not an ordinary site-major operator matrix: flattening it before native
  construction loses the contraction convention.

## Current upstream opportunities

The current upstream API exposes several useful directions for Pepsy. Treat
these as adoption guidance, not permission to bypass Pepsy's existing backend
and metadata contracts:

- **Explicit conversion:** Symmray arrays provide `.to(...)` for backend,
  dtype, and device conversion. A future Pepsy adapter can use it after an
  explicit user/backend decision; it must not silently move or densify blocks.
- **Backend-generic code:** Symmray advertises Python Array API and Autoray
  dispatch. Prefer public `autoray`/array-namespace operations over private
  backend type checks when adding generic code.
- **Contraction policy:** `tensordot(mode="fused")` can be faster but may
  materialize missing blocks; `mode="blockwise"` has more Python overhead and
  is the safer fallback for sparse, mixed, or metadata-sensitive native
  fermionic operands. Let Symmray prepare phases before either mode.
- **Local terms:** the upstream local builders cover pairing and expose
  coordination-aware Hubbard terms. Compare basis order, operator ordering,
  and charge metadata against Pepsy before adopting them in `Fermion`.
- **Serialization and linalg:** `to_pytree`/`from_pytree`, `from_blocks`, and
  the native truncated linalg drivers are candidates for checkpointing and
  GPU/autodiff work, but each needs a backend-preserving regression.

Do not make a native implementation depend on a README-only feature. First
probe the installed API, then add the smallest test that proves metadata,
phases, reconstruction, and backend behavior.

## Golden path for a new native term

1. Use the Symmray local basis exactly. Spinless is `(|0>, a†|0>)` with
   charges `[0, 1]`; spinful is `(|00>, d†|0>, u†|0>, d†u†|0>)` with total-
   `U1` charges `[0, 1, 1, 2]` or `U1U1` charges
   `[(0, 0), (0, 1), (1, 0), (1, 1)]`.
2. Define every local term with `FermionicOperator` labels and construct the
   complete local product with Symmray's fermionic-local helpers. Do not
   assemble separated odd operators with `numpy.kron`.
3. Convert raw data through Pepsy's `symm_operator_from_dense(...,
   fermionic=True, sites=n)` with explicit physical sectors and operator
   charge, then exponentiate the native operator. Do not reshape the raw
   tensor into a plain matrix first.
4. Keep coefficients on their original backend. The JAX path must use the
   indexed elements plus `.at[...]` because Symmray's dense helper performs
   mutable indexed updates; Torch/autodiff values must not be cached or
   silently converted to NumPy. Use Symmray `.to(...)` only for an explicit
   backend/dtype/device conversion at Pepsy's boundary.
5. Test a two-site hopping term against both the native local-term path and a
   small dense/JW oracle. Include a non-contiguous edge so skipped-site signs
   are exercised.

## The sign and phase traps

| Situation | Correct rule |
| --- | --- |
| Odd array or odd local operator | Give it a stable `label=` and retain its `dummy_modes`; unlabeled odd arrays can have an unresolved global phase. |
| Native contraction | Let Symmray's `tensordot`/Quimb graph planner perform the graded ordering. Before raw block arithmetic, use `phase_sync`; do not replace the contraction with a hand-written pairwise loop. |
| Native MPO channel crossing an omitted site | Insert scalar `-identity` for an odd operator-Schmidt channel. Do not insert a JW parity operator: that makes the native operator state-dependent. |
| Bra/conjugation | Preserve Symmray's conjugation phase convention. Apply `phase_flip`/`phase_transpose` only for the documented dual-leg or contraction layout, then synchronize before ordinary block operations. |
| Fermionized state reconstructed from a bosonic MPS | Put the physical leg last before `FermionicArray.from_blocks`; apply the inverse per-site gauge signs and restore `label=site`. Diagonal energy can look correct while hopping and other off-diagonal terms remain wrong if this is skipped. |
| Native QR | Route lossless tree/native QR through Pepsy's shared helper. For Symmray block-sparse tensors use `stabilized=False` to avoid structural-zero phase `0/|0|` NaNs; use the explicit graded SVD for truncation. |
| Gate cache | Cache by operator contents and metadata, never Python `id`. Do not cache tensors carrying an autodiff graph. |

## MPS, PEPS, and SymDMRG2 boundaries

### MPS

Native fermionic FIT deliberately works with a conjugated native working MPS,
real outside overlap environments, and explicit dual-leg corrections. After
local writeback, resolve any remaining odd dummy-mode global phase. Native
sector growth and graded auto-swap are the source of truth; do not warm-start
this path by densifying or by adding random dense padding. With zero cutoff,
remove only structural zeros using the smallest positive dtype-safe cutoff.

### PEPS and PEPO

For a native fermionic MPO expectation, apply the MPO sitewise to the ket and
then contract the resulting factorized network; an arbitrary MPO/bra/ket
interleaving can change graded phases. Native PEPO construction follows the
chosen snake ordering, keeps odd-term labels, and treats identity and
periodic-bond cases as specialized native constructions. Use the scoped
CTMRG compatibility context for cyclic native PEPS/PEPO calls; do not patch
installed Quimb or silently fall back to a JW object.

### SymDMRG2

The hot local DMRG matvec intentionally uses bosonized Symmray arrays. This is
an internal performance boundary, not evidence that native fermion metadata is
unnecessary: compiled plans must retain phase/dummy-mode signatures and fall
back to direct Symmray blockwise contraction for mixed, fused, non-NumPy, or
metadata-mismatched operands. Validate any compiled plan against a fresh
direct contraction before trusting it. `fermionic_state()` is the explicit
debosonization boundary; check native term energy and off-diagonal correlators,
not only the bosonic-MPO energy.

For convergence, start SymDMRG2 from a product-grown state with a gentle bond
ramp (or a canonical random-unitary state). A full-chi random block fill can
lock the calculation into poor charge sectors; a mixer does not reliably fix
that initialization.

## Minimal debugging loop

1. Print the encoding of every state, operator, and MPO/PEPO: native
   fermionic, bosonic Symmray, or mixed.
2. Check symmetry, charge, duals, physical-leg position, labels, and
   `dummy_modes` before inspecting numerical values.
3. Compare three small-system quantities where applicable: native local-term
   energy, native MPO/PEPO energy, and dense/JW ED. A diagonal-only comparison
   is insufficient.
4. Repeat with an odd operator, a skipped/non-contiguous support, both edge
   directions, and `U1U1` if the implementation claims spin-resolved support.

## Upgrade checklist

For each upstream audit:

1. Compare the installed version with the docs/changelog and note the date in
   `docs/development/notes/symmray.md`.
2. Check whether changed APIs affect local fermionic builders, phase handling,
   contraction modes, linalg drivers, or backend conversion.
3. Classify each candidate as **adopt**, **prototype**, or **defer**. Keep
   proposals out of stable native behavior until a focused native-vs-dense
   regression passes.
4. Run the closest fermion/Symmray tests and `python -m ruff check src tests`.

This keeps the agent current without turning every upstream release into an
unreviewed behavior change.

## Commit trail worth revisiting

These commits contain the tested patterns behind this helper:

- `0893bd8` — unified `Fermion` model-facing API and native workflows.
- `793b630` — scalar phases for native MPO channels across skipped sites.
- `1b064cd`, `b82b902` — SymDMRG2 debosonization and physical-leg ordering.
- `0397d96`, `6945767` — native fermionic FIT environments, writeback, and
  sector-safe warm starts.
- `45035e5`, `e3ffba8`, `f14a826` — encoding guards and native PEPS energy/
  CTMRG boundaries.
- `b4ffaf6`, `ed94c67`, `07fb646` — native PEPO/BP, periodic bonds, and the
  safe identity PEPO route.
- `f8ac1c5`, `f0c5ebf`, `e6009ec`, `fd5f383`, `0b22d90` — SymDMRG2 compiled
  block plans, validation, and the bond-dimension-scoped performance limits.
- `a1bb7ff` — content-addressed native fermion gate caching.

When a future change conflicts with one of these rules, inspect the commit and
its regression test before generalizing the implementation.
