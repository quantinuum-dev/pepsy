---
name: pepsy-fermion-operators
description: Design, implement, review, or document Pepsy's native Symmray fermion model-facing API, including a unified spinless/spinful helper, local operators, observables, Hamiltonian terms, native gates, gate streams, charge handling, and compatibility aliases. Use when changing src/pepsy/tensors/symmetric.py or symm_fermions.py, fermionic SymmMPS/SymPEPS workflows, Fermi-Hubbard or spinless t-V examples, or tests/docs for these APIs.
---

# Pepsy Fermion Operators

Use this skill to extend Pepsy's model-facing native fermion helpers without
losing Symmray's graded-array semantics. Keep the public workflow simple for
users while making spin flavor, symmetry, local basis, operator charge, and
gate construction explicit internally.

## Read first

- `AGENTS.md` and any nested instructions.
- `references/design.md` in this skill for the studied basis, charge, and API
  conventions.
- `src/pepsy/tensors/symmetric.py`, especially the current
  `SpinfulFermion`, `symm_operator_from_dense`, local fermion helpers, and
  `SymHamiltonian` model dispatch.
- `src/pepsy/tensors/symm_fermions.py` and `src/pepsy/tensors/__init__.py` for
  the model-facing namespace and aliases.
- `tests/test_symmetric_tensors.py`, `tests/test_public_api.py`, and
  `docs/api/tensors/symmetric.md` before changing public behavior.
- The concrete workflows under
  `/home/reza.haghshenas@quantinuum.com/pepsy_examples/fermi_hubbard`, starting
  with `README.md`, `PLAN.md`, `fh_energy.ipynb`, `fh_mps.ipynb`, and
  `mps/mps_fermions_helpers.py`.
- The installed/upstream `symmray.fermionic_local_operators` implementation,
  especially `FermionicOperator`, `build_local_fermionic_elements`,
  `build_local_fermionic_array`, `get_spinless_charge_indexmap`,
  `get_spinful_charge_indexmap`, and the two Hubbard local-array builders.

## Target public shape

Unless the user requests another name, make `Fermion` the canonical helper and
accept `spinful: bool` as the model-space switch. Preserve the existing
`SpinfulFermion` and `SpinfulFermionHubbard` names as compatibility aliases or
thin compatibility constructors until an explicit breaking change is agreed.

Keep one model-facing object responsible for:

- local dense operators and aliases, such as number, creation/annihilation,
  parity, spin, doublon, and pair operators where the local space supports
  them;
- native Symmray observables with the correct physical sectors, fermionic flag,
  dual legs, and operator charge;
- one-site onsite gates and two-site hopping/pairing gates;
- deterministic first/second-order gate streams over sites and edges;
- matching `SymHamiltonian` construction for spinful Hubbard or spinless t-V;
- occupation/charge helpers and model metadata.

Prefer an explicit local-space constructor, with physical couplings supplied
at the term, Hamiltonian, or gate-stream call site:

```python
spinless = Fermion(spinful=False, symmetry="U1")
spinful = Fermion(spinful=True, symmetry="U1U1")

spinless.hamiltonian(edges, t=1.0, V=0.5)
spinful.strang_gate_stream(edges, dt=0.01, t=1.0, U=8.0)
```

`Fermion` must not store `t`, `U`, `V`, or `mu`. Reject a Hubbard `U` on a
spinless object instead of silently dropping it.

Do not hide the spinless/spinful choice behind a model-name string when a
boolean or explicit local-space descriptor can make it clear. Model-name
strings may remain accepted for compatibility and Hamiltonian dispatch.

## Symmray invariants

Use Symmray's native fermionic construction as the source of truth:

1. Represent each mode with `symmray.FermionicOperator` labels.
2. Define local terms as ordered products of those operators.
3. Build raw local tensor data with
   `build_local_fermionic_elements` or
   `build_local_fermionic_array`; these include internal anticommutation signs.
4. Convert through Pepsy's `symm_operator_from_dense` only when its metadata
   contract is appropriate, always passing the physical sectors, charge,
   `fermionic=True`, and the number of acted-on sites.
5. Exponentiate the native local term for a native gate, preserving the
   Symmray fermionic array rather than converting odd operators to a plain
   `numpy.kron` product.

Never add Jordan-Wigner or parity strings to the native Symmray path. A
bosonic/JW MPO is a distinct compatibility/reference path and must be clearly
named, tested separately, and never silently substituted for native local-term
measurements or native gate evolution.

## Design workflow

### 1. Resolve the local space and symmetry

Validate the `spinful` and `symmetry` combination before constructing arrays.
Spinless local spaces use two states and support `U1` or `Z2`; spinful local
spaces use four states and support `U1`, `U1U1`, `Z2`, or `Z2Z2` as supported by
the installed Symmray version. Reject unsupported combinations with a focused
error.

Keep these concepts separate:

- `spinful`: number of local fermionic modes;
- `symmetry`: conserved charge group used by Symmray;
- `fermionic=True`: graded array algebra, not a choice of Hamiltonian;
- model metadata: Hubbard (`U`) versus spinless t-V (`V`) and optional terms.

### 2. Centralize basis and operator metadata

Create one internal registry for each local space. It must define the ordered
local basis, physical sectors, mode labels, operator aliases, and charge
functions. Derive dense operators and Symmray operators from that registry;
do not duplicate spinless and spinful sign logic in unrelated gate helpers.

Use the exact Symmray basis ordering recorded in `references/design.md`.
Treat pair ordering as an API convention and test it algebraically, for
example `pair_create @ pair_annihilate == doublon` where that identity applies.

Reject spin-only names (`number_up`, `Sx`, `pair_create`, etc.) for spinless
objects unless a mathematically well-defined spinless interpretation is
explicitly documented.

### 3. Assign operator charges deliberately

Number, parity, density, spin-conserving hopping, and diagonal onsite terms
are neutral under the relevant symmetry. Creation, annihilation, spin-flip,
and pair operators carry nonzero charges as documented by the registry.

For every operator exposed through `observable(...)`, verify that its charge
matches its Symmray array metadata. In particular, total-`U1` spinful creation
has charge `+1`, while `U1U1` creation has a flavor-specific tuple charge.
Use negative charges for annihilation and the corresponding summed charge for
pairs. Apply group reduction for `Z2` variants.

### 4. Build Hamiltonians and gates through shared terms

Make `hamiltonian(...)`, `hopping_gate(...)`, onsite interaction gates, and
gate-stream builders share the same term definitions and parameter resolution.
Support scalar, per-site, per-edge, and callable parameters only where nearby
Pepsy APIs already support them, and preserve edge orientation for Peierls
phases.

Keep gate streams and energy-term mappings conceptually separate. A native
Strang stream may contain explicit one-site interaction gates in its two
onsite half-layers. The default site-layout `SymHamiltonian.terms` mapping is
instead keyed by edges: Symmray distributes each onsite Hubbard and
chemical-potential contribution over the incident edges by site coordination,
so summing the edge terms counts each one-site contribution once. If an
example or optimizer workflow needs the bookkeeping to be visible, construct
the edge terms with `U=0` and `mu=0`, then add
`U * fermion.observable("double") - mu * fermion.observable("number")` under
each one-site key `(site,)`. Use `(site,)`, rather than the bare integer
`site`, for Quimb local-expectation mappings. Restore duplicate edge
multiplicity only when the input edge list contains repeated keys that the
term dictionary has collapsed.

Return canonical bundled entries `[(gate, where), ...]` wrapped in
`SymGateStream` with accurate `dt`, `imaginary`, and Trotter `order` metadata.
Use deterministic edge coloring for Strang streams so each hopping layer is
vertex-disjoint; validate empty edges, self-edges, and invalid site arguments.

### 5. Preserve backend behavior

Keep dense construction in NumPy only when it is an intentional small local
construction. Convert Symmray blocks through Pepsy's existing backend helper
or `to_backend` callback. Do not silently coerce user-provided gate arrays to a
different backend. Test at least NumPy and any installed optional backend that
the changed path claims to support.

### 6. Update the public surface together

For a public class or alias, update the owning `__all__`, lazy top-level
exports in `src/pepsy/__init__.py`, API docs, and public API/layout tests.
Keep the model-family namespace (`SymmFermions`) useful for factories, but do
not make it the only way to construct the helper.

## Validation

Run focused tests first, then broaden as appropriate:

```bash
source ~/envs/py312/bin/activate
pytest -q tests/test_symmetric_tensors.py
pytest -q tests/test_public_api.py tests/test_package_layout.py
```

For a new unified helper, add tests covering both `spinful=False` and
`spinful=True`, every advertised symmetry family, basis dimensions and charge
maps, operator aliasing, native-array metadata, gate unitarity or imaginary
time behavior, stream ordering, and Hamiltonian model dispatch. Include a
small dense reference for spinless and spinful hopping signs and keep native
Symmray-vs-JW comparisons explicitly separate.

Before finishing, inspect `git diff`, preserve unrelated user changes, and
report any optional-dependency or example-environment limitations.

## Reference navigation

Read [references/design.md](references/design.md) for the detailed study
findings, recommended API table, exact local basis conventions, and example
workflow map. Use it as design guidance, then verify current source and
installed Symmray APIs because both can evolve.
