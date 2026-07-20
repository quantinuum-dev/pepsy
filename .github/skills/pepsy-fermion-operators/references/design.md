# Unified Pepsy fermion helper: study reference

This reference records the local study used to create the skill. It is a
design aid, not a substitute for reading the current source or installed
Symmray implementation.

## Current Pepsy state

Pepsy currently exposes `SpinfulFermion` from `pepsy` and `pepsy.tensors`. The
implementation lives in `src/pepsy/tensors/symmetric.py`, is re-exported by
`src/pepsy/tensors/symm_fermions.py`, and currently provides:

- spinful `U1` and `U1U1` local spaces;
- dense local operators and aliases;
- native Symmray one-site observables;
- onsite interaction and two-site hopping gates;
- deterministic second-order edge-colored gate streams;
- `SymHamiltonian` construction for `fermi_hubbard` and
  `fermi_hubbard_u1u1`;
- `SpinfulFermionHubbard = SpinfulFermion` compatibility alias.

Pepsy already supports spinless model metadata and Hamiltonian paths through
`fermi_hubbard_spinless`, but the model-facing helper does not yet unify that
space with the spinful helper. The intended evolution is a canonical
`Fermion(spinful=...)` helper, retaining the existing spinful names as
compatibility aliases.

## Symmray local conventions

The installed `symmray.fermionic_local_operators` module defines the following
local bases and charge maps.

| local space | ordered basis | physical charge map | supported symmetry |
| --- | --- | --- | --- |
| spinless | `|0>`, `a†|0>` | `[0, 1]` | `U1`, `Z2` |
| spinful | `|00>`, `d†|0>`, `u†|0>`, `d†u†|0>` | `U1: [0,1,1,2]` | `U1`, `Z2` |
| spinful | same basis | `U1U1/Z2Z2: [(0,0),(0,1),(1,0),(1,1)]` | `U1U1`, `Z2Z2` |

The `d†u†` order in the doubly occupied basis vector is deliberate. Preserve
it when constructing dense operators, local terms, and observables. Do not
replace it with a convenient Kronecker-product order without an explicit
fermionic sign check.

Symmray's local helper uses `FermionicOperator` labels and graded ordering to
compute internal signs. Its `build_local_fermionic_dense` output is raw local
tensor data: the physical operator action is recovered together with
fermionic contraction algebra. It is therefore not interchangeable with a
plain bosonic matrix product for off-site odd operators.

Useful upstream entry points are:

- `FermionicOperator`;
- `build_local_fermionic_elements`;
- `build_local_fermionic_dense`;
- `build_local_fermionic_array`;
- `get_spinless_charge_indexmap`;
- `get_spinful_charge_indexmap`;
- `fermi_hubbard_spinless_local_array`;
- `fermi_hubbard_local_array`;
- number and spin local-array helpers.

Primary source: [jcmgray/symmray](https://github.com/jcmgray/symmray),
particularly `symmray/fermionic_local_operators.py`.

## Recommended helper contract

Use one object with an explicit local-space switch. The exact name can change
only with user direction; `Fermion` is the recommended canonical name.

```python
spinless = Fermion(spinful=False, symmetry="U1", t=1.0, V=0.5, mu=0.0)
spinful = Fermion(spinful=True, symmetry="U1U1", t=1.0, U=8.0, mu=0.0)

spinless.dense_operator("number")
spinful.observable("number_up")
spinful.hopping_gate(dt=0.01)
spinful.interaction_gate(dt=0.01, site=3)
spinful.strang_gate_stream(edges, dt=0.01, sites=range(L))
spinful.hamiltonian(edges)
```

Recommended metadata and methods:

| surface | responsibility |
| --- | --- |
| `spinful` | choose two-mode Hubbard versus one-mode t-V local space |
| `symmetry` | choose `U1`, `Z2`, `U1U1`, or `Z2Z2` where valid |
| `model` | resolve to `fermi_hubbard`, `fermi_hubbard_u1u1`, or `fermi_hubbard_spinless` |
| `physical_sectors` | return the exact Symmray sector map |
| `dense_operator(name)` | return a documented local dense/raw operator with aliases |
| `operator_charge(name)` | return the charge used by Symmray metadata |
| `observable(name)` | cache a native fermionic Symmray operator |
| `hamiltonian(edges, **params)` | dispatch to the matching Pepsy Hamiltonian model |
| `hopping_gate(...)` | exponentiate a native two-site hopping term |
| `interaction_gate(...)` | exponentiate the local `U n_up n_down` or spinless density term |
| `strang_gate_stream(...)` | create deterministic canonical bundled entries |

The spinless parameter should use `V` for nearest-neighbor density interaction;
do not silently interpret a spinless `U` as a doublon interaction. Spinless
pairing or superconducting terms should be added only with explicit symmetry
and charge semantics, since they break particle-number conservation.

## Operator and charge registry

At minimum, the spinless registry should cover:

- `identity`, `parity`, `annihilate`, `create`, and `number`;
- aliases `n`/`occupation` for `number` if the API adopts aliases;
- neutral diagonal observables and neutral hopping gates;
- `create`/`annihilate` charges `+1`/`-1` for `U1`, reduced to parity charge
  `1` for `Z2`.

At minimum, the spinful registry should cover:

- `identity`, `parity`, `annihilate_up`, `create_up`, `number_up`;
- `annihilate_down`, `create_down`, `number_down`, `number`, `charge`, `sz`;
- `doublon`, `pair_create`, and `pair_annihilate`;
- aliases such as `n_up`, `n_down`, `number_up`, `number_down`, `spin_z`,
  `doublon`, and `pair_annihilation`.

For total `U1`, both spin flavors carry charge `+1` on creation and `-1` on
annihilation. For `U1U1`, use one tuple component per flavor, matching the
physical-sector convention rather than guessing from a user-facing label.
Pair charges are `+2/-2` for total `U1`, or `(1,1)/(-1,-1)` for `U1U1`.
Diagonal density, doublon, `sz`, and pair-create times pair-annihilate are
neutral. Apply modulo reduction for `Z2` and `Z2Z2`.

The pair convention must be consistent everywhere. In the current spinful
implementation, `pair_create = create_up @ create_down` and
`pair_annihilate = annihilate_down @ annihilate_up`, with the resulting local
identity `pair_create @ pair_annihilate == doublon`. Keep this convention in
eta-pairing examples and tests.

## Native versus JW workflows

The examples distinguish two valid paths:

1. Native path: `SymMPS`/`SymPEPS` with `fermionic=True`, native Symmray local
   terms, and native fermionic gate streams. Use term-by-term local expectation
   evaluation for a native state.
2. Compatibility/reference path: a bosonic/Jordan-Wigner MPO or dense exact
   reference for small systems. Explicitly allow any encoding conversion and
   compare energies or observables only after documenting the convention.

Never form a plain `numpy.kron` of separated odd local operators for the native
path. A raw tensor product misses graded signs. For custom multi-site terms,
construct the complete local term with `FermionicOperator` labels and
`build_local_fermionic_elements`, then convert it with correct sector maps and
charge metadata.

## Gate streams versus energy terms

These are related but different public representations. A gate stream should
show the product formula explicitly, for example onsite interaction gates in
the first and final half-layers around forward and reverse hopping layers.
An energy mapping is a sum of local Hamiltonian operators, not a list of
gates.

The compact `Fermion.hamiltonian(edges).terms` site layout returns an
edge-keyed dictionary. The upstream local Hubbard builder places

```text
(U / z_i) n_i_up n_i_down + (U / z_j) n_j_up n_j_down
```

inside the term for edge `(i, j)`, with analogous coordination-weighted
chemical-potential pieces. Here `z_i` counts edge occurrences incident on
site `i`, so summing all edge terms produces each onsite contribution exactly
once. This is correct for native local-term measurements, but the one-site
interaction is not a separate dictionary entry.

For examples that prioritize visible bookkeeping, build hopping-only edge
terms with `U=0` and `mu=0`, then add one-site native observables:

```python
hop = fermion.hamiltonian(edges, U=0.0, mu=0.0)
energy_terms = dict(hop.terms)
onsite = (
    fermion.U * fermion.observable("double")
    - fermion.mu * fermion.observable("number")
)
energy_terms.update({(site,): onsite for site in range(L)})
```

Use the one-element tuple `(site,)` as the local-expectation key. If repeated
edges were collapsed by a dictionary, multiply the corresponding edge terms
by their input multiplicity; do not use that correction to represent onsite
counting in the explicit one-site form.

## Example workflow map

Study these files when implementing or reviewing the helper:

- `pepsy_examples/fermi_hubbard/README.md`: native-fermion goals and
  spin-symmetry conventions;
- `pepsy_examples/fermi_hubbard/PLAN.md`: expected observables, gates,
  weighted edges, and future bilayer/checkerboard extensions;
- `fh_energy.ipynb`: direct fermionic `SymPEPS` and Hubbard energy setup;
- `fh_img.ipynb` and `fh_continoustime.ipynb`: imaginary/real-time workflows;
- `fh_mps.ipynb`: native terms, DMRG-to-native conversion, and gate evolution;
- `mps/mps_fermions_helpers.py`: reusable number/doublon/pair observables;
- `mps/fh_etienne_helper.py`: custom spin-changing gates and explicit
  `build_local_fermionic_elements` use;
- `mps/README.md`: native MPS versus bosonic MPO warning and Trotter usage.

The examples emphasize that total `U1` is needed for the spin-mixing shallow
state-preparation circuit, while `U1U1` is appropriate when the physical
Hubbard dynamics conserve each spin number separately. Do not force one
symmetry choice for all workflows merely because the local space is spinful.

## Minimum regression matrix

For a unified helper, test at least:

1. Spinless `U1`: dimensions, sectors, number/creation/annihilation, t-V
   Hamiltonian, native hopping gate, and a two-site dense reference.
2. Spinless `Z2`: parity charge and native operator metadata.
3. Spinful total `U1`: four-state sectors, pair/doublon algebra, Hubbard
   Hamiltonian, and spin-preserving hopping.
4. Spinful `U1U1`: tuple charges for up/down/pair operators and Hubbard gate
   stream.
5. Spinful `Z2`/`Z2Z2` if advertised: modulo charge behavior and conversion.
6. NumPy-backed native arrays plus optional Torch/JAX/CuPy backend conversion
   when installed.
7. Compatibility aliases and top-level lazy exports.

Use small systems and exact dense/JW references only as sign or energy oracles;
assert that the actual native tensors and gates remain Symmray fermionic arrays.
