# PePsY Library

<img src="assets/pepsy-icon.svg" alt="PePsY icon" width="220" />

**PePsY** is a tensor-network package for circuit simulation, contraction,
optimization, sampling, and variational Monte Carlo workflows.

The name **PePsY** is a stylized shorthand for “PEPS in Python”: it points to
projected entangled-pair states while leaving room for PePsY’s broader MPS,
circuit, sampling, stabilizer, and VMC workflows. Use **PePsY** for the
project name and `pepsy` for the Python package and import name.

Current package version: `0.4.0` (from `pyproject.toml` / `pepsy.__version__`).
See the [changelog](CHANGELOG.md) for release history and versioned changes.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development and test profiles.

## Package Layout

Core namespaces are organized by responsibility:

- `backends/`: backend selection, conversion, and linear algebra registration
- `tensors/`: maps, constructors, contractions, observables, and symmetric tensors
- `operators/`: gates, gate application, MPO/PEPO builders, and Hamiltonians
- `boundary/`: PEPS boundary states, sweeps, norms, and overlaps
- `solvers/`: gradient-based and finite-difference solvers
- `fitting/`: local tensor fitting routines
- `optimizers/`: MPS, MPO, PEPS, sweep, and global workflows
- `sampling/`: MPS, PEPS, vector, and tree samplers

Advanced namespaces are explicit:

- `bp/`: belief propagation, loop corrections, and PNE
- `vmc/`: optional Torch and NetKet/JAX VMC adapters
- `experimental/`: lazy entry points for advanced domains
- `_internal/`: private formatting and utility helpers
- `examples/`: lightweight runnable examples kept with the package
- `../pepsy_examples/`: external notebooks and smoke examples, including direct
  fermionic Symmray Fermi-Hubbard starters under `fermi_hubbard/`
- `docs/`: Markdown documentation source
- `tests/`: package tests

## Install
```bash
pip install -U -e .
# Optional backends:
# pip install -e .[contraction]  # accelerated contraction search
# pip install -e .[torch]
# pip install -e .[solvers]
# pip install -e .[symmetry]
# pip install -e .[stabilizer]
# pip install -e .[vmc-torch]
# pip install -e .[vmc-netket]
# pip install -e .[layout]
# pip install -e .[mpi]       # MPI shot ensembles
# Optional plotting helpers:
# pip install -e .[viz]
```

## Quick Usage
```python
import pepsy
import quimb.tensor as qtn

ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=1, dtype="complex128")
ket_tagged, norm = pepsy.build_bra_ket(ket=ket)

bdy = pepsy.BdyMPS(tn_flat=ket_tagged, tn_double=norm, chi=32, single_layer=False)
res = pepsy.contract_boundary(norm=norm, bdy=bdy, direction="y", n_iter=2)

print(pepsy.__version__, res.cost)
```

## Symmetric Fermionic States

PePsY includes optional Symmray-backed symmetric tensor-network wrappers. For
spinful Fermi-Hubbard work, `model="fermi_hubbard"` uses total particle-number
`U1`, while `model="fermi_hubbard_u1u1"` uses spin-resolved `U1U1` charges
`(N_up, N_down)`.

For direct fermionic Fermi-Hubbard work, the main PePsY/Symmray methods
reference is Gao et al., "Fermionic tensor network contraction for arbitrary
geometries", Phys. Rev. Research 7, 023193 (2025),
https://doi.org/10.1103/PhysRevResearch.7.023193. It motivates keeping
fermionic parity and leg-order metadata in Symmray arrays while letting quimb
choose graph-level contraction orders.

The current finite-chain Fermi-Hubbard MPO convention and validation record are
tracked in `docs/development/notes/fermionic_mpo.md` and
`docs/development/fermi_hubbard_u1u1_mpo_notes.md`.

```python
import pepsy as py

psi = py.SymMPS.for_model(
    "fermi_hubbard_u1u1",
    16,
    bond_dim=4,
    site_charge=py.site_charge_from_occupations([(1, 0), (0, 1)] * 8),
)

assert psi.overall_charge() == (8, 8)

ordering = psi.fermionic_ordering()
assert ordering["enabled"]
assert ordering["methods_reference"]["doi"] == "10.1103/PhysRevResearch.7.023193"
```

## Documentation
Documentation is maintained as Markdown under `docs/`. No documentation
builder or documentation-specific package extra is required.

See [the API stability policy](docs/stability.md) for the distinction between
stable core modules and advanced domains.

Main docs sections:

- `getting_started`
- `tutorials/`
- `howto/`
- `api/`

## Notes
- `.gitattributes` marks notebooks as binary to avoid noisy diffs.
- `.gitignore` excludes checkpoints, generated caches, logs, and build output.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
pytest -q -o addopts=""  # include integration and slow suites
ruff check src tests
```
