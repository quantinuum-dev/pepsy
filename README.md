# Pepsy Library

<img src="assets/pepsy-icon.svg" alt="pepsy icon" width="220" />

`pepsy` is a Python package for circuit simulation and related DMRG fitting workflows.

Current package version: `0.1.1` (from `pyproject.toml` / `pepsy.__version__`).

## Package Layout
- `src/pepsy/`: installable library code
  - `backends/`: backend selection, conversion, and linear algebra registration
  - `tensors/`: lattice maps (`OneDMap`), state/MPO/PEPO builders, contraction helpers
  - `operators/`: gates, gate application, MPO/PEPO builders, and Hamiltonian helpers
  - `boundary/`: boundary state initialization (`BdyMPS`), sweeps (`CompBdy`), and metrics
  - `solvers/`: gradient-based and finite-difference solvers
  - `fitting/`: local tensor fitting routines (`FIT`)
  - `optimizers/`: sweep, global, energy, MPS, and MPO optimizers
  - `sampling/`: `MpsSampler` and related sampling utilities
  - `_internal/`: private formatting and utility helpers
- `examples/`: runnable examples (e.g. `MpsMagnetization/` 2D ITF Trotter MPS evolution)
- `docs/`: Sphinx documentation source
- `tests/`: package tests

## Install
```bash
pip install -U --no-deps -e .
# Optional backends:
# pip install -e .[torch]
# pip install -e .[solvers]
# jax backend (manual, platform-specific wheels):
# pip install jax jaxlib
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

## Documentation
Build docs locally:

```bash
pip install -e .[docs]
NUMBA_CACHE_DIR=/tmp PYTHONPYCACHEPREFIX=/tmp \
sphinx-build -W -b html docs docs/_build/html
```

Main docs sections:

- `getting_started`
- `tutorials/`
- `howto/`
- `api/`

## Notes
- `.gitattributes` marks notebooks as binary to avoid noisy diffs.
- `.gitignore` excludes checkpoints, caches, `cash/`, and `nohup.out`.
