# Pepsy Library

<img src="assets/pepsy-icon.svg" alt="pepsy icon" width="220" />

`pepsy` is a Python package for circuit simulation and related DMRG fitting workflows.

Current package version: `0.1.1` (from `pyproject.toml` / `pepsy.__version__`).

## Package Layout
- `src/pepsy/`: installable library code
  - `boundary_states.py`: boundary state initialization (`BdyMPS`)
  - `boundary_sweeps.py`: sweep/contraction runner (`CompBdy`)
  - `boundary_metrics.py`: input preparation + contraction (`build_bra_ket`, `contract_boundary`, `BoundaryContractResult`)
  - `gates.py`: gate application (`gate`, `gate_simple`, `renorm_gauge`) with long-range SWAP routing
  - `core.py`: lattice maps (`OneDMap`), state/MPO/PEPO builders, backend defaults
  - `ham.py`: Hamiltonian helpers
  - `optimize_sweep.py`, `optimize_global.py`, `optimize_energy.py`: optimizers
  - `optimize_mps.py`, `optimize_mpo.py`: `MpsOptimizer`, `MpoOptimizer`
  - `gradient_solver.py`, `ft_solver.py`: gradient-based and FT/FD solvers
  - `sampler.py`: `MpsSampler` and related sampling utilities
  - `fit.py`, `_backend_utils.py`, `_backend_linalg.py`, `_backend_linalg_torch.py`, `_backend_linalg_jax.py`
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
