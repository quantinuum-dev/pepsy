# Pepsy Library

<img src="assets/pepsy-icon.svg" alt="pepsy icon" width="220" />

`pepsy` is a Python package for circuit simulation and related DMRG fitting workflows.

Current package version: `0.1.1` (from `pyproject.toml` / `pepsy.__version__`).

## Package Layout
- `src/pepsy/`: installable library code
  - `boundary_states.py`: boundary state initialization (`BdyMPS`)
  - `boundary_sweeps.py`: sweep/contraction runner (`CompBdy`)
  - `boundary_metrics.py`: input preparation + contraction (`prepare_boundary_inputs`, `ContractBoundary`)
  - `optimize_sweep.py`, `optimize_global.py`, `gate.py`, `gradient_solver.py`, `debug.py`
  - `fit.py`, `core.py`, `linalg_registrations.py`
- `example/`: example notebooks
- `docs/`: Sphinx documentation source
- `tests/`: package tests

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional backends:
# pip install -e .[torch]
# pip install -e .[solvers]
# Optional plotting helpers:
# pip install -e .[viz]
```

## Quick Usage
```python
import pepsy
from pepsy import BdyMPS, CompBdy, ContractBoundary, prepare_boundary_inputs

print(pepsy.__version__)
```

## Documentation
Build docs locally:

```bash
pip install -e .[docs]
sphinx-build -W -b html docs docs/_build/html
```

Main docs sections:

- `getting_started`
- `tutorials/`
- `howto/`
- `api/`

Guided notebook example:

- `example/norm.ipynb`

## Notes
- `.gitattributes` marks notebooks as binary to avoid noisy diffs.
- `.gitignore` excludes checkpoints, caches, `cash/`, and `nohup.out`.

## TODO: Symmetry Roadmap (`symmray`)
Potential enhancement: evaluate integrating [`symmray`](https://github.com/jcmgray/symmray) for symmetry-aware tensor workflows (starting with Z2 in 1D MPS simulation).

- [ ] Feasibility study: map current `MpsOptimizer` / `MpoOptimizer` tensor assumptions to `symmray` block-sparse arrays.
- [ ] Prototype a minimal Z2-conserving 1D path (`gate_tn_1d`, `build_mpo_from_gates`, `expec_mpo`).
- [ ] Define index/charge conventions (site ordering, parity sectors, bra/ket index layout).
- [ ] Benchmark memory/runtime vs dense backend on representative circuits.
- [ ] Add optional dependency group (e.g., `.[symmetry]`) without affecting default installs.
- [ ] Add tests that compare symmetric vs dense results on small systems.
- [ ] Document limitations (supported gates, allowed charge sectors, fallback paths).
