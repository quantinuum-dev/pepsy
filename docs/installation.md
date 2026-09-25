# Installation

Pepsy requires **Python 3.12 or newer** for the base package and every extra.
Install it from the GitHub repository; it is not published on PyPI.

## Install from a checkout

Clone the tagged release if you do not already have a checkout:

```bash
git clone --branch v0.5.0 https://github.com/quantinuum-dev/pepsy.git
cd pepsy
```

From a Pepsy checkout, install the base package in your selected environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development, use `python -m pip install -e ".[dev]"` instead. Respect any
existing local environment override when working in a maintained checkout.

The base dependencies are NumPy, Quimb, Cotengra, Autoray, and tqdm; these
may install further dependencies. See the
[dependency audit](development/notes/dependency_minimums_2026_09.md) for
minimum versions and tested combinations.

## Optional features

Install only the features you use. These commands are alternatives, not a
sequence to run in full:

| Feature | Install from checkout |
| --- | --- |
| Torch arrays and autodiff | `python -m pip install ".[torch]"` |
| Torch VMC | `python -m pip install ".[vmc-torch]"` |
| NetKet/JAX VMC | `python -m pip install ".[vmc-netket]"` |
| Both VMC integrations | `python -m pip install ".[vmc]"` |
| Symmetry and fermions | `python -m pip install ".[symmetry]"` |
| Stabilizer simulation | `python -m pip install ".[stabilizer]"` |
| Accelerated contraction search | `python -m pip install ".[contraction]"` |
| SciPy/NLopt solvers | `python -m pip install ".[solvers]"` |
| Layout search | `python -m pip install ".[layout]"` |
| Plotting | `python -m pip install ".[viz]"` |
| MPI execution | `python -m pip install ".[mpi]"` |
| Guppy interoperability | `python -m pip install ".[guppy]"` |

`vmc-torch` reuses `torch`; `vmc-netket` includes `symmetry`; `vmc` combines
both VMC profiles. `test-extended` combines optional test dependencies and is
intended for contributors, including Autograd for its explicit autodiff tests.

The NetKet profile currently requires JAX below 0.11.1: released NetKet 3.22.x
fails during import with the newer `ArrayLike` type annotation. This bound
also applies to the combined `vmc` extra and can be revisited after an upstream
fix to its [online statistics annotations](https://github.com/netket/netket/blob/v3.22.4/netket/_src/stats/online_stats/operations.py).

Combine extras when needed, and add `-e` for an editable development install:

```bash
python -m pip install -e ".[torch,symmetry]"
```

## Build the documentation

Read the Markdown in `docs/` directly, or build a searchable HTML site with
generated API pages:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The build reads the source tree statically through AutoAPI, so optional
Torch, JAX, NetKet, Stim, and Symmray integrations do not need to be enabled
just to generate the API navigation.
