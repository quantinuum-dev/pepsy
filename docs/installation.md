# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e .[torch]
pip install -e .[solvers]
pip install -e .[viz]
pip install -e .[symmetry]
pip install -e .[stabilizer]
pip install -e .[layout]
```

Documentation sources are plain Markdown under `docs/`. The generated API
reference is optional and is built with Sphinx:

```bash
source ~/envs/py312/bin/activate
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

The build reads the source tree statically through AutoAPI, so optional
Torch, JAX, NetKet, Stim, and Symmray integrations do not need to be enabled
just to generate the API navigation.
