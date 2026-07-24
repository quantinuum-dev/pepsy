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
```

Documentation is plain Markdown under `docs/`; no documentation builder is
needed for package development or installation.
