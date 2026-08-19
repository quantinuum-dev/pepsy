# Contributing to Pepsy

Pepsy uses a `src/` layout and supports Python 3.10 and newer versions tested
by CI. Create an isolated environment, install the development profile, and
run the fast test suite before opening a change:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m pyflakes src tests
python -m ruff check src tests
```

The default suite excludes integration and stress tests to keep local
iteration short. Run the complete suite when changing an advanced domain:

```bash
python -m pip install -e ".[dev,test-extended]"
pytest -q -o addopts=""
```

Test tiers and domain markers are available when a narrower pass is useful:

```bash
pytest -q -o addopts="" -m "core and not optional"
pytest -q -o addopts="" -m optional
pytest -q -o addopts="" -m vmc       # replace with bp, tree, peps, or another domain
```

The `core and not optional` profile is the dependency-light API contract;
`core` without the exclusion includes stable APIs exercised through optional
backends. The full command includes all domain and slow tests.

To measure the local import boundary without importing an advanced domain:

```bash
python -X importtime -c "import pepsy" 2> /tmp/pepsy-importtime.txt
tail -n 1 /tmp/pepsy-importtime.txt
```

Use the smallest relevant optional profile when developing a backend:

- `.[contraction]` for accelerated Cotengra path search;
- `.[symmetry]` for Symmray-backed tensors;
- `.[stabilizer]` for Stim-backed stabilizer workflows;
- `.[vmc-torch]` or `.[vmc-netket]` for VMC integrations;
- `.[solvers]` and `.[layout]` for external solver and layout backends.

Keep the stable API small and add new advanced functionality under its
responsibility-based module or `pepsy.experimental`. Add a regression test for
observable behavior and update the handwritten Markdown documentation when a
public API changes. Deprecated imports should emit `DeprecationWarning` and
remain functional during the documented compatibility window.

Keep algorithm tests focused: add one deterministic regression test for the
changed behavior, plus a boundary or invariant case only when it covers a
distinct failure mode. Avoid large parameter matrices, repeated seeds, and
duplicate integration tests. Mark optional-backend, differential, and stress
coverage as `integration` or `slow`; do not make it part of the default loop.

Do not commit generated caches, build output, notebook execution artifacts, or
local environment files. For numerical changes, include the backend, dtype,
seed, and tolerance assumptions in the test or documentation.
