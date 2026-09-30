# Package simplicity assessment — September 24, 2026

Status: assessment and proposals, not new agent policy or implemented refactors.
Inspected the current `develop` working tree at baseline `64f7e1f`, including
the existing uncommitted work. The goal is clearer imports, ownership, and
instructions while retaining numerical behavior and compatibility.

## What already works

- `src/pepsy`, `pyproject.toml`, responsibility-based namespaces, optional
  dependency extras, lazy imports, and wheel-install CI are already present.
- A fresh root import did not load NumPy, Quimb, SciPy, Torch, JAX, or Symmray.
  One local warm-filesystem observation took 0.076 seconds; this is not a
  portable performance benchmark or an installation-size measurement.
- The root compatibility registry has 309 names and is checked against a
  frozen manifest. These are exported names, not 309 duplicate implementations.
- Import/public-API/layout checks passed: **54 passed, 8 deprecation warnings**.
  This does not establish a clean numerical suite.
- Agent guidance already separates repository workflow, domain invariants,
  API documentation, proposals, and dated history. Keep that separation.

## Recommended order

| Priority | Concrete change | Acceptance criterion |
| --- | --- | --- |
| 1 | Make examples and navigation consistently teach owning namespaces. Add the missing `interop` ownership entry to README, AGENTS, and the layout guide. Clarify that advanced, optional, and experimental describe different properties. | Each capability has one recommended import; existing imports continue to work. |
| 2 | Add lightweight `__dir__` support to lazy public namespaces; check consistency between runtime exports and static typing declarations. | Fresh `dir(pepsy)` exposes advertised names without importing their implementations. |
| 3 | Migrate internal imports away from `pepsy.tensors.core` in small owning-subsystem changes. | Domain checks pass; external compatibility hooks remain valid; no new cycles or eager backend imports. |
| 4 | Consolidate repeated extra requirements and strengthen base-wheel CI with a tiny actual numerical operation. | Feature extras resolve correctly; a clean base installation performs the selected operation without undeclared optional dependencies. |
| 5 | Extract cohesive private responsibilities from the largest modules when working on their behavior. | Public classes/signatures stay stable and relevant numerical invariants pass; file-count reduction is not a target. |

### Public API and aliases

Prefer, for example, `from pepsy.optimizers import MpsOptimizer` and
`from pepsy.boundary import BdyMPS`. Keep documented root conveniences usable.
The [migration guide](../api-migration.md) already lists deprecated spellings
and promises no alias removal in the current 0.x line. Reduce their use in
new examples first. Removal requires a separately planned breaking release.

`pepsy.experimental` currently also routes to existing explicit domains such
as `pepsy.bp` and `pepsy.vmc`. Teach the owning namespace first and describe
the discovery facade separately. Optional installation does not itself mean
an unstable API; avoid inferring stability from an import path alone.

The root and sampled lazy domain facades have `__getattr__` but no `__dir__`.
In a fresh process, 321 names advertised in root `__all__` were absent from
`dir(pepsy)`. Adding discovery support is a smaller first improvement than
introducing another lazy-loading dependency or redesigning the exports.

### Internal ownership

`tensors/core.py` is a compatibility aggregator, yet callers in boundary,
fitting, operators, sampling, and optimizers still use it. Its wrappers also
preserve monkeypatch hooks by temporarily changing contraction-module globals.
This requires a caller-and-test audit rather than a blanket import replacement.
Prefer direct owning-module imports inside the implementation and public
domain imports in user examples.

There are 186 Python files. Large modules include MPS optimizer (12,482 lines),
symmetric tensors (11,904), stabilizer MPS (9,188), and BP series (8,420).
Line counts include comments and docstrings; they identify review candidates,
not proof of defects. Candidate seams include MPS stream planning/diagnostics
and symmetry model/state construction. Inspect state ownership and numerical
contracts before selecting an extraction; avoid a generic `utils` dumping ground.

### Installation weight

The base declares five direct dependencies. Installed development Quimb also
requires SciPy and Numba, among others, so lazy root imports do not imply a
small installed numerical stack. Measure the resolved clean environment before
claiming size savings. Do not remove direct requirements merely because an
upstream currently installs them transitively.

`torch` and `vmc-torch` currently repeat Torch; the combined `vmc` extra repeats
its component extras. Composed extras can reduce metadata drift while keeping
feature-oriented installation names. That improves maintenance, not necessarily
download size. Preserve contraction fallback and optimizer dependency contracts.
Current base-wheel CI checks namespace imports; add a deterministic calculation
to exercise the selected base path. No new installation was performed here.

### Agent guidance

Keep `AGENTS.md` as the short workflow/ownership entry point, the maintainer
skill as the router, domain skills as invariant owners, API docs as behavior
references, and history as evidence. Repair missing ownership and contradictory
examples before adding more instruction files. Existing catalog CI checks
structure; it does not prove semantic consistency. A future focused guard can
prevent new internal imports through compatibility aggregators while explicitly
tracking existing exceptions. Do not activate these proposals as agent tasks.

## Comparison with current public guidance

- [PyPA's src-layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
  explains isolation from repository-root files and installed-package testing.
  Pepsy already uses this structure.
- [Scientific Python SPEC 1](https://scientific-python.org/specs/spec-0001/)
  supports explicit lazy submodule exports and permits module `__getattr__`;
  Pepsy's mechanism is compatible with that approach. A new loader dependency
  is not necessary just to follow modern practice.
- [PyPA's pyproject guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
  documents feature extras and composing extras instead of copying requirements.
  Pepsy can apply this selectively without changing its build backend.

Recommendation: start with priorities 1 and 2; then migrate one internal
compatibility-import cluster at a time. Keep the existing package hierarchy.
