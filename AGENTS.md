# Pepsy agent guide

## Scope

These instructions apply to the whole Pepsy repository. Keep changes focused
on Pepsy itself; Gaugy, Tensy, and other sibling-package code belongs in its
own repository unless the user explicitly requests an integration artifact.

The main implementation is under `src/pepsy/`, tests under `tests/`, public
documentation under `docs/`, implementation notes under
`docs/development/`, and lightweight examples under `examples/`.

## Startup and safety

- Run `git status --short --branch` before editing. Existing changes belong to
  the user unless this task created them.
- Find nested instructions with `rg --files --hidden -g 'AGENTS.md'`.
- Read the closest focused tests before changing behavior. For API or import
  work, read `tests/test_public_api.py`, `tests/test_package_layout.py`, and
  `docs/development/package_layout.md`.
- Use `apply_patch` for source, test, and documentation edits. Keep temporary
  scripts and generated output under `/tmp`.
- Never include unrelated changes in a commit or push.

## Package architecture

Use responsibility-based namespaces for new code:

```text
pepsy.backends       backend selection, conversion, and linalg registration
pepsy.tensors        maps, constructors, contractions, observables, symmetry
pepsy.operators      gates, gate application, MPO/PEPO builders, Hamiltonians
pepsy.boundary       PEPS boundary states, sweeps, norms, and overlaps
pepsy.fitting        local tensor fitting
pepsy.solvers        gradient and finite-difference parameter solvers
pepsy.optimizers     MPS, MPO, PEPS, sweep, tree, MERA, and trajectory flows
pepsy.sampling       MPS, PEPS, vector, and tree samplers
pepsy.bp             belief propagation and loop/PNE methods
pepsy.vmc            optional Torch and NetKet/JAX VMC integrations
pepsy.experimental   one lazy namespace for advanced domains
pepsy._internal      private formatting and small utilities only
```

The old flat modules (`pepsy.core`, `pepsy.gates`, `pepsy.optimize_mps`, and
similar) and the duplicate `pepsy.extensions` namespace are removed. Do not
reintroduce them. Use `pepsy.experimental` for advanced-domain discovery and
the owning namespace for stable imports.

Keep public re-exports in the owning package `__init__.py`. Preserve the
top-level convenience API only when it is already documented and useful;
advanced implementation details should remain behind their domain namespace.

Before changing a specialized subsystem, read its skill:

- `pepsy.optimizers.mps` → `.github/skills/mps-optimizer/SKILL.md`
- `pepsy.optimizers.tree` → `.github/skills/tree-optimizer/SKILL.md`
- stabilizer TN → `.github/skills/stabilizer-tensor-networks/SKILL.md`
- tree stabilizer → `.github/skills/tree-stabilizer-optimizer/SKILL.md`
- belief propagation → `.github/skills/belief-propagation/SKILL.md`
- VMC → `.github/skills/pepsy-vmc/SKILL.md`
- fermion operators → `.github/skills/pepsy-fermion-operators/SKILL.md`
- qMERA → `.github/skills/qmera-energy-optimizer/SKILL.md`
- SymDMRG2 → `.github/skills/symdmrg2/SKILL.md`

Keep domain-specific invariants in those skills or their direct references;
do not duplicate them here.

## Dependency and backend rules

- Prefer public `quimb`, `cotengra`, `cotengrust`, and `autoray` APIs over
  local reimplementations.
- Keep Torch, JAX, CuPy, SciPy, NLopt, Stim, Symmray, and Nevergrad optional
  unless a feature genuinely requires them at package import time.
- `cotengrust` is currently required by Pepsy's accelerated contraction helper;
  only move it to an extra after adding and testing a cotengra fallback.
- `cmaes` is selected by the default cotengra optimizer string; do not remove
  it without changing and testing the default contraction path.
- Preserve backend, dtype, device, canonical-center, and tensor-network tag
  invariants. Emit explicit warnings for intentional compatibility coercions.
- Do not vendor upstream internals.

## Documentation and skills

- Keep user-facing API docs under `docs/api/` and concise implementation maps
  under `docs/development/modules/`.
- Keep design rationale and historical records under `docs/development/notes/`
  and `docs/development/plans/`.
- Each skill lives in `.github/skills/<name>/` with a concise `SKILL.md`; put
  large method notes or API maps in one-level `references/` files.
- Keep `.github/skills/README.md` synchronized with the skill directories.
- Do not retain references to removed benchmark scripts in active docs or
  skills. Performance harnesses belong outside the package repository.

## Python and validation

Always activate the shared Python 3.12 environment before Python, pytest, or
notebook commands:

```bash
source ~/envs/py312/bin/activate
```

Focused checks:

```bash
pytest -q tests/test_public_api.py tests/test_package_layout.py
pytest -q tests/test_prepare_boundary_inputs.py
pytest -q tests/test_gate.py
pytest -q tests/test_tensor_constructors.py
pytest -q tests/test_sampler.py
```

For optimizer or numerical changes, run the closest domain suite first. Use
`pytest -q -o addopts='' ...` for non-smoke coverage and
`pytest -q -o addopts=''` for the full suite when changes cross subsystems.
Run `python -m ruff check src tests` for the repository lint gate.

Keep the default smoke loop small. Prefer one deterministic regression for a
new invariant over broad Cartesian grids or duplicate end-to-end tests.

## Examples and handoff

Use public namespace imports in examples. Do not modify generated notebook
outputs unless explicitly requested. Finish with a concise summary of changed
files, validation performed, and remaining compatibility or dependency risks.
