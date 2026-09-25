# Pepsy repository instructions

These instructions govern work in this repository. Pepsy owns the code under
`src/pepsy/`; sibling projects such as Gaugy and Tensy have their own scope.

## Start here

1. Read any device-local `AGENTS.override.md`, this file, `README.md`, and
   `CONTRIBUTING.md`. A local environment override takes precedence over
   environment examples in shared documentation. Never stage or commit the
   device-local override.
2. Run `git status --short --branch`. Preserve existing changes and keep this
   task's edits distinguishable. Use `apply_patch` for tracked-file edits.
3. Find applicable nested instructions with
   `rg --files --hidden -g 'AGENTS.md' -g 'AGENTS.override.md'`. Read those
   governing the files you will touch.
4. Use the [maintainer router](.github/skills/pepsy-maintainer/SKILL.md) to
   select the relevant domain skill. Read the affected code, closest tests,
   and handwritten API documentation before changing behavior.
5. Read the [session journal guide](history/README.md) and the newest one to
   three entries relevant to the task. Check their claims against Git status
   and current code; suggested next steps do not expand the user's request.

Follow the user's requested scope. These files do not authorize unrelated
refactors, dependency upgrades, sibling-repository edits, commits, or releases.
Use existing session authorization; do not add redundant approval steps.

## How to interpret repository guidance

| Source | Purpose |
| --- | --- |
| `AGENTS.md` and applicable local/nested instructions | Workflow, scope, environment, and repository-wide constraints |
| [.github/skills/](.github/skills/README.md) | Task routing and maintained subsystem invariants |
| `docs/api/`, public exports, focused tests | Public behavior and its checks; confirm against the implementation |
| `docs/development/modules/` | Implementation ownership and navigation |
| [history/](history/README.md) | Dated session handoffs, validation results, and unfinished work; not active policy |
| `docs/development/notes/`, `plans/`, benchmark records | Dated evidence, rationale, and proposals; not an implementation task list |
| Installed upstream source and signatures | Capabilities of the environment actually executing this task |

A plan or changelog does not prove that a feature is integrated or validated.
Distinguish **implemented**, **proposed**, **measured**, and **unverified** claims.
When code, tests, and documentation disagree, investigate the discrepancy;
do not silently rewrite numerical behavior to match a stale note or weaken a
test to match a regression. Record remaining uncertainty.

## Python and Environment

Before Python, tests, linters, builds, or package commands, activate the
environment specified by the current session or device-local override in the
same shell. Use its interpreter. Do not switch environments because a skill
contains an old example path.

If no environment is specified, use the project's existing development
environment. For a new checkout, follow the virtual-environment setup in
[CONTRIBUTING.md](CONTRIBUTING.md). Do not modify a shared environment merely
to inspect a newer dependency; use isolated probes when appropriate.

Commands below use `python` to mean that activated interpreter. Run from the
repository root. Keep temporary scripts and generated diagnostic output under
`/tmp`; generated documentation under `docs/_build/` is not source material.

## Package ownership

Use the existing responsibility-based namespaces:

| Namespace | Responsibility |
| --- | --- |
| `pepsy.backends` | Backend selection, conversion, and linear algebra registration |
| `pepsy.tensors` | Maps, constructors, contractions, observables, and symmetry |
| `pepsy.operators` | Gates, application helpers, MPO/PEPO builders, Hamiltonians |
| `pepsy.boundary` | PEPS boundary states, sweeps, norms, and overlaps |
| `pepsy.fitting` | Local tensor fitting |
| `pepsy.interop` | Adapters for external circuit and tensor-network representations |
| `pepsy.solvers` | Parameter and finite-difference solvers |
| `pepsy.optimizers` | MPS, MPO, PEPS, tree, qMERA, stabilizer, and trajectory orchestration |
| `pepsy.sampling` | MPS, PEPS, vector, and tree samplers |
| `pepsy.bp` | Belief propagation, loop corrections, and PNE |
| `pepsy.vmc` | Optional Torch and NetKet/JAX VMC integrations |
| `pepsy.experimental` | Lazy discovery of advanced domains |
| `pepsy._internal` | Small private utilities, not a home for domain algorithms |

Place a change with the subsystem that owns its behavior. Reuse an existing
helper before adding another abstraction. Keep backend conversion out of
algorithm-specific copies and keep optimizer orchestration out of tensor
constructors. Package restructuring is a separate task, not an automatic
consequence of finding a large module.

Keep public exports in the owning package's `__init__.py`; preserve documented
top-level conveniences. Do not restore removed flat modules such as
`pepsy.core`, `pepsy.gates`, or `pepsy.optimize_mps`, or the old
`pepsy.extensions` namespace. See the
[package layout](docs/development/package_layout.md) for import examples.
Use public namespace imports in examples. Teach the owning namespace first;
`pepsy.experimental` is an optional discovery facade, not a second owner.
Installation extras and API stability are separate concerns; use the
[stability policy](docs/stability.md) for compatibility guarantees.

## Numerical and dependency contracts

- Preserve backend, dtype, device, gradients, fermionic ordering, charge
  sectors, canonical-center metadata, and tensor tags on affected paths.
  Preserve exact targets and the requested truncation/normalization policy.
- Prefer public Quimb, Cotengra, Cotengrust, and Autoray APIs. Keep optional
  libraries optional at import time. Check dependency declarations in
  `pyproject.toml` and the actual callers before changing extras.
- Cotengrust is used by accelerated contraction; provide and test a Cotengra
  fallback before making that path independent of it. The default Cotengra
  optimizer selects CMA-ES; preserve its dependency or change and test the
  optimizer policy together.
- Warn about intentional compatibility coercions. Approximate algorithms must
  remain explicit choices, with capability checks and accuracy validation.
- Read the relevant [shared numerical contracts](.github/skills/pepsy-maintainer/references/numerical-contracts.md)
  for Torch SVD/QR registration, cyclic CTMRG, or downstream comparisons.
  Tree QR/TreeMPO, MPS permutation, stabilizer naming, fermion, and sampling
  contracts live in the domain skills linked by the maintainer router.

## Upstream compatibility work

Before changing contraction, compression, canonicalization, gating, layout,
backend, or Symmray behavior, check the
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
and [repository](https://github.com/jcmgray/symmray).

At the start of each relevant maintenance task, record installed versions and
inspect the signatures and dispatch capabilities used by the change. Recheck
when dependencies change or a failure points to upstream behavior. An audit
already performed for the same active task and unchanged environment can be
reused. If a page is unavailable, record that and use official source/release
notes plus the installed implementation. Documentation-only reorganization
does not require a numerical upstream audit.

Classify findings as **adopt**, **compatibility shim**, **prototype**, or
**defer**. Shims must be narrow, scoped, capability-gated, and regression-tested;
never edit installed libraries or vendor their internals. Validate dense,
native Symmray, and relevant backend paths separately when affected. Record
dated evidence in `docs/development/notes/`; update API docs and
`CHANGELOG.md` when behavior changes.

## Validation and completion

Choose checks by the change, after activating the selected environment:

| Change | Required validation |
| --- | --- |
| Instructions, skills, or documentation only | Relevant link/catalog checks and `git diff --check`; no numerical suite unless behavior also changes |
| Public API, imports, or package layout | `python -m pytest -q tests/test_public_api.py tests/test_package_layout.py` |
| Numerical algorithm or optimizer | Closest domain suite with `python -m pytest -q -o addopts='' <test paths>`; include meaningful reconstruction, reference, or gradient checks |
| Changes across implementation subsystems | Closest suites first, then `python -m pytest -q -o addopts=''` |
| Python implementation or tests | `python -m ruff check src tests` and `git diff --check` |

Use deterministic regressions that test observable behavior. Keep optional,
integration, and slow checks out of the default smoke loop; use
`CONTRIBUTING.md` for test profiles. Report failures and missing backends
explicitly. Do not infer full-suite success from a passing selection.

Keep handwritten API docs under `docs/api/`, implementation maps under
`docs/development/modules/`, and historical evidence under notes/plans.
For skill changes, follow [.github/skills/SKILL_POLICY.md](.github/skills/SKILL_POLICY.md)
and synchronize the [catalog](.github/skills/README.md) and
[upload manifest](.github/skills/agent-bundle.yaml). Do not put machine paths
or current benchmark numbers in this entry point.

Preserve generated notebook outputs unless requested otherwise. Before
handoff, review the diff for unrelated changes. State what changed, validation
performed, unresolved limitations, and whether anything was committed or
published.
For substantive work, append a dated handoff in `history/` using its guide.
Distinguish committed changes from working-tree edits and new checks from
earlier results. Link detailed evidence rather than duplicating it.
