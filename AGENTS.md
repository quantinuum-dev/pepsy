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

## Native fermionic tree QR policy

The native `TreeTensorNetwork` QR policy is centralized in
`TreeTensorNetwork._native_qr_split` and `_native_qr_options`:

- Every lossless QR split on native Symmray tree tensors must use
  `_native_qr_split`; do not add a direct `tensor.split(method="qr")` call to a
  native tree route.
- The helper sets `stabilized=False` only for Symmray block-sparse tensors.
  Symmray's stabilized QR phase-normalizes each diagonal of `R`; an exact
  structural-zero diagonal makes that phase `0 / |0|`, which can become NaN in
  `complex64`. Dense tensors retain Quimb's normal stabilized-QR default.
- Network-level canonicalization, which does not expose one tensor at a time,
  obtains the same option from `_native_qr_options()` when the tree is
  fermionic. Keep this policy aligned if another native canonicalization route
  is added.
- Skipping the phase convention is lossless: `Q @ R` is unchanged, and the
  resulting `left_inds` isometry metadata remains valid. Native truncating
  compression still uses the explicit graded SVD and its configured cutoff.
- This safeguard is scoped to `TreeTensorNetwork` / `TreeOptimizer`. It does
  not change the separate `MpsOptimizer` QR implementation or globally patch
  Quimb/Symmray.

## Native TreeMPO contract

The tree-native operator API lives in `pepsy.optimizers.tree.operators`:

- `TreeMPO` is the primary tree measurement object. Prefer
  `TreePlan.to_tree_mpo(...)` or `Fermion.to_tree_mpo(..., tree=plan)` when the
  consumer is a `TreeTensorNetwork`.
- `TreePlan.to_mpo(...)` and `tree_mpo(...)` remain compatibility constructors
  for the ordinary low-bond chain MPO. They attach the `TreeMPO`, but the
  chain MPO is never moved into the tree, densified, or used as the tree
  operator during `expectation_mpo_exact`.
- Neutral native term sums are factorized from their native Symmray tensors on
  each term's TreePlan Steiner subtree and amalgamated into one direct-sum
  TTNO. Do not replace this with a Jordan--Wigner dense factorization or a
  list of ordinary hyperedges. The compact eta-pair observable is an explicit
  structured exception with its four-state TTNO automaton.
- `TreeMPO.expectation(...)` and `TreeTensorNetwork.expectation_mpo_exact(...)`
  contract separate bra, operator, and ket networks. `TreeMPO.canonicalize()`
  is lossless native QR gauge fixing; `TreeMPO.compress(...)` is the explicit
  native graded SVD truncation stage.
- Native operator QR must use the shared `_native_qr_split_tensor` policy (and
  therefore the same `stabilized=False` structural-zero safeguard as the
  state). Do not add direct `tensor.split(method="qr")` calls to TreeMPO
  canonicalization.

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

## Upstream compatibility audit

Pepsy is tightly coupled to the Quimb tensor-network stack. Before changing
any contraction, compression, canonicalization, gating, layout, backend, or
Symmray path, check all of the following upstream sources, even when the
reported issue initially appears to involve only one dependency:

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
- [Autoray repository](https://github.com/jcmgray/autoray)
- [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/#) and
  its [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html)
- [Symmray Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
  and the [Symmray repository](https://github.com/jcmgray/symmray)

Perform this audit at the start of each relevant maintenance task, whenever a
dependency or development checkout changes, and again when a failing test may
be caused by an upstream API or numerical behavior change. In the same active
Pepsy environment, record the installed versions and inspect the actual
callable signatures and dispatch tables for every upstream API being used;
documentation and release headlines alone are not sufficient, especially for
development versions.

For each upstream change, classify the result as **adopt**, **compatibility
shim**, **prototype**, or **defer**. Preserve Pepsy's existing defaults,
ordering, backend and metadata invariants unless the user explicitly requests
a breaking change. New upstream algorithms must be opt-in and capability
gated rather than selected solely by a version string. Compatibility shims
must be narrow, in-memory, covered by a regression test, and must not edit
installed packages or vendor upstream implementation. If an upstream change
affects more than one layer, validate dense, native Symmray, and relevant
Autoray backend paths separately, and run the closest Quimb/Cotengra
contraction regression before broadening the change.

Keep the audit result and any deferred opportunities in the relevant
`docs/development/notes/` file, including the audit date, installed versions,
API probes, affected Pepsy paths, and focused tests. Update the handwritten
API documentation and `CHANGELOG.md` whenever an upstream-facing behavior or
new opt-in compatibility path becomes part of Pepsy.

## Torch SVD/QR policy

Keep Torch linear-algebra registration behind the single public
`pepsy.TorchLinalgConfig` class. Its `register()` operation configures the
Autoray SVD and QR rules together and can additionally configure Quimb's raw
Symmray split drivers. Do not add a new optimizer call that independently
combines `reg_*_torch` helpers or `register_torch_linalg(...)` arguments.

- `stabilized=False` is the native, fastest default for ordinary simulation.
  `stabilized=True` is the autodiff policy: it installs Pepsy's
  relative-regularized SVD VJP and the configured rank-aware QR behavior for
  difficult or rank-deficient tensor-network splits.
- `svd_driver="gesvdj"` and `"gesvd"`, plus CPU Torch/Scipy `gesdd`/`gesvd`,
  are non-approximate choices. `gesvda` is approximate and must remain behind
  `allow_approximate=True`.
- `quimb_split_drivers=True` is required when raw Torch Symmray blocks enter
  Quimb's `svd_truncated` or `qr_stabilized` paths, because those blocks bypass
  ordinary Autoray dispatch. PEPS optimizers should enable it automatically
  when Symmray data is detected.
- `PepsOptimizer` accepts `torch_linalg_config=TorchLinalgConfig(...)` and
  should use that policy for global Torch cleanup. The legacy
  `register_torch_svd` switch remains only for compatibility.
- Preserve dtype/device behavior: driver selection changes the underlying
  decomposition, not `complex64`/`complex128` promotion. Add a focused
  reconstruction or gradient test when changing a backend route.

## Cyclic CTMRG compatibility

- Use `pepsy.boundary.quimb_ctmrg_projector_compat` around Quimb CTMRG calls
  for cyclic PEPS/PEPO networks whose effective bond dimensions can vary,
  especially native U(1) term-by-term replays.
- This is a scoped compatibility context: it redirects projector insertion to
  the current network and restores Quimb's method on exit. It does not modify
  installed `site-packages`, alter boundary-MPS contractions, or replace
  CTMRG with MPS.
- Keep the workaround at the Pepsy boundary API. Do not copy Quimb's CTMRG
  implementation into Pepsy or edit the installed Quimb source. Add a focused
  regression test when changing the compatibility behavior.
- The shared CTMRG entry points already apply this context for
  `contract_flat(..., method="ctmrg")`, native Torch PEPS VMC models with
  `contraction="ctmrg"`, and the NetKet/JAX PEPS amplitude validation path.
  Keep exact, HOTRG, and boundary-MPS routes independent of this workaround.

## Documentation and skills

- Keep user-facing API docs under `docs/api/` and concise implementation maps
  under `docs/development/modules/`.
- Keep design rationale and historical records under `docs/development/notes/`
  and `docs/development/plans/`.
- Each skill lives in `.github/skills/<name>/` with a concise `SKILL.md`; put
  large method notes or API maps in one-level `references/` files.
- Keep `.github/skills/README.md` synchronized with the skill directories.
- Start cross-cutting work with `.github/skills/pepsy-maintainer/SKILL.md`.
- Follow `.github/skills/SKILL_POLICY.md` when adding, updating, deprecating,
  merging, or removing skills. Run the catalog validator for those changes.
- Use `.github/skills/agent-bundle.yaml` as the canonical Workspace Agent
  upload map. Upload `SKILL.md` and relevant `references/**`; keep
  `agents/openai.yaml` for local skill metadata rather than agent context.
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
