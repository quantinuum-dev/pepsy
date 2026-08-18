# Lightweight package plan

Status: active, phases 0-3 complete
Last updated: 2026-08-18
Owner: Pepsy maintainers

This plan reduces installation, import, and validation weight while preserving
the existing stable namespaces and numerical behavior. It deliberately starts
with measurement and contracts; dependency moves happen only after a tested
fallback exists.

## Baseline

At commit `cc4b2d9` the repository contains approximately:

- 9.6 MB of tracked files;
- 169k Python lines under `src/pepsy`;
- 74k lines under `optimizers`, 23k under `bp`, and 23k under `vmc`;
- 1,800 test declarations across 48 test files.

The package has five direct mandatory dependencies: NumPy, Quimb, Cotengra,
Autoray, and tqdm. Torch, JAX/NetKet, Symmray, Stim,
SciPy/NLopt, Nevergrad, Guppy, and plotting dependencies are already optional
extras.

The top-level `pepsy` facade, core domain facades, and `pepsy.experimental`
use lazy imports. Explicit regression tests prevent optional dependencies from
leaking into the core import path.

## Dependency audit — 2026-08-18

The first static audit found:

- `cotengra` is directly imported by the shared contraction implementation
  and the advanced qMERA compiled-contraction implementation;
- `cotengrust` is imported when Pepsy constructs a reusable contraction
  optimizer, where the current implementation intentionally requires its
  accelerated pathfinder;
- `cmaes` is selected by the current default Cotengra optimizer string rather
  than imported directly by Pepsy.

Cotengra remains mandatory because it is part of the shared Quimb contraction
route. The first safe reduction is now implemented: Cotengrust is best-effort,
and missing CMA-ES falls back to Cotengra's built-in `sbplx` search. Both
acceleration packages are available through the `[contraction]` extra. A
future removal of Cotengra itself still requires a separate provider boundary,
fallback implementation, and benchmark campaign. qMERA can be isolated
behind its existing advanced namespace independently, but that does not by
itself remove shared runtime dependencies from the base installation.

The current base metadata consequently has five mandatory dependencies; the
`contraction` extra restores the previous accelerated-search profile.

### Transitive runtime weight

The direct requirement count is not the same as the clean-install footprint.
Quimb's current metadata brings in `autoray`, `cotengra`, `cytoolz`, `numba`,
`numpy`, `psutil`, `scipy`, and `tqdm` for its standard tensor-network route.
Cotengra is therefore both a direct Pepsy dependency and a Quimb runtime
dependency. This explains why the base wheel is smaller and the import path is
lighter without claiming that a base installation is numerically lightweight.

The next material reduction would require replacing or isolating Quimb's
standard contraction route, or upstreaming a supported lighter Quimb profile.
That is a separate benchmark and compatibility project; it is not safe to
solve by merely moving `cotengra` to an extra while keeping the same Quimb
runtime path.

Core namespace facades now resolve their implementation modules lazily, so
importing `pepsy.backends`, `pepsy.boundary`, `pepsy.fitting`, `pepsy.interop`,
`pepsy.solvers`, or the other stable domains does not load Torch, SciPy, JAX,
Symmray, Stim, NetKet, or Nevergrad. The accelerated implementation remains
available when its public symbol is requested.

## Target architecture

Keep one source repository and one `pepsy` public package initially. Organize
the product into two dependency layers:

### Core

The core should cover NumPy tensor construction, operators, boundary methods,
basic contraction, fitting, sampling, and the stable MPS/MPO/PEPS workflows.
Its import path must not load Torch, JAX, Symmray, Stim, NetKet, NLopt, or
Nevergrad.

### Advanced domains

Keep these behind explicit modules and opt-in extras:

- belief propagation;
- tree tensor networks and QMERA;
- stabilizer tensor networks;
- Symmray and fermionic workflows;
- Torch/JAX/NetKet VMC;
- specialized solver and layout integrations.

Existing canonical namespaces remain valid. This is a dependency and loading
boundary, not an immediate package rename or compatibility break.

## Phases

### 1. Import and test contracts

- Add tests proving a fresh `import pepsy` does not import optional backends.
- Add tests proving documented core namespace imports do not import optional
  backends.
- Add focused optional-dependency error tests with actionable install hints.
- Replace broad file-based test classification with explicit core/domain
  markers where practical.
- Keep the default loop small and deterministic; retain extended coverage.

### 2. Dependency boundary

- Measure which core APIs truly require Cotengra, Cotengrust, and CMA-ES.
- Add a supported contraction fallback before moving any of them to an extra.
- Keep backend-specific dependencies lazy at module and function boundaries.
- Add minimal-install and wheel smoke checks to CI.

### 3. Test tiers and scheduled validation

- Tag tests with `core`, `optional`, and responsibility-based domain markers.
- Keep the default `smoke` profile deterministic and fast.
- Run the complete available extended suite, including slow tests, nightly.

### 4. Domain decomposition

- Keep stable MPS/MPO/PEPS code cohesive.
- Reduce cross-imports from core into BP, tree, QMERA, stabilizer, and VMC.
- Move shared protocols and small utilities into responsibility-based modules.
- Consider separate distributions only if dependency or release measurements
  show that extras are insufficient.

## Acceptance criteria

- Core installation succeeds without advanced extras.
- `import pepsy` and documented core imports do not import optional stacks.
- The core smoke suite has a predictable short runtime.
- Each advanced domain has an explicit dependency profile and test command.
- Existing stable import paths and numerical invariants remain unchanged.
- No coverage is removed merely to lower the test count; duplicate coverage is
  reduced only when an equivalent invariant test remains.

## Guardrails

- Do not split the repository into multiple packages during the first pass.
- Do not move Cotengra/Cotengrust/CMA-ES until the fallback is implemented and
  tested.
- Do not change algorithm defaults while reducing package weight.
- Do not remove public symbols from the top-level facade in this effort.
- Keep the work on `develop` and make each phase independently reviewable.
