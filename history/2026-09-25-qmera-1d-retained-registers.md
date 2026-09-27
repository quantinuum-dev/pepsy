# 2026-09-25 — spin 1D retained-register qMERA

- Scope: bring the ml4mb-style 1D spin qMERA layout into Pepsy's main `QMeraBuilder` path, with explicit block length, boundary disentanglers, retained wires, repeated pair gates, and a path toward later 2D development.
- Branch / baseline commit: `develop` / `d0496b3` (local branch already ahead of `origin/develop` by five commits at task start).
- Commit status: Committed in the same qMERA commit as this entry (`HEAD` at handoff); not pushed. Pre-existing Tree layout edits were kept outside the commit.

## What changed

- `src/pepsy/optimizers/qmera/schedules.py` generates the unmoded spin 1D retained-register layout, including ternary odd tails, boundary pair windows, retention policies, and coarse-to-fine preparation order. It now offers `structure="brickwall"|"ladder"` independently for 1D isometry and boundary stages; ladder closes each block and repeats the complete ordered sequence for each circuit-depth unit. `QMeraBuilder(system_size=N)` names the 1D site count explicitly while retaining `shape=N` and tuple shapes for older and multidimensional callers.
- `gates.py` adds configurable composite Pauli pair gates, the five ml4mb global-X Z2 families, and an unrestricted 15-angle template; `builders.py` exposes `spin_symmetry=` and `pair_ansatz=` for spin 1D, selects the `z2_zz_yy_rx` template by default, and offers a separate `initial_state=` choice for `|0...0>` or `|+...+>`. The circuit symmetry and input-sector choices are documented separately. The ml4mb minimal template's shared initial angle value is retained without tying its trainable parameters.
- `lightcones.py` shares the input-state choice with direct, eager, and compiled local-energy paths. `QMeraPairSpec.rotation_sequence` exposes the exact chronological logical gates and pair-local wires. The preferred `pair_ansatz=` keyword and descriptive family names (`z2_zz_yy_rx`, `z2_rx_zz_yy_xx_rx`, `z2_zz_rx`, `z2_yz_zy`, `z2_all_paulis`, `all_paulis`) now replace vague names in the main API; `ansatz=` and all six prior family names remain accepted. Public exports, qMERA API docs, module map, changelog, and focused tests were updated.
- `docs/development/notes/qmera_1d_retained_registers_2026_09.md` records the design comparison and upstream API audit.

## Why

- The previous spin 1D Pepsy schedule coarse-grained every layer to one wire per block and used one RXX gate per pair. The ml4mb reference retains a configurable pair register and uses ordered two-qubit Pauli rotations. The new Pepsy path makes block and boundary structure explicit while leaving 2D and explicit-mode native fermion schedules in place.

## How it was validated

- `python -m pytest -q -o addopts='' tests/test_optimize_qmera.py` → 78 passed after cyclic-ladder schedule, dense-state, and direct/eager/compiled energy regressions; `tests/test_public_api.py tests/test_package_layout.py` → 58 passed; `python -m ruff check src tests` and `git diff --check` passed.
- The independent dense Pauli-circuit test checks the full 1D gate order and statevector for both brickwall and ladder. A depth-2 three-wire ladder and a wider boundary ladder verify closing gates, distinct parameter keys across repetitions, and agreement of direct, eager, and compiled energies. Symmetry tests verify all five ml4mb pair templates commute with global X and an unrestricted ZI rotation does not. Torch regressions confirm finite compiled gradients and compiled/direct energy agreement for Z2 and unrestricted templates. Separate JAX JIT and gradient probes passed for both. Earlier cross-checks against ml4mb at lengths 3, 4, and 7 agreed to approximately `4.5e-16`.
- An optional full repository run was stopped after 354 passed and 1 skipped, with no failures to that point. It was not a full-suite pass; the changed qMERA subsystem and public API suites completed.

## Decisions / findings

- The base input is `|0...0>` by default. `initial_state="plus"` (or the earlier `initial_hadamards=True` spelling) yields the ml4mb `|+>` input when appropriate for the Hamiltonian. All five ml4mb pair templates preserve global-X Z2; Pepsy's `unrestricted` template is an explicit extra choice that allows symmetry breaking.
- This task changes the main unmoded spin 1D builder. Native Symmray explicit-mode fermion and 2D geometry paths remain distinct, pending a separately designed 2D retained-register layout.

## Suggested next step (subject to current task scope)

- Use the validated 1D symmetry and block contract as the basis for a separately specified 2D retained-register layout.
