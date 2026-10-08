# Follow-up PEPS implementation review

User requested another review/test pass after stopping production PEPS.
Branch develop, baseline 0ba2a05; existing edits remain uncommitted. Applied
the existing maintainer, fitting, and BP instructions; reused the upstream
audit from this unchanged maintenance task.

Reproduced and fixed two edge cases in PepsOptimizer: custom final
normalization overriding selected chi with the starting cap, and failed
zero-norm probes handing off missing/stale boundary handles. Added regression
tests, asymmetric-gate/site-order tests, and three-gate cached/fresh comparisons
across both orientations and direct/DMRG boundary modes.

Focused selection: 99 passed. Bounded CUDA:0 3×3 D2 retest completed; cached
and uncached sweep exact fidelities agree within 4e-14, outputs normalized.
Details are in the [review note](../docs/development/notes/2026-10-08-peps-boundary-reuse-full-update.md).
Ruff/Pyflakes unavailable. Production PEPS was not restarted; no commit/push.

Broader PEPS/API/layout rerun: 338 passed, one existing installed-version
metadata mismatch (installed 0.4.0 versus checkout 0.5.0). Compilation and
diff checks passed. No whole-repository pass is claimed for this follow-up.
