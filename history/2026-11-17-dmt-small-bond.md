# DMT rank-aware regions and small bonds

Working-tree implementation: rank-aware selection, contained-region reuse,
and an explicit weighted-SVD relaxed fallback with trace and density-error
reporting. The original strict mode remains the default. The direct constrained
local-density optimizer and CuPy validation remain
unfinished. Existing unrelated sampling changes are preserved.

The opt-in `check_regions=False` skips full-region verification but retains
bond and trace guards. Density-error report fields are then unavailable.
The option is available on ordinary and Pauli MPO entry points and as
`dmt_check_regions` in replay. A fixed 6×6 rank-136 input at χ=8 took 2.156 s
with full checks and 1.335 s with fast checks in single CPU runs; the full
check measured 0.0586 maximum relative plaquette error. These are not a
statistical performance result or a long-time accuracy claim.

The 27 new focused tests pass; the optional CuPy case skips because CuPy is
not installed. Ruff, the catalog validator and `git diff --check` pass. The
full suite stops at the previously documented BP Autoray namespace error
after 104 passes and seven skips. The direct-density constrained optimizer
remains a proposal; the relaxed fallback uses a weighted protected-direction
surrogate, followed by a trace correction, and is not that optimizer.
The expanded DMT regression selection passed 354 tests with one expected
failure before the final routed fast-check adjustment; the focused selection
subsequently covered that adjustment, including a nonlocal gate.

An independent six-site counterexample shows that exact plaquette preservation
per step does not guarantee exact future local dynamics. A six-site parity
correlation invisible to every plaquette is transformed by five CNOT gates
into a single-site observable.

The 2026 higher-dimensional DMT paper gives `6 L_perp + 2` for its particular
three-site-path protection on a square cylinder, not a universal plaquette
minimum. See https://arxiv.org/abs/2610.40776, Sec. III and Appendix.
