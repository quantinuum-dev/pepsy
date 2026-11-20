# 2026-11-20 — Strict SRC warm starts for CUDA MPO fitting

On `develop` at `a5fb031`, dense MPO FIT now requires the requested SRC warm
start in `MpoOptimizer` and `MpoErrorMit`. Both one-site `dmrg` and `dmrg2`
used seeded CUDA SRC in the numerical checks. Older builds report a missing
backend-native SRC capability. The benchmark used a private Quimb 1.16.0
and Autoray 0.11.0 overlay.

The study is in the sibling examples repository. The depth-10 run evaluated
MPO χ=8, 16, and 32 against 19,200 MPS χ=128 readouts, with shot subsets from
600 to 19,200. At χ=32 the identity coefficient still reached 1.15422. At the
largest shot count the trace ratio moved energy per site from −1.74682 to
−1.79665 toward the noiseless −1.82110. [Study results](../../pepsy_examples/experiments/tensor_network_error_mitigation/RESULTS.md)

The focused modern environment selection passed 83 tests, including CUDA.
The legacy capability selection passed 28 tests. The examples CUDA smoke
check resumed two saved shot chunks and scored them with a reloaded SRC MPO.
Ruff and `git diff --check` passed. The changes are in the working trees.
