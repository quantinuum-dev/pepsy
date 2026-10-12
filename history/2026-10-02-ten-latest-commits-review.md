# 2026-10-02 — Review of the ten latest Pepsy commits

- Scope: review the latest ten local commits in Pepsy and Gaugy; no fixes,
  staging, commits, pushes, or remote fetches.
- Branch / baseline: `develop` / `d7d509b`.
- Commits reviewed, newest first: `d7d509b`, `45d6b5e`, `ab9a445`,
  `25e9e18`, `608b664`, `5558857`, `3bfaaa9`, `5eadd9f`, `c796530`,
  `7773d2e`. The merge's second-parent MPO changes are included.
- Existing working-tree changes were preserved. Numerical validation imports
  a `git archive` of HEAD under `/tmp/review-pepsy-d7d509b`, excluding the
  unrelated uncommitted solver and test edits.
- This new review journal is uncommitted; package implementation is unchanged.

## Confirmed findings

All three regressions were introduced by `c796530`. Each succeeds using the
default public API on its parent `7773d2e` and fails on current HEAD.

1. **P1: default delinearization can remove an order-one operator term.**
   `MPOAutomaton.to_mpo()` now calls `_delinearize_mpo_arrays` by default.
   The suffix sweep accepts a rank reduction using only a local relative
   residual, without accounting for the opposing tensor's scale. Build two
   product terms `X ⊗ I` and `(1e-15 Z) ⊗ (1e15 Z)`. The unreduced MPO
   equals `X ⊗ I + Z ⊗ Z` exactly, while the default result loses `X ⊗ I`,
   has bond dimension one and maximum entry error 1. The equivalent Torch
   float64 case also fails; float32 with scales `1e-6` / `1e6` fails too.
   Local roundoff bounds do not certify the represented global operator.
   Relevant code: `src/pepsy/operators/_structural_compression.py:551` and
   `src/pepsy/operators/mpo_automaton.py:1723`.
   Suggested fix: balance channels or bound the error after applying the
   opposing tensor before accepting a reduction; otherwise keep that rank.

2. **P1: default Torch rank reduction drops a zero-initialized parameter's
   derivative.** For `H(θ)=X ⊗ X + θ Z ⊗ Z`, construct an automaton with a
   float64 Torch coefficient `θ=0` requiring gradients. Differentiate
   `sum(H * (Z ⊗ Z))/4`. The default returns derivative 0; the unreduced
   result and the parent commit return the correct derivative 1. At `θ=.1`
   both return 1. The pointwise reduced column space omits a direction needed
   by parameter derivatives. This can stall optimization from zero.
   Relevant code: `_backend_linear_factor`'s JAX-only tracing guard at
   `src/pepsy/operators/_structural_compression.py:91` and discrete rank
   selection at line 120.
   Suggested fix: preserve trainable Torch channels unless their dependencies
   hold symbolically for all parameter values, analogous to the JAX guard.

3. **P2: the automatic supported-backend check admits Torch float16 arrays
   that cannot run the required SVD.** A two-site `X ⊗ X` automaton with
   float16 CPU Torch operators builds successfully without delinearization
   and on the parent commit. The default now raises
   `RuntimeError: "linalg_svd_cpu" not implemented for 'Half'`.
   Relevant code: `src/pepsy/operators/mpo_automaton.py:1716` and
   `src/pepsy/operators/_structural_compression.py:116`.
   Suggested fix: gate automatic reduction by dtype/device capability, or
   use a validated promoted decomposition while preserving output policy.

The standalone reproduction is `/tmp/review-mpo-delinearization.py`; output
is `/tmp/review-mpo-delinearization.log`. Temporary files are not durable.
The following public construction reproduces finding 1:

```python
import numpy as np
from pepsy.operators import MPOAutomaton

x = np.array([[0., 1.], [1., 0.]])
z = np.diag([1., -1.])
a = MPOAutomaton.from_product_terms(2, [
    ((0, 1), (x, np.eye(2)), 1.),
    ((0, 1), (1e-15*z, 1e15*z), 1.),
])
expected = np.kron(x, np.eye(2)) + np.kron(z, z)
print(np.max(abs(a.to_mpo().to_dense() - expected)))  # 1.0
print(np.max(abs(a.to_mpo(delinearize=False).to_dense() - expected)))  # 0.0
```

For finding 2, construct the same two-site public automaton using Torch X/Z
operators and terms `((0,1),(X,X),1.)`, `((0,1),(Z,Z),theta)`.
Use `theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)` and
`torch.autograd.grad((mpo.to_dense()*torch.kron(Z,Z)).sum()/4,theta)`.
For finding 3, use one `((0,1),(X.half(),X.half()),1.)` term.

## Fresh validation

Activated the shared `~/envs/py312` environment, with one BLAS/OpenMP thread.
Main snapshot suite:

```text
python -m pytest -q -o addopts='' \
  tests/test_tree_factor_sampler.py tests/test_tree_symmray_factor.py \
  tests/test_tree_sampler_validity.py tests/test_tree_sampler.py \
  tests/test_structural_compression.py tests/test_optimize_peps.py \
  tests/test_prepare_boundary_inputs.py tests/test_gate.py
```

**897 passed, 2 skipped, 3 warnings in 268.92 seconds.** These existing
fixtures do not cover the three independent probes above.

Additional automaton/Hamiltonian/SciPy selection (`test_mpo_automaton.py`,
`test_ham.py`, `test_gradient_solver.py`, filtered with
`-k 'automaton or delinear or scipy'`) initially passed 33 tests and failed
two GPU tests during competing process initialization: JAX could not allocate
its default GPU memory pool, and CuPy could not create a cuSolver handle.
A retry with `XLA_PYTHON_CLIENT_PREALLOCATE=false` while the competing jobs
still ran also failed on GPU allocation/handle initialization. The JAX case
passed separately on CPU (1 passed in 11.96 seconds). After the other jobs
completed, **both GPU cases passed in 10.48 seconds** with preallocation
disabled. This gives **932 passing focused checks and 2 skips**, excluding
the repeated CPU case. The initial GPU errors are environment/resource
failures, not additional confirmed code findings.

No full-package suite or performance benchmark is claimed. Installed versions:
Quimb `1.15.1.dev79+gb5e316200`, Autoray `0.11.1.dev9+g1291702f9`,
Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev11+g1a3481803`,
Torch `2.6.0+cu124`, JAX `0.10.2`, igraph `1.0.0`.

## Follow-up

Address the three findings in a separately authorized implementation task.
This review does not activate older proposed work or publish changes.
