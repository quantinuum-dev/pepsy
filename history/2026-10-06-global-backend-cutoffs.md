# 2026-10-06 — Backend-specific global contraction cutoffs

User requested registered Torch/JAX SVD rules for global PEPS fitting, Torch
cutoff 1e-10, JAX JIT cutoff zero, and confirmation of MPS boundary contraction.
Baseline develop 4f8935e plus the uncommitted fidelity diagnostic policy.
These changes remain uncommitted; no restart or push performed.

PepsOptimizer already registers TorchLinalgConfig(stabilized=True) before
Torch optimization and register_jax_linalg(stabilized=True) before JAX
optimization/tracing. Explicit registration opt-outs/configuration are retained.
JAX already defaults to jit_fn=True, loss cutoff=0 and strip_exponent=False;
its QR derivative remains native JAX. No new registration code is needed.

Torch global norm/overlap contraction defaults now use cutoff=1e-10 rather
than inheriting the gate cutoff. Explicit global metric mappings win. Both
the norm and overlap in the global objective still use Quimb's MPS
contract_boundary with the requested separate caps. Gate/warm-start cutoff,
outer normalization, and acceptance policies are unchanged.

Validation: test_optimize_peps.py, test_global_optimizer_safeguards.py, and
test_peps_global_jax.py: 155 passed, 13 warnings (66.45 s). Includes actual
Torch optimization and JAX JIT gradient/fidelity tests on CPU, registration
before tracing, output dtype preservation, and explicit-option precedence.
Compileall and git diff --check passed; Ruff is unavailable. No full suite.
This does not establish that the new cutoff fixes the saved 5x6 gradient
failure. The active CUDA0 process has its old code loaded.

During validation the user asked whether the current global fit is failing.
Confirmed the worker remains alive and t=.9 is completed. Through completed
steps: 105 global attempts, 102 NLopt early-stop errors, 48 accepted recovered
candidates. Individual abnormal fit termination is distinct from overall
evolution failure. The live process/settings were not changed.
