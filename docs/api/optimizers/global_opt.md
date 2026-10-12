# `pepsy.optimizers.global_opt`

`GlobalOptimizer` evaluates PEPS norms and fits a state to a target using
Quimb's `TNOptimizer`. Use [energy optimizers](energy.md) when the objective
is a Hamiltonian expectation rather than a target-state overlap.

## Norms and target loss

```python
import quimb.tensor as qtn
from pepsy.optimizers import GlobalOptimizer

state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=1, dtype="complex128")
target = state.copy()
optimizer = GlobalOptimizer(state, target)
target_norm = optimizer.norm(state=target, mode="exact", opt="greedy")
loss = optimizer.loss(
    mode="exact", opt="greedy", cost_f="fid", val_=target_norm,
)
assert abs(loss) < 1e-10
```

`norm()` returns the squared norm, `⟨state|state⟩`. `normalize()` changes
the supplied state in place. A target is optional for these operations;
`loss()` and optimization require one. Use `set_target(...)` to replace it.
For fidelity loss, `target_norm` supplies the target's squared norm; `val_`
remains a legacy alias. Constructor `target_norm=1.0` assumes a normalized
target without contracting its norm. `set_target(..., target_norm=1.0)` resets
the old norm when replacing the target.

Constructor `chi=(chi_norm, chi_overlap)` governs normalization and loss.
Per-metric mappings can override it. Loss and norm accept `layer_tags` to
control bra/ket absorption order (default `("KET", "BRA")`). Use
`mode="mps", mode_="direct"` for Quimb direct boundary compression, or retain
`mode_="mps"`. These are differentiable Quimb contractions, separate from
Pepsy's iterative boundary FIT solver.
For split-gate targets, MPS contractions temporarily combine factors within
each site and layer exactly. Loss and norm evaluation preserve the caller's
tensor data, tags, and index names; this preparation can enlarge temporary
site tensors while retaining gradients.
`canonize` controls the boundary canonicalization pass. For explicit gate
factors, `canonize=False` can avoid rank-deficient QR derivative charts; it
does not eliminate SVD truncation or promise equal finite-chi objectives.

## Optimization

Use `make_tn_optimizer(...)` to configure a Quimb optimizer, or
`optimize(n=..., autodiff_backend=..., optimizer=...)` to run it. Configure
the objective through `loss_kwargs`; `return_losses=True` returns the
optimized state and loss history. The default autodiff backend is Torch,
which must be installed separately.

Standalone optimizers register stabilized Torch/JAX derivative rules before
constructing a trace. Pass `torch_linalg_config=TorchLinalgConfig(...)` for an
explicit Torch policy, or `register_linalg=False` to manage registration.
Torch defaults to no JIT; JAX defaults to JIT, zero loss cutoff (fixed SVD
ranks), and `strip_exponent=False`. Explicit choices override these defaults.
Registrations are process-global.

Both optimization entry points restore each returned tensor to its input
array backend, dtype, and device before optional output normalization. Native
Symmray arrays retain their structure while their blocks are converted.
Standalone output normalization defaults to `normalize=True`; pass
`normalize=False` to disable it. `normalize()` uses `normalize_kwargs`,
including separately configured normalization options.

`optimize_nlopt(...)` is the separate entry point for NLopt solvers. Install
optional backends using the [installation profiles](../../installation.md#optional-features).

The NLopt route restores the best finite evaluated parameter vector, including
after a caught NLopt exception. `losses` remains the original evaluation
history; `final_loss` is the score associated with the restored vector, before
optional normalization. For fidelity objectives this score is invariant under
normalization. `optimization_info` records `best_restored`, `stopped_early`,
`returned_loss`, and `termination_reason`, with error details when relevant.
If a failed run has no valid evaluated vector, the input state is retained
and `success=False` is recorded. Convergence is not implied by recovery.

Best-vector tracking uses the installed Quimb evaluation history and
vectorizer capabilities. It adds no objective or target-norm contractions.
