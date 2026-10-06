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
For fidelity loss, `val_` supplies the target's squared norm.

## Optimization

Use `make_tn_optimizer(...)` to configure a Quimb optimizer, or
`optimize(n=..., autodiff_backend=..., optimizer=...)` to run it. Configure
the objective through `loss_kwargs`; `return_losses=True` returns the
optimized state and loss history. The default autodiff backend is Torch,
which must be installed separately.

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
