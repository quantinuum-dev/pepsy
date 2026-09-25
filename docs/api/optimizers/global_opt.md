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

`optimize_nlopt(...)` is the separate entry point for NLopt solvers. Install
optional backends using the [installation profiles](../../installation.md#optional-features).
