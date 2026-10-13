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

`mode="direct"` instead uses Pepsy's existing `contract_layered` worker with
`fit_mode="direct", fit_layer_mode="sequential"`. It absorbs and compresses
each tagged layer separately through Quimb's 1D compressor, retaining the
backend scalar for Quimb `TNOptimizer` autodiff. `direction="y"` selects
column boundaries; `"x"` selects rows. This mode does not precombine the
operator and input ket into one effective site tensor. Its efficiency relative
to native Quimb boundaries has not been benchmarked.

For an explicit lazy PEPO target, give the operator factors the `PEPO` tag
and pass `overlap_layer_tags=("BRA", "PEPO", "KET")`. `layer_tags` continues
to control the two-layer candidate norm. The overlap-specific setting retains
disjoint operator/ket layers instead of grouping the PEPO into the ket before
compression; its ordered tags must be distinct, include `BRA` and `KET`, and
reference existing target layers. Use `chi=(chi_norm, chi_overlap)` with a
larger overlap cap for the extra operator layer. Both inputs remain unchanged.

```python
U = U.copy()
U.add_tag("PEPO")
target = U.apply(old_state, contract=False)
fit = GlobalOptimizer(guess, target, chi=(chi_norm, chi_overlap),
                      target_norm=measured_target_norm)
out = fit.optimize(
    n=100, optimizer="L-BFGS-B", autodiff_backend="torch", normalize=False,
    loss_kwargs={"mode": "direct", "direction": "y", "cutoff": 0.,
                 "layer_tags": ("BRA", "KET"),
                 "overlap_layer_tags": ("BRA", "PEPO", "KET")},
)
```

For split-gate targets, native MPS contractions temporarily combine factors within
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
