# Tutorial: Fidelity Diagnostics

`contract_boundary` returns:

- `res.cost`: final contracted scalar
- `res.fidel`: per-step fidelity values collected during sweeps

Use `res.fidel` to understand where approximation error accumulates.

## Basic interpretation

If values are close to `1.0`, local fit quality is strong at those steps.
Lower values identify harder regions.

## Do not mix fidelity meanings

Pepsy uses two different diagnostics that are both commonly called
“fidelity”:

- `local_norm_fidelity` and `cumulative_norm_fidelity` are retained-norm
  compression proxies. They are available from canonical MPS/Tree updates and
  their stabilizer coefficient states without a reference-state overlap.
- `fit_overlap_fidelity` is a genuine target-state overlap produced by a
  successful DMRG/FIT update when its disposable exact target is available.

The first metric measures compression loss through the canonical tensor norm;
the second compares two states. A DMRG overlap is therefore an additional
quality check, not a replacement for the norm ledger used by MPO, SVD, Tree,
or stabilizer compression.

For Tree optimizers, `track_truncation=True` is a third, independent diagnostic:
it performs extra local spectrum work to report per-edge discarded weight. It
is not needed for the cheap path-level norm ledger.

## Split left/right products

For `direction="y"` and `max_separation=0`, a common split is `Ly // 2`:

```python
import numpy as np

split = ket.Ly // 2
f_left = np.prod(res.fidel[:split]) if split > 0 else 1.0
f_right = np.prod(res.fidel[split:]) if split < len(res.fidel) else 1.0

print("left product:", f_left)
print("right product:", f_right)
```

For `direction="x"`, use `split = ket.Lx // 2`.

## What to change if fidelity drops

1. Increase `chi`.
2. Increase `n_iter`.
3. Try `fit_mode="two-site"` with `fit_sweep_sequence="RL"` so the boundary
   can discover better bond subspaces up to `chi`.
4. Try `fit_mode="global"` as a slower reference solve.
5. Compare `direction="y"` vs `"x"` and choose the stabler one.

## Pitfall

A near-`1.0` fidelity at first step and lower values later is common; it does not
necessarily indicate a one-side bug. Later steps usually carry larger effective
environments and truncation pressure.
