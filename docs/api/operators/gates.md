# `pepsy.operators.gates`

## Operator convention

Dense MPO/PEPO builders consume gates in application order: a stream
`[A, B]` represents `B @ A`. Gates can be matrices or tensors with all output
axes followed by all input axes. Operators use Quimb's native convention:
upper `k...` indices are outputs and lower `b...` indices are inputs.
This agrees with Quimb's `to_dense`, operator application, and lazy composition.

Use `pepsy.tensors.tns_align(state, operator)` for lazy application in this
convention. Its explicit `transpose=True` option applies the operator's
transpose, for callers that knowingly hold a transposed representation.
Earlier dense builders transposed each gate individually. Rebuild saved
operators from their original streams: a final transpose alone cannot in
general repair both their orientation and ordering.

Raw lower-leg gates and lazy lower-leg operators have different meanings in
Quimb. A raw gate `G` on operator `X` produces `X @ G.T`; request
`transpose=True` to obtain `X @ G`. A lazy lower-leg sub-MPO `A` produces
`X @ A` directly. `gate_with_submpo(..., inplace_mpo=False)` defensively copies
the applied MPO; it still applies it. The separate `inplace` flag controls
whether the target network is mutated.

## Truncation policy

`gate` and `gate_simple` default to `cutoff="auto"` and
`cutoff_mode="auto"`. Pepsy resolves these before dispatching to Quimb, using
the input tensor network's dtype and the same policy as `MpsOptimizer`:

| Network dtype | Automatic cutoff |
| --- | --- |
| `float64` / `complex128` | `1e-12` |
| `float32` / `complex64` | `1e-6` |
| 16-bit floating point | `1e-3` |

Automatic cutoff mode is `rsum2` (relative discarded squared singular-value
weight). Explicit nonnegative numeric cutoffs, including zero to disable
cutoff truncation, and explicit cutoff modes override the defaults.
Resolution reads dtype metadata without moving arrays or densifying native
Symmray tensors. It occurs once per public call, before any gate mutation;
subsequent internal SWAPs use the resolved policy. Supply compatible network
and gate backends explicitly; this does not convert user gates.

For `gate`, the optional final compression `chi_cutoff` also accepts and
defaults to `auto`. Both APIs accept `path_compress_cutoff="auto"`;
the existing `None` default inherits the resolved local gate cutoff.
Final `chi` compression uses canonical left compression for 1D networks and
the lattice compression API for PEPS/PEPO networks, preserving their native
array backend. It is separate from each gate's local `max_bond` split cap.

`build_pepo_from_gates(..., cutoff_mode="rsum2")` applies the requested
truncation policy to both gate splits and fallback compression. The fallback
also uses the supplied `cutoff`, rather than a separate hard-coded threshold.

## Gauge scale extraction

With dense arrays and `smudge=0`, `gate_simple` handles exactly zero external
bond weights by projecting onto their nonzero support before and after the
update. Original external weights are restored, including on an exception;
nonzero weights, requested cutoffs and bond caps are unchanged. This is an
exact support projection, not an epsilon regularization. It does not define
derivatives when the support changes or alter native Symmray gauge handling.
When differentiating a subsequent contraction at zero weights, absorb each
full weight once; splitting it as two square roots has a singular derivative.

For scale-preserving simple update, use:

```python
from pepsy.operators import gate_simple

gate_simple(psi, gates, gauges=gauges, max_bond=D, cutoff="auto",
            cutoff_mode="auto", renorm=False, strip_exponent=True)
```

`strip_exponent=True` normalizes each updated bond gauge to unit RMS and
stores its removed positive scale in `psi.exponent`, including every routed
SWAP. It also normalizes existing internal gauges before the gate stream.
This is scalar bookkeeping, with no additional SVD or BP solve. The physical
state includes `10**psi.exponent`; the gauges and core tensors alone are its
mantissa. Keep this metadata when copying or saving the state.

This option defaults to `False` for compatibility and requires `renorm=False`.
`renorm=True` intentionally discards the gate's singular-value scale and
cannot be combined with scale-preserving exponent tracking. Nonzero cutoffs
must be relative (`rel`, `rsum1`, `rsum2`); absolute cutoffs depend on the
representation's scalar gauge and are rejected. `cutoff=0` is also supported.
The existing `inplace=False` contract remains: the tensor network is copied,
but the supplied gauge dictionary is updated, so copy that dictionary too if
you need to retain the original core-plus-gauge state.

To measure the actual squared norm, absorb gauges once into a copy:

```python
physical = psi.copy()
physical.gauge_simple_insert(gauges)
norm2 = physical.make_norm().contract(all, optimize="auto-hq")
# Avoid materializing very large/small values:
mantissa, exponent = physical.make_norm().contract(
    all, optimize="auto-hq", strip_exponent=True
)
```

Pepsy D2BP/loop-cluster contractions likewise include the network exponent;
use their `strip_exponent=True` option for a mantissa/exponent result. The
norm-network exponent is twice the ket's exponent. This scale handling does
not make finite-cluster BP exact or remove SU truncation error.

`renorm_gauge(network, gauges, where, smudge=1e-12)` divides a bond's weights
by a detached positive scale and adds the logarithm of **that same scale**
to `network.exponent`. It preserves the represented operator and the first
derivative of its reconstructed weights. RMS evaluation uses weights relative
to their largest magnitude, avoiding squares of very large or tiny inputs.

`smudge` is a nonnegative finite floor on a nonzero RMS scale. An all-zero
gauge uses scale one and remains zero, with finite exponent bookkeeping.
This tracks numerical scale; it does not normalize a physical overlap.

Native Symmray `BlockVector` gauges use the same rule, reducing over backend
blocks and weighting every singular value equally even when sector sizes
differ. Blocks remain native, and the scale is detached at the block backend.

## Prepared dense simple update

`pepsy.operators.PreparedSimpleUpdate` snapshots a dense complex Torch PEPO,
its real SU weights, and an ordered schedule of `(sites, leg)` entries.
Legs are `"upper"` or `"lower"`; gates have one leading axis for independent
network copies. Qubit one-site and nearest-neighbor two-site gates are supported.
Cutoff and cutoff mode default to the same dtype-aware `"auto"` policy as
`gate_simple`, resolved once when the plan is constructed.

```python
from pepsy.operators import PreparedSimpleUpdate

plan = PreparedSimpleUpdate(target, weights, schedule, max_bond=4,
    cutoff=1e-12, cutoff_mode="rsum2", renorm_bonds=True)
outputs = plan.apply(gates)  # list of (PEPO, external_weights), with autograd
```

The plan batches equal-shaped operations on disjoint sites and retains every
site's supplied gate order. Independent networks are a second batching axis.
Each fast update checks the requested truncation rank for every member. If
any rank differs from the stored shape, the whole call replays the original
adaptive `gate_simple` algorithm. It never chooses a batch-wide maximum rank.
`plan.info` reports operation/batch counts and the last call's fallback status.
Targets and weights are copied; returned weights remain external and the
network exponent retains the removed scale. Empty schedules are unsupported.

`PreparedSimpleUpdate.apply_many(plans, gate_batches)` also batches compatible
ready updates across independent plans, including different schedule lengths.
Each gate list follows `apply`'s contract; the result is one output list per
plan in input order. Every plan retains its own cutoffs, site order and fallback.
Only matching array shapes, numerical options and execution backends share a
kernel. Weight conditioning is performed on the combined independent vectors.
`plan.info["shared_batches"]` counts that plan's updates shared with other plans
in the last call. Use `apply` for a single plan.

Use the desired `TorchLinalgConfig.activated()` scope around evaluation.
The default `torch_backend=None` uses batched eager array kernels.
`torch_backend="aot_eager"` captures each kernel's forward and registered
stabilized backward together; boundary contraction is outside this plan.
This optional capture uses Torch's internal AOT interface and is version-sensitive.
Inductor is rejected: the tested complex backward failed reference checks.
No CUDA graph replay or complete tensor-network compilation is claimed.
Native Symmray and fermionic arrays retain their existing public paths.

## Native fermionic gates

The gate-to-operator builders accept native Symmray fermionic gates directly;
they do not convert them through dense arrays. Charge-neutral gates work by
default, such as `Fermion.hopping_gate(...)`:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
gate = fermion.hopping_gate(0.01, t=1.0)

mpo = pepsy.build_mpo_from_gates(gate, where=(0, 1), max_bond=16)
pepo = pepsy.build_pepo_from_gates(
    gate,
    where=((0, 0), (0, 1)),
    mapper=pepsy.OneDMap(2, 2, mode="snake-row-major"),
    max_bond=16,
)
```

The resulting tensors remain `U1U1FermionicArray` (or the corresponding
`U1`/`Z2` native type). `build_pepo_from_gates` uses `OneDMap` to choose the
MPO ordering before embedding it on the 2D lattice. A definite-charge native
operator can also be used explicitly:

```python
charged_mpo = pepsy.build_mpo_from_gates(
    charged_gate,
    where=(0, 1),
    allow_charged=True,
)
charged_pepo = pepsy.build_pepo_from_gates(
    charged_gate,
    where=((0, 0), (0, 1)),
    mapper=pepsy.OneDMap(2, 2, mode="snake-row-major"),
    allow_charged=True,
)
```

`allow_charged=True` means the returned operator carries the accumulated
charge of the sequential gate product. It is opt-in because charged gates
change the symmetry sector; ordinary charge-preserving evolution should leave
it disabled.


## Gate transforms

`pepsy.operators.gates.gate` and `gate_simple` accept `dagger=True` or
`transpose=True`. The option is capability-checked against the installed
Quimb gate API and applied to the requested user gate. If a non-local gate is
routed through internal SWAPs, those SWAPs are always applied normally; only
the final requested gate is transformed.
