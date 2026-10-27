# `pepsy.optimizers.energy`

## Batched MPS and tree readout

`MpsEnergyOptimizer.measure()` and `TreeEnergyOptimizer.measure()` evaluate
total energy and named local sums together:

```python
from pepsy.optimizers.energy import MpsEnergyOptimizer
from pepsy.tensors import build_optimizer

reader = MpsEnergyOptimizer(
    mps, terms=hamiltonian_terms, route="canonical",
    contraction_opt=build_optimizer(progbar=False, parallel=False),
)
result = reader.measure(
    {"magnetization": {(0,): Z, (1,): Z}, "correlation": {(0, 1): ZZ}},
    route="envs", torch_batch="vmap", chunk_size=16,
)
energy = result["energy"]                 # total, not per-site
magnetization = result["observables"]["magnetization"]
actual_route = result["metadata"]["route"]
```

Operators are arrays; supports use ordered site numbers. Names map to sums
of local terms, so divide a magnetization sum by the site count when a mean
is wanted. Energy follows the class's `real` setting; observables retain their
complex values. `normalized` follows the class setting. `measure` requires a
local-term Hamiltonian; the existing MPS `energy()` API continues to support MPOs.

MPS `route` is available on the constructor, `energy`, `loss`, and `measure`:

| Route | Readout |
| --- | --- |
| `canonical` | One disposable MPS, moving its canonical center across supports |
| `envs` | Shared exact block environments built in one sweep across requested intervals |
| `exact` | Full-network local contractions; the existing MPS energy default |

Dense combined readout builds one reduced density matrix per unique ordered
support. Different operators on that support reuse it. `info` on `measure`
can supply trusted canonical-center metadata; readout copies it and preserves
the live state. Canonical readout requires open boundaries; `envs` and `exact`
also support cyclic MPSs. Native Symmray arrays retain exact graded local
expectations, reported by `metadata["route"] == "exact"`, regardless of the
requested route. They are not converted to dense arrays for batching.
Older Quimb builds without `compute_partial_traces` use exact readout with a
compatibility warning and report the effective route in the metadata.

Tree `measure` uses one canonical tree copy and one RDM per support for dense
arrays. Native trees use their established graded readout, sharing contraction
paths and their state-versioned norm cache. Tree readout has no MPS `route`
argument; its effective route is reported as `canonical` or `native_exact`.

`torch_batch=False` is the default. `True` or `"einsum"` groups compatible
dense Torch RDM/operator pairs by shape, dtype and device and evaluates batched
traces. `"vmap"` vectorizes the same pure tensor contraction. The default chunk
size is 32. These options batch the **final operator contractions**, not the
environment sweep, QR operations, or mutable tensor-network objects. Non-Torch
and native arrays use the ordinary loop, reported in `metadata["torch_batch"]`.
Both state and operator gradients are preserved; batching makes no truncation
or normalization approximation. No CPU/GPU speedup is assumed.

Only operators and contraction paths may persist across state updates.
Numerical RDMs and MPS environments are rebuilt on every measurement call.
The automatic path optimizer for combined MPS readout is created lazily and
reused; an explicit optimizer object controls the search policy. Large global
exponents cancel before normalized readout. Unnormalized readout restores
their squared scale consistently across routes.

PEPS boundary energies adapt to Quimb's installed `method`/`mode` API and use
exact operator-inserted contractions on single-row or single-column states.
This path preserves the selected Boolean normalization and backend gradients.
An explicit measurement `route` can be supplied through `compute_kwargs`;
the default remains boundary contraction. Select a compatible compressor with
`boundary_mode`, for example `boundary_mode="direct"` with `route="envs"`.



## Torch SVD/QR policy

`PepsEnergyOptimizer` and `MpsEnergyOptimizer` accept the same optional
`torch_linalg_config` argument as `PepsOptimizer`:

```python
policy = pepsy.TorchLinalgConfig(
    mode="complex",
    stabilized=True,
    svd_driver="gesvd",
)
optimizer = pepsy.PepsEnergyOptimizer(
    state,
    hamiltonian,
    torch_linalg_config=policy,
)
```

The supplied policy controls both SVD and QR registration. PEPS energy
optimization automatically extends a policy to Quimb's raw Symmray split
drivers when needed, while preserving the user's driver and rank settings.

## Symmray fermionic encodings

For a native fermionic Symmray MPS, prefer native local Hamiltonian terms:

```python
import pepsy

estimate = pepsy.MpsEnergyOptimizer(
    fermionic_state,
    terms=hamiltonian,
    energy_per_site=False,
).energy()
```

``terms=`` accepts either a local-term mapping or the complete
``SymHamiltonian`` returned by ``fermion.hamiltonian(terms)``. For an explicit
one-site-plus-two-site Hamiltonian, passing the ``SymHamiltonian`` preserves
the fermionic ordering metadata and lets a native fermionic MPS evaluate the
terms directly without constructing the bosonic MPO.

The MPO returned by a Fermi-Hubbard ``SymHamiltonian.to_mpo(...)`` uses the
bosonic/Jordan-Wigner representation required by the DMRG MPO path. Pepsy now
rejects an implicit contraction of that MPO with a native fermionic MPS, since
the automatic re-encoding can create very large block-sparse intermediates.
For a deliberately small or explicitly managed conversion, pass
``allow_encoding_conversion=True`` to ``MpsEnergyOptimizer``.

An MPO returned by ``Fermion.build_mpo(...)`` is native graded and can be measured
directly with a native fermionic MPS. Pepsy applies that MPO sitewise as a
factorized graded MPO-MPS network, preserving Symmray's contraction order
without materializing an exponentially sized operator.

``Fermion.to_mpo(...)`` remains a compatibility alias of
``Fermion.build_mpo(...)``.

Repeated native-MPO evaluations reuse a per-optimizer cotengra path cache. The
default is uncompressed and exact. For a controlled approximation, pass for
example ``native_mpo_compression={"max_bond": 64, "cutoff": 1e-12,
"method": "svd"}``; always compare against an uncompressed result when setting
the truncation cap.

## PEPS optimizer step guard

For a truncated or boundary-MPS PEPS energy objective, an L-BFGS line search
can probe a high-energy trial point even when it retains a lower-energy best
state. A cheap parameter-space guard can limit this without performing a
second energy contraction:

```python
state, losses = optimizer.optimize(
    n=100,
    optimizer="L-BFGS-B",
    optlib="nlopt",
    max_parameter_step=5e-2,
    return_losses=True,
)
```

This places finite lower and upper bounds ``initial_parameter +/-
max_parameter_step`` on the flattened real optimizer vector for that run.
Use ``bounds=...`` when explicit per-parameter limits are needed. The guard
is opt-in; omitting both arguments preserves the unconstrained optimizer
behavior.

For occasional protection against a truncated local loss descending for
numerical reasons, enable the compact validation mode:

```python
state, losses = optimizer.optimize(
    n=100,
    optimizer="L-BFGS-B",
    optlib="nlopt",
    validate=True,
    validation_interval=None,  # automatic from n; e.g. 20 for checks every 20 evals
    return_losses=True,
)
```

Validation is opt-in and runs a few optimizer chunks, checking each candidate
with twice the training boundary bond dimension. If that validation energy
worsens or becomes non-finite, Pepsy rolls back to the last validated state.
Candidates reconstructed by Quimb are converted back to the selected
autodiff backend before validation, which is required for native Symmray
states whose blocks must not mix NumPy and Torch (or another backend).
With ``progbar=True``, this mode uses one outer progress bar showing the
training/validation chi values, validation step, local optimizer energy, the
higher-chi checked energy per site, and chunk status;
the nested optimizer bars are suppressed to keep the output readable.
Set ``validation_interval`` to a positive number to choose the number of
optimizer evaluations between checks. With ``None`` (the default), Pepsy
chooses an interval from ``n`` with at most five scheduled checks and a
minimum interval of ten evaluations.
Here ``local_E`` is the best local objective recorded by the optimizer at the
training chi, while ``check_E`` is the higher-chi validation energy.
The normal fast path is unchanged when ``validate=False``.

## Tree tensor networks

``TreeEnergyOptimizer`` mirrors the ``MpsEnergyOptimizer`` energy surface for
a :class:`~pepsy.optimizers.tree.TreeTensorNetwork`. It reports
``sum_i <psi|H_i|psi> / <psi|psi>`` term by term using the tree's own exact,
fermion-safe contraction, returns the same :class:`EnergyEstimate`, and can
optimize the tree tensors through Quimb's autodiff ``TNOptimizer``:

```python
import pepsy

estimate = pepsy.TreeEnergyOptimizer(
    tree_state,
    terms=hamiltonian,        # {where: operator} mapping or a SymHamiltonian
    energy_per_site=True,
).energy()

optimizer = pepsy.TreeEnergyOptimizer(tree_state, terms=hamiltonian)
tree_state, losses = optimizer.optimize(
    n=100,
    autodiff_backend="torch",
    optimizer="adam",
    progbar=False,
    return_losses=True,
)
```

The terms are dispatched through
:meth:`~pepsy.optimizers.tree.TreeTensorNetwork.local_expectations`, which
shares one contraction optimiser across every term (pass a reusable
``pepsy.build_optimizer(...)`` as ``contraction_opt`` to cache paths across
same-topology contractions) and reuses the memoized graded norm for ordinary
readout. During autodiff optimization, Quimb injects tensor arrays below the
TTN mutation hooks, so the loss instead sums unnormalized numerators and
divides by a freshly contracted full-tree norm on every call. Afterward the
canonical metadata is invalidated rather than rebuilt around an arbitrary
post-optimization centre; native fermionic normalized readouts therefore stay
gauge invariant. The returned state remains a ``TreeTensorNetwork`` and the
scalar history is available as ``optimizer.losses``.
