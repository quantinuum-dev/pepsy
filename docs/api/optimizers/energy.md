# `pepsy.optimizers.energy`


> API details are maintained as handwritten Markdown in this page.

## Torch SVD/QR policy

`PepsEnergyOptimizer` and `MpsEnergyOptimizer` accept the same optional
`torch_linalg_config` argument as `PepsOptimizer`:

```python
policy = pepsy.TorchLinalgConfig(
    mode="complex",
    stabilized=True,
    svd_driver="auto",
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
