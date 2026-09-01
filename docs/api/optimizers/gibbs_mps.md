# GibbsMps

`GibbsMps` prepares a finite-temperature Gibbs state with purification and
ordinary MPS gate replay:

```python
from pepsy.optimizers import GibbsMps

terms = [
    (("ZZ", 1.0), (0, 1)),
    (("X", 0.25), 0),
]

gibbs = GibbsMps(terms, shape=8)
gibbs.prepare(beta=0.4, dt=0.01, chi=64, progress=True)

purification = gibbs.mps
rho = gibbs.to_mpo()                    # normalized, Tr(rho) = 1
rho_raw = gibbs.to_mpo(normalized=False)
Z = gibbs.partition_function()
```

The internal MPS has `2 * L` sites in the order
`(physical_0, ancilla_0, physical_1, ancilla_1, ...)`. Physical site `i` is
therefore MPS site `2 * i`, and every Hamiltonian gate acts only on those even
sites. The initial state is a product of Bell pairs

\[
|I\rangle = \bigotimes_i |\Phi_i\rangle,
\qquad
|\Psi_\beta\rangle =
(e^{-\beta H/2} \otimes I)|I\rangle.
\]

Tracing the odd ancilla sites produces a positive operator proportional to
`exp(-beta * H)`. `to_mpo(normalized=True)` divides by its represented trace;
`partition_function()` returns the physical `Tr(exp(-beta * H))`, including the
`d**L` factor when normalized Bell pairs are used.

## Terms and lattice layouts

Terms use the same forms as `MPOBasis.from_terms`, including direct 1D terms:

```python
terms = [
    (("ZZ", J), (i, i + 1)) for i in range(L - 1)
]
terms += [(("X", h), i) for i in range(L)]
```

Regular lattice coordinates use `shape` and `map_mode`, or an explicit
`OneDMap`:

```python
gibbs = GibbsMps(
    terms,
    shape=(Lx, Ly),
    map_mode="snake",
)
```

The first implementation accepts one-site and two-site terms, including
long-range two-site couplings. General operators acting on more than two
sites and explicit string operators across a gap are intentionally rejected;
they need a multi-site gate application route that is not yet part of this
API.

## Imaginary-time stepping and compression

`prepare(beta, n_steps=N)` applies `N` symmetric second-order Trotter steps to
imaginary time `beta / 2`. If `dt` is supplied instead, `N` is chosen by
ceiling so the actual step does not exceed the requested value. With neither
argument, one Trotter step is used.

The default replay mode is `mode="mpo"`, which is appropriate for the
non-unitary gates and the interleaved physical/ancilla layout. Other ordinary
open-boundary `MpsOptimizer` compression modes can be selected. `GibbsMps`
always sets `non_unitary=True`; unitary stabilization and unitary overlap
diagnostics are not used. `to_backend` is forwarded through term compilation,
Bell-pair construction, generated exponentials, MPS replay, and ancilla
tracing.

The returned `GibbsMps` object stores the live optimizer in `gibbs.optimizer`,
the prepared gate stream in `gibbs.gates`, and the requested temperature in
`gibbs.beta`. Calling `prepare` again starts from fresh Bell pairs.
