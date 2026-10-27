# GibbsMps

`GibbsMps` prepares a finite-temperature Gibbs state with purification and
ordinary MPS gate replay:

```python
from pepsy import bell_to_mps
from pepsy.optimizers import GibbsMps

terms = [
    (("ZZ", 1.0), (0, 1)),
    (("X", 0.25), 0),
]

gibbs = GibbsMps(terms, shape=8)
gibbs.prepare(beta=0.4, beta_step=0.05, chi=64, progress=True)

purification = gibbs.mps
energy_per_site = gibbs.energy() / gibbs.length
rho = gibbs.to_mpo()                    # normalized, Tr(rho) = 1
rho_raw = gibbs.to_mpo(normalized=False)
Z = gibbs.partition_function()
log_Z = gibbs.log_partition_function()  # natural log, safe for large scales
```

The reusable identity purification is also available directly as
`pepsy.bell_to_mps(L, phys_dim=2, normalized=True, to_backend=...)`. It is a
Quimb `MatrixProductState`, so it can be inspected, converted with
`apply_to_arrays`, or passed to other MPS workflows before any thermal gates
are applied.

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
`d**L` factor when normalized Bell pairs are used. For large or low-temperature
systems, use `log_partition_function()` to keep the partition-function scale in
log-space rather than exponentiating it. Internally, ancilla tracing uses
Quimb's native partial-trace reducer, which preserves Pepsy's MPS exponent in
MPO metadata. If `contract_opts` are supplied, GibbsMps uses equivalent native
tag-wise `strip_exponent=True` contractions because those options are not part
of Quimb's public partial-trace signature. Neither route densifies the
purification.

## Terms and lattice layouts

Terms use the same forms as `MPOBasis.from_terms`, including direct 1D terms:

```python
terms = [
    (("ZZ", J), (i, i + 1)) for i in range(L - 1)
]
terms += [(("X", h), i) for i in range(L)]
```

Regular lattice terms can keep their natural coordinates. If `shape` is
omitted, `GibbsMps` infers the location dimension and the smallest enclosing
2D/3D shape from the term locations, then creates a `OneDMap` using
`map_mode` (defaulting to `"snake"`):

```python
import quimb.tensor as qtn

edges = qtn.edges_2d_square(Lx, Ly, cyclic=True)
sites = sorted({site for edge in edges for site in edge})
terms = [(("zz", J), (u, v)) for (u, v) in edges]
terms += [(("x", h), site) for site in sites]

gibbs = GibbsMps(terms, map_mode="snake")  # infers shape=(Lx, Ly)
# Other supported traversals can be selected, for example:
# gibbs = GibbsMps(terms, map_mode="row-major")
```

The same inference distinguishes flat integer locations as a 1D chain and
coordinate tuples such as `(x, y)` or `(x, y, z)` as lattice sites. All terms
in one Hamiltonian must use the same location dimension. Supply `shape`
explicitly when the terms do not mention every site or when the intended
lattice extends beyond the largest coordinate.

Alternatively, regular lattice terms can use explicit `shape` with
`map_mode`, or an explicit `OneDMap`:

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

`prepare(beta)` starts from the identity purification at β=0 and applies
second-order Trotter steps to the target β. The default `beta_step=0.05`
bounds the **thermal β increment**. For positive β,

$$
N=\left\lceil\frac{\color{blue}\beta}{\Delta\beta_{\max}}\right\rceil,
\qquad {\color{blue}\Delta\beta}=\frac{\beta}{N},
\qquad {\color{purple}\Delta\tau}=\frac{\Delta\beta}{2}.
$$

Each physical-side gate evolves imaginary time Δτ because the purification
uses `exp(-beta * H / 2)`. The selected values are `gibbs.n_steps`,
`gibbs.beta_step` (Δβ), and `gibbs.trotter_step` (Δτ). For example,
`beta=0.5` selects ten steps with Δβ=0.05 and Δτ=0.025. At β=0 no gates run.
The default `trotter_order=2` uses Quimb's
`LocalHamGen.get_trotter_gates`, which automatically groups non-overlapping
Hamiltonian edges into commuting layers and emits the symmetric palindromic
schedule. `trotter_order=1` and `trotter_order=4` are also available. Supply
`beta_step` to set another maximum increment, or `n_steps` to set the count
directly. For compatibility, an explicit `dt` retains its original meaning
as a maximum **purification imaginary-time step Δτ**, selecting
`N=ceil(beta / (2 * dt))`; it overrides `beta_step`. Explicit `n_steps` takes
precedence over both. `beta` must be finite and nonnegative; numeric step
limits must be finite and positive. These choices set the product-formula resolution;
convergence in Trotter step size and bond dimension remains problem dependent.

This preparation API requires a Quimb build that provides
`LocalHamGen.get_trotter_gates` (unavailable in Quimb 1.15.0). Pepsy reports
that missing capability explicitly; constructing and reading the beta-zero
purification remain available.

For example, the graph ordering and fusion controls can be made explicit:

```python
gibbs.prepare(
    beta=0.4,
    n_steps=8,
    trotter_order=2,
    trotter_ordering="sort",       # or None, "random", or edge layers
    trotter_fuse_adjacent=True,
    trotter_alternate=True,
)
```

The Hamiltonian terms are first combined by logical edge. One-site terms on
connected sites are lifted into the incident edges with an equal partition, so
the represented Hamiltonian is unchanged. A one-site term on a site with no
incident edge is kept as an exact one-site exponential; no artificial graph
edge is introduced. The generated Quimb metadata is available as
`gibbs.trotter_gates` (`frac`, `layer`, `step`, and logical `where`) and
`gibbs.trotter_layers`. The executable `gibbs.gates` stream contains the same
gates with logical sites mapped to the even physical positions of the
purification. For Hamiltonians with disconnected one-site terms, the
executable stream additionally contains those exact one-site gates, which do
not have a Quimb edge-layer record.

The graph-layer ordering is resolved once per preparation and passed explicitly
to Quimb's scheduler. Stable automatic orderings are cached by the Gibbs
object, while `trotter_ordering="random"` creates a fresh ordering for each
preparation. Thus `trotter_layers` describes the exact schedule that is
replayed, without a second graph-coloring pass.

The default replay mode is `mode="direct"`, which is appropriate for the
non-unitary gates and the interleaved physical/ancilla layout. Other ordinary
open-boundary `MpsOptimizer` compression modes can be selected. `GibbsMps`
always sets `non_unitary=True`; unitary stabilization and unitary overlap
diagnostics are not used. `to_backend` is forwarded through term compilation,
Bell-pair construction, generated exponentials, MPS replay, and ancilla
tracing.

The `mode="mpo"` spelling remains accepted as a compatibility alias for the
same Quimb direct compressor. `mode="dmrg"`, `mode="svd"`, and other ordinary
open-boundary MPS modes can be selected when a different compression path is
needed.

Common compression controls can be passed directly:

```python
gibbs.prepare(
    beta=0.4,
    n_steps=8,
    mode="dmrg",             # opt-in FIT; default is "direct"
    chi=128,
    contraction_opt="auto",
    n_iter=4,
    normalize_every=True,
    normalize_final=True,
)
```

The remaining `MpsOptimizer` constructor options and run options are available
through `optimizer_kwargs={...}` and `run_kwargs={...}`. Normalization is
non-unitary scale bookkeeping: the optimizer stores removed scale in its
exponent, so `to_mpo(normalized=False)` still represents the same raw operator.

The returned `GibbsMps` object stores the live optimizer in `gibbs.optimizer`,
the prepared gate stream in `gibbs.gates`, and the requested inverse temperature in
`gibbs.beta`. Calling `prepare` again starts from fresh Bell pairs.

## One trajectory with intermediate measurements

`evolve` yields the live state at β=0, every five completed Trotter steps by
default (`measure_every=5`), and at the final β. It reuses one optimizer and
the evolving purification:

```python
measurements = []
for state in gibbs.evolve(beta=1.0, beta_step=0.05, chi=128):
    measurements.append((state.beta, state.energy() / state.length))
```

This reaches the requested measurement points in one trajectory. The
initial β=0 energy is available too. `n_steps=N` selects an explicit count;
other options follow `prepare`, except `dt` is excluded to keep the β-step
meaning explicit. Each new `evolve` call starts from fresh Bell pairs.
Use one `evolve` call for a temperature curve; a loop of `prepare(beta)` calls
restarts from Bell pairs for every point.

Measurements are explicit calls in the loop. The iterator does not compute
energy, form a density MPO, or retain observable results automatically.
Use `measure_every=1` if measurements are needed after every integration step,
or specify the desired inverse temperatures exactly:

```python
for state in gibbs.evolve(
    beta=1.0, beta_step=0.05, measure_at=[0.13, 0.41, 0.73], chi=128,
):
    measurements.append((float(state.beta), state.energy(), state.local_infidelity))
```

Both schedules include β=0 and the final β. `measure_at` must be strictly
increasing and within `[0, beta]`; it cannot be combined with `n_steps` or a
non-default `measure_every`. Each interval is divided into equal integration
steps no larger than `beta_step`. For QMC temperatures, convert to β and sort
in increasing β before supplying the measurement targets.

Measure inside the loop; yielded objects share the same live state. Use
`state.mps.copy()` when a persistent purification snapshot is needed.
Use a copy for Quimb measurements that move the canonical center too. If the
live MPS is intentionally changed, call `state.optimizer.sync_canonicalization()`
before continuing evolution.

Equal batches on the uniform grid reuse their gate payloads and installed
queue. Multiple steps between yields can fuse adjacent layers. Fusion stops
at each yielded boundary so measurements follow complete steps. This changes the compression
sequence relative to a fully fused `prepare` call, so finite-χ results need
not be identical. During iteration `n_steps` counts completed steps; `gates`,
`trotter_gates`, and optimizer timing describe the latest measurement interval.

## Local thermal energy

After `prepare`, `gibbs.energy()` sums the prepared Hamiltonian's one- and
two-site expectations directly on the purified MPS:

$$
{\color{blue}E(\beta)}=
\frac{\langle\Psi_\beta|({\color{purple}H}\otimes I)|\Psi_\beta\rangle}
{\langle\Psi_\beta|\Psi_\beta\rangle}.
$$

It uses the coefficients selected by that preparation and a reusable Pepsy
`build_optimizer()` for the local contractions. It returns a backend scalar
without forming a reduced density MPO. Divide by `gibbs.length` for energy per
physical site; `to_mpo()` remains available when the reduced operator is needed.
The bound Hamiltonian operators are reused across energy measurements and
rebuilt after a new `prepare` call. The canonical measurement copy receives
the optimizer's known center, avoiding a full-chain center scan while leaving
the live MPS and its canonical metadata unchanged. Arbitrary observables
called directly through Quimb follow the options supplied by the caller;
`energy()` owns this optimized Hamiltonian readout.

`energy(route="envs")` selects shared-environment readout; `"canonical"`
remains the default, and `"exact"` is available as a reference. The implementation
uses the existing `MpsEnergyOptimizer` measurement layer. To share reduced
density matrices and environments between energy and other observables, measure
them together:

```python
observables = {
    "magnetization": {(i,): Z for i in range(gibbs.length)},
    "zz_01": {(0, 1): ZZ},
}
for state in gibbs.evolve(beta=1.0):  # measure_every=5 by default
    result = state.measure(observables, route="envs", torch_batch="vmap", chunk_size=16)
    measurements.append((float(state.beta), result["energy"], result["observables"]))
```

Here `Z` and `ZZ` are operator arrays, and supports use **flat physical site
numbers**, before interleaving: physical site `i` is mapped to MPS site `2*i`.
Named observables are sums of their local terms; energy is the total physical
energy. The metadata records the actual route, batching method, and number of
distinct supports. Torch batching is optional and applies to compatible final
RDM/operator contractions; other backends use the normal loop. RDMs and
environments are refreshed at every beta. See the
[shared readout guide](energy.md#batched-mps-and-tree-readout) for scope and
fallback behavior. Neither the environment route nor batching changes the
default until performance evidence supports doing so.

## Compression infidelity

`gibbs.local_infidelity` automatically reports the latest compressed gate or
FIT window in `direct` and `dmrg*` modes. Reading it adds no contractions. It
returns `None` before compression, in unsupported modes, or if the diagnostic
fails; `gibbs.optimizer.get_nonunitary_diagnostics()` supplies its support,
method, scope, raw fidelity and any error.

For exact post-gate target $t$ and compressed state $a$, the diagnostic is

$$
{\color{blue}\epsilon_{\mathrm{local}}}
=1-\frac{|\langle t|a\rangle|^2}{\langle t|t\rangle\langle a|a\rangle}.
$$

Dense paths cancel the canonical exterior. DMRG reuses the terminal FIT norm
$A$ and contracts only the target norm $T$, giving $1-(A/T)^2$ by the terminal
projection identity. Direct compression contracts the target norm and overlap
inside the active interval and reads the retained canonical-center norm.
Neither path treats the physical norm change under cooling as compression
loss. Both add work to evolution; the DMRG numerator is already available,
but its non-unitary target norm is not free. Native symmetry arrays use a
conservative full-network overlap fallback to preserve graded signs.

The value is for the **latest compression**, not a maximum over the beta step,
an accumulated error bound, Trotter error, or Gibbs-state fidelity. The unitary
norm-survival ledger remains separate. Explicit FIT overlap readout reuses
this measurement for non-unitary replay.
