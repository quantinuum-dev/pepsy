# `pepsy.operators.MPOAutomaton`

`MPOAutomaton` is the explicit structural layer for open-boundary MPOs. A
channel is a virtual state on one bond cut; a transition is a local operator
edge between two channel states. Every path from `start_state` to
`done_state` contributes one product operator.

```python
import numpy as np
from pepsy.operators import MPOAutomaton

x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

automaton = MPOAutomaton(6)
automaton.add_local_term(2, x, coefficient=0.3)
automaton.add_factorized_term((0, 5), (z, z), coefficient=-1.2)

# Exact tensor assembly. No SVD, QR, canonicalization, or compression occurs.
mpo = automaton.to_mpo()
```

For a nontrivial operator string between two endpoints, pass the intermediate
local operators explicitly:

```python
automaton.add_factorized_term(
    (1, 4),
    (x, x),
    string_operators=(z, z),
)
```

Use `add_product_term` for a product with more than two supported sites. The
support is given in chain order, and `string_operators` contains all local
operators on the omitted sites:

```python
automaton.add_product_term(
    (0, 2, 5),
    (x, z, x),
    string_operators=(z, z),
)
```

Automata can also be combined exactly at the channel level. `a.compose(b)`
forms the operator product `a @ b`, while `a.power(n)` forms an exact
non-negative integer power. `add_automaton` adds a direct-sum path with state
remapping, which is useful for explicit polynomial or time-step expansions:

```python
identity = MPOAutomaton.identity(6, 2)
correction = automaton.power(2)
identity.add_automaton(correction, coefficient=0.5)
mpo = identity.to_mpo()
```

These operations only assemble structural paths. They do not compress the
resulting channel space. `trim()` is the one exact structural cleanup: it
removes channels that are unreachable from either boundary without changing
any accepted operator path. `to_mpo()` rank-reveals NumPy, Torch, CuPy, or JAX
layered arrays in two directional passes before Quimb MPO materialization by
default; pass `delinearize=False` to skip it. Numerical bond compression
remains an explicit follow-up.

The model-facing `ham_tn.to_mpo(...)` builder keeps its existing structural
cleanup. For supported dense inputs, `delinearize` defaults to `True` on
`ham_tn.to_mpo` and `MPOAutomaton.to_mpo`; pass `False` to skip the two
rank-revealing passes before the optional Quimb SVD. On the automaton builder
path these run on the layered arrays before Quimb MPO materialization. The
returned MPO exposes pass counts
and bond reductions in `.pepsy_delinearization`. With a builder-level
`to_backend`, the automaton builder runs the sweep on converted tensors before
the final numerical compression. NumPy, Torch, CuPy, and JAX dense arrays are
supported; the rank decision is discrete and reads scalar diagnostics on the
host. When JAX is tracing a function, the data-dependent rank reductions are
skipped so gradients flow through the unreduced MPO. Structured Symmray arrays
keep their backend-specific compression path. NumPy rank-
revealing QR uses optional SciPy; without it, exactly proportional channels
are still removed.

The existing channel/transition tuple representation can be wrapped with
`MPOAutomaton.from_legacy(...)` and emitted with `to_legacy()`. This allows the
current Symmray MPO assembly code to adopt the common structural model without
changing its backend-specific charge handling in the first migration step.

`MPOAutomaton.to_mpo(compress=True)` is intentionally rejected. If an
approximate bond truncation is wanted, call Quimb's `mpo.compress(...)` as a
separate, visible operation after inspecting or recording the raw bond
dimensions.

## Build from product terms

`from_product_terms` validates the complete term list, then shares exact
prefixes and suffixes before emitting transitions:

```python
automaton = MPOAutomaton.from_product_terms(
    3,
    [((0, 1), (z, z), 0.5), ((1, 2), (z, z), 0.5), ((1,), (x,), 0.2)],
)
mpo = automaton.to_mpo()
```

Sites must be strictly increasing integers in `[0, L)`. All operators have
the same square shape `(d, d)`; omitted sites between endpoints use identity
operators unless `string_operators` are supplied in a term mapping. Inputs
are not modified. Backend arrays remain native, and sharing uses conservative
object identity when host fingerprinting would disrupt tracing or gradients.
`share_channels=False` keeps a separate path for each term.

The default result is an `MPOAutomaton`. `return_slots=True` additionally
returns coefficient slots in input term order for parameterized builders.
In shared mode, slot operators omit the supplied scalar coefficients so they
can be bound later; identical terms can share a slot. Unshared mode retains
the coefficients. Ordinary operator construction should use the default.

## Tensor axes and ownership

`to_arrays()` returns a tuple in increasing chain-site order. With physical
dimension `d` and neighboring channel counts `D_left`, `D_right`:

| Site | Array shape / axes |
| --- | --- |
| Single-site chain | `(d, d)` = `(output, input)` |
| First site | `(D_right, d, d)` |
| Interior | `(D_left, D_right, d, d)` |
| Last site | `(D_left, d, d)` |

`to_mpo()` attaches Quimb indices using this same `lrud` convention.
Materialization does not modify the automaton or the supplied local operators;
Torch/JAX arrays retain their backend and differentiation graph. Structural
methods such as `add_product_term` do modify the automaton. Treat stored
operator payloads as shared inputs rather than assuming the builder copies
their array storage.

## Cluster exponentials

For automaton construction of a finite cluster exponential or ordered product,
use [`prepare_cluster_channels(..., preparation="automaton")`](cluster_channels.md#mpo-preparation-with-operator-aware-automaton-reduction).
It compiles reachable disjoint-cluster paths and reduces states with certified
Pauli identities. `max_bond=None` uses no SVD/QR and preserves the declared
cluster family. This separate opt-in path does not change `MPOAutomaton`'s raw
transition conversion or make a Hamiltonian automaton an exponential evaluator.
