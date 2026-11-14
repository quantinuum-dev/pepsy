# Pauli-basis MPO

`PauliMPO` stores sparse qubit operators or native coefficient cores in the
local `I`, `X`, `Y`, `Z` basis. Algebra, compression and DAOE filtering can
remain in that basis; general tensor-network work is compiled through
Pepsy's existing `FirstDegreeMPO` and Quimb MPO boundary.

```python
from pepsy.operators import PauliMPO

op = PauliMPO.from_terms(
    3,
    [
        (0.35, "ZIZ"),  # one full-chain term
        (-1.0, "ZZ"),   # translated nearest-neighbor sum
        (1.5, "X"),     # translated onsite sum
    ],
)
```

Short strings are translated over the chain. Use `boundary="periodic"` for
wrapped translations, or explicit support when a single placement is wanted:

```python
op = PauliMPO.from_terms(
    8,
    [(0.2, (0, 7), "ZZ")],
    boundary="periodic",
)
```

The exact Pauli-native operations are:

```python
op.trace()
op.dagger()
op.conjugate()
op.transpose()
op @ other
op.commutator(other)
op.partial_trace((0, 1))
op.norm()
```

Native operator products stay in the Pauli coefficient tensor train.  To keep
the whole product path native, first make the operands core-backed, then
multiply and compress the virtual bonds:

```python
left = PauliMPO.from_terms(8, [(1.0, "XX"), (0.3, "Z")]).compress_pauli(
    max_bond=32,
)
right = PauliMPO.from_terms(8, [(2.0, "ZZ"), (-0.2j, "YXY")]).compress_pauli(
    max_bond=32,
)

product = left @ right
product, report = product.compress(
    max_bond=32,
    cutoff=1.0e-10,
    return_report=True,
)
```

For native cores with shapes ``A[l, a, p]`` and ``B[b, r, q]``, the local
product uses the explicit Pauli structure tensor ``M[p, q, s]``:

```text
C[(l, b), (a, r), s] = sum(p, q) A[l, a, p] B[b, r, q] M[p, q, s]
```

The physical index remains the four-label ``I/X/Y/Z`` basis.  The product
temporarily multiplies the virtual bond dimensions, and native SVD
compression reduces those bonds without converting to a qubit MPO or
reindexing ket/bra legs.  ``to_mpo()`` is still the explicit conversion
boundary: it applies the local map ``P[p, upper, lower]`` and produces a
standard Quimb MPO with two dimension-two physical legs.

A four-row, vertically periodic flat network for ``trace(A @ B @ C @ D)`` can
be built directly from native `PauliMPO` cores. Apply the basis map locally at
each site so that no dense global operator or intermediate Quimb MPO is
formed. The Pauli-MPO tests exercise this construction against dense
references.

Canonicalization combines equal Pauli words, orders them deterministically,
and can prune small coefficients without leaving the Pauli basis:

```python
op = op.canonicalize(atol=1.0e-12, rtol=1.0e-10)
```

For MPO-style bond canonicalization and compression while retaining Pauli
physical legs, use the native coefficient tensor-train boundary:

```python
canonical = op.canonicalize_native(center=op.nsites - 1)
compressed, report = op.compress_pauli(
    max_bond=64,
    cutoff=1.0e-10,
    form="right",  # also "left" or an integer center
    return_report=True,
)

assert all(core.shape[2] == 4 for core in compressed.to_pauli_cores())
print(report.final_bond_dimensions, report.discarded_singular_weight)
for bond in report.bond_reports:
    print(bond.bond, bond.final_bond, bond.discarded_weight)
```

The native cores have shape `(left_bond, right_bond, 4)` with physical order
`I, X, Y, Z`. QR canonicalization and SVD rounding act on the coefficient
tensor train, never on a dense Hilbert-space matrix. The result remains a
`PauliMPO`; `to_mpo()` is only the execution boundary. Native compression is
the default: `op.compress(...)` and `op.compress(basis="native", ...)` are
equivalent to `compress_pauli`. Use `op.compress(basis="mpo", ...)` or
`compress_numerical(...)` when a numerically compressed Quimb MPO is wanted.
Native compression follows Quimb's `max_bond`, `cutoff`,
`cutoff_mode` (`rel`, `abs`, `sum1`, `rsum1`, `sum2`, `rsum2`), `form`, and
`renorm` conventions for the native SVD backend. The native canonical forms
are `"right"`, `"left"`, and an integer center. `form="flat"` performs
disjoint left and right sweeps without promising a canonical center. The
native backend supports full (`"svd"`), randomized (`"rsvd"`), and
iterative-style (`"svds"`/`"isvd"`) methods. Absolute and sum cutoffs are
converted for the Hilbert-Schmidt normalization of the physical Pauli
operators, so rank selection agrees with the compiled Quimb MPO.
Partial-SVD reports include the uncomputed spectrum through the local
reconstruction residual; `singular_values` contains only computed values.
Rank selection still uses that computed spectrum. For strict cutoff studies,
use full SVD and `renorm=False`. Reports describe compression diagnostics,
not an observable-error guarantee, especially for `form="flat"` or renormalized
results.

## Protected density matrix truncation (DMT)

`compress_dmt` preserves all partial traces on up to three contiguous sites
through compression, to numerical precision. For a density matrix these are
the local reduced density matrices; consequently compression preserves energy
when every Hamiltonian term fits inside such a region.

```python
rho = PauliMPO.from_terms(8, [
    (2.0**-8, "IIIIIIII"),
    (0.05 * 2.0**-8, "Z"),
])
rho, report = rho.compress_dmt(
    max_bond=32, variant="density", return_report=True,
)
print(report.final_bond_dimensions, report.max_constraint_error)
```

### Physical lattice regions

Pass `regions` to replace the default chain-local regions with complete
reduced-operator constraints on explicit MPO site sets. The sets may be
noncontiguous in MPO order. For a square lattice, use the same `OneDMap` to
construct the MPO and the region inventory:

```python
from pepsy.operators import PauliMPO, square_lattice_dmt_regions
from pepsy.tensors import OneDMap

mapper = OneDMap(3, 3, mode="snake")
plaquettes = square_lattice_dmt_regions(3, 3, mapper=mapper, kind="plaquettes")
l_shapes = square_lattice_dmt_regions(3, 3, mapper=mapper, kind="l_shapes")
rho = PauliMPO.from_terms(9, [(2.0**-9, "I" * 9), (0.1 * 2.0**-9, (0,), "Z")])
compressed = rho.compress_dmt(max_bond=32, regions=plaquettes + l_shapes)
```

`kind="triples"` includes both straight and L-shaped connected three-site
regions. The helper generates open square-lattice neighborhoods; pass explicit
MPO-site index sequences for other shapes, arbitrary ordering, or periodic
neighbors. The mapping must match the order actually used to construct `rho`.
The same `regions` argument is accepted by `apply_gate_dmt` and `PauliDMT`.

`map_mode="snake"` follows `OneDMap`'s default column-oriented snake; it is
not the same as `"snake-row-major"` or `"row-major"`. For a 3-by-2 lattice,
the entries below are MPO indices at coordinates `(x, y)`, with `x` as the
row and `y` as the column:

```text
snake       snake-row-major    row-major
0 1         0 5                0 1
3 2         1 4                2 3
4 5         2 3                4 5
```

An explicit `mapper` takes precedence over `map_mode`, as in
`MPOBasis.from_square_lattice`. `PauliMPO` does not store lattice coordinates,
so it cannot detect a mismatched mapping. Check that the mapper describes the
ordering of the density MPO, not just the order of an unrelated state.

When the density MPO is constructed in the **storage order** of an MPS whose
logical sites were permuted by `MpsOptimizer`, supply its position-to-logical
mapping as `site_order`:

```python
logical_mapper = OneDMap(3, 3, mode="row-major")
regions = square_lattice_dmt_regions(
    3, 3, mapper=logical_mapper, site_order=mps_optimizer.logical_order,
)
```

Here the logical labels must use `logical_mapper`. If the density MPO was
constructed from a logical-order readout instead, omit `site_order`. The
returned regions are a snapshot: when a permutation changes the storage order,
rebuild them and create a new `PauliDMT` workspace before further compression.
This helper only maps regions; it does not convert an MPS wavefunction into a
density MPO.

Each constrained split preserves every selected region immediately before and
after **compression**, not across the preceding gate. The region list replaces
the default contiguous chain regions; include both explicitly if needed. A
per-split, log-scaled physical-region check raises on excessive numerical
error before committing that split. A standalone `inplace` sweep commits only
after the entire sweep succeeds. The protected subspace construction can reject
a bond budget even when some other constrained factorization might fit.

The full Pauli panel of each region is checked, so its cost grows as `4**k`
for `k` protected sites. Selected subspaces can require large bonds, and a
numerically resolved trace and stable precision are still required. Neither
positivity nor accuracy of the complete time evolution is guaranteed.
The adjacent-gate methods require MPO-neighboring sites, including for a 2D
lattice. For a physically local but MPO-nonadjacent pair, one available route
is `rho.apply_gate(gate, sites, atol=0.0).compress_dmt(..., regions=regions)`;
the exact gate embedding can substantially increase intermediate bond sizes.
There is no geometry-aware automatic nonlocal gate scheduler here.

The variants are explicit:

| `variant` | Truncation target | Use |
| --- | --- | --- |
| `"density"` (default) | Unprotected **connected** correlation block | Density operators with a numerically well-conditioned nonzero trace |
| `"operator"` | Unprotected **raw** block | General operators, including traceless Heisenberg observables |

The density variant follows [White et al., III](https://link.aps.org/accepted/10.1103/PhysRevB.97.035127);
the traceless variant follows [Yi-Thomas et al., Appendix D](https://arxiv.org/pdf/2310.06886).
Both use Quimb's public `array_split` with Autoray backend dispatch. The
implementation first exposes compact Schmidt support, protects the local
directions, truncates only the complementary block, and reconstructs through
thin factors and one directional QR. Existing-chain compression works with
the bond centre rather than forming a larger two-site tensor.

`max_bond` is a positive integer. `cutoff` defaults to zero. A numeric cutoff
or `"auto"` applies **only** to the unprotected singular spectrum; protected
directions have no physics cutoff and there is no post-truncation
renormalization. `"auto"` uses Pepsy's dtype-aware policy: `1e-12` for
`float64`/`complex128` and `1e-6` for `float32`/`complex64`, as in
`MpoOptimizer`. It differs from newer Quimb exact-SVD `"auto"` defaults.
This is a rank-selection heuristic, not an automatic error or stability bound.

`cutoff_mode` defaults to `"rel"`; `"auto"` and `None` also select `"rel"`.
The supported modes follow Quimb's interpretation, applied to the **residual**
singular values. For `"abs"`, `"sum1"` and `"sum2"`, units are those of the
un-normalized Hilbert–Schmidt norm of the represented operator, matching
ordinary `PauliMPO.compress_pauli`; internal Pauli-coefficient scaling is
accounted for. The relative modes are invariant under overall rescaling.

| Mode | Cutoff criterion on the residual spectrum |
| --- | --- |
| `"rel"` | Drop singular values below `cutoff * largest` |
| `"abs"` | Drop singular values below `cutoff` |
| `"sum1"` / `"sum2"` | Bound the sum of discarded values / squared values by `cutoff` |
| `"rsum1"` / `"rsum2"` | Bound those sums relative to the corresponding total |

For example, `cutoff="auto", cutoff_mode="rsum2"` uses a relative **squared**
weight threshold of `1e-12` in double precision; this is not a `1e-12`
norm-error threshold. `max_bond` can force additional truncation beyond the
cutoff, and the residual criterion is not a bound on later physical-observable
error. When an already fitting operator has no room under the conservative
protected-rank construction, it can be returned unchanged even with a cutoff.

Up to four protected directions on each side reserve up to
eight channels. If the conservative rank budget cannot accommodate them,
the operation raises rather than changing the constraints. An already fitting
operator can pass through with a smaller cap. This conservative rejection
does not prove no alternative lower-rank completion exists.

Only dense `float32`/`float64`/`complex64`/`complex128` cores on **open** chains
are currently supported.
Source backend, homogeneous dtype and device are retained. The initial and
final support factorizations use zero cutoff: small singular directions can
carry resolved local data on long chains. Protected-basis rank decisions
account for support conditioning, check the original partial-trace constraints,
and retry with a stricter rank tolerance when needed. The density variant
rejects a trace at roundoff relative
to the active operator norm, rather than regularizing division by the trace.

The centre and norms are rescaled to avoid underflow for long normalized
density matrices. Standalone `compress_dmt` additionally checks every
three-site marginal after the sweep using log-scaled trace environments,
with linear chain cost and a tolerance proportional to chain length and
machine precision. It raises `FloatingPointError` when this check fails;
increase `max_bond` or use double precision. Strongly correlated 128-site
mixtures at cap 12 can fail this check; cap 24 recovers their exact rank.
Neither floating-point arithmetic nor DMT guarantees preservation at an
arbitrarily small bond dimension.

The full marginal check applies to `compress_dmt` only. `apply_gate_dmt` and
`PauliDMT` also offer `trace_guard`, a per-gate comparison of the gated
target's trace with the split result. It raises `FloatingPointError` if a
single relative trace defect exceeds `32 * sqrt(machine_epsilon)`.

- `trace_guard="auto"` enables the guard for `variant="density"`. For
  `variant="operator"`, it enables the guard only when the initial trace
  coefficient exceeds eight machine epsilons times the global coefficient
  norm. This avoids treating ordinary roundoff in a traceless operator as a
  meaningful nonzero trace.
- `trace_guard=True` forces the guard when a nonzero trace is physically known
  but too small for that heuristic. `False` disables it; this can be useful
  for genuinely traceless observables with accumulated trace roundoff.
  `run.trace_guard_active` reports the resolved choice on `PauliDMT`.

This guard stops the previously observed large drift in the 128-site float32
identity-gate test when explicitly enabled, and catches a 192-site
double-precision example with `"auto"` before its trace becomes badly
corrupted. The raw/operator 128-site float32 input has an unresolved initial
trace by the auto heuristic, so its guard remains off unless explicitly
enabled. Auto classification is heuristic and can also mistake accumulated
roundoff for a nonzero trace. The guard does not undo preceding gates, cap
accumulated drift, or check all local marginals. Check trace, target observables,
bond dimension and precision convergence during long evolutions.

For NumPy input, one possible mitigation is to explicitly promote the cores
before constructing the driver, so its working state is double precision:

```python
from pepsy.optimizers import PauliDMT

rho64 = PauliMPO.from_pauli_cores([
    core.astype("float64") for core in rho.to_pauli_cores()
])
run = PauliDMT(rho64, max_bond=24, variant="density")
```

For complex cores use `complex128` instead. This changes the working and
output dtype; it reduced the error in one difficult 128-site test but does
not guarantee stability. Check physical observables and convergence even in
double precision.

`PauliDMTReport` and `PauliDMTBondReport` are exported from `pepsy.operators`.
Per-bond records contain input/output rank, protected ranks, residual rank,
discarded residual weight and support loss. The report also contains the
resolved numeric `cutoff` and `cutoff_mode`. Weights use the Pauli coefficient
norm, equivalent to the normalized Hilbert-Schmidt norm. Protected ranks are
reported as zero when an already-fitting bond takes a QR pass-through without
computing the protected bases; zero then means **not computed**. Constraint residuals
are collected only with `return_report=True`; `max_constraint_error` checks
the original bond matrix and physical constraint vectors.
`max_relative_constraint_error` divides each panel error by its target norm.
This ratio can be large for nearly zero partial traces of traceless operators;
the split guard uses a normwise backward error there. Absolute coefficient
residuals can be tiny while physical density-matrix errors are appreciable,
which motivates the additional standalone sweep check. These are separate from ordinary SVD norm-survival
diagnostics. With `inplace=True`, a failed standalone call leaves the source
unchanged.

### Local gates and persistent evolution

```python
rho = rho.apply_gate_dmt(
    two_qubit_gate, (3, 4), max_bond=32, variant="density",
    mode="conjugate",  # U rho U†; use "heisenberg" for U† O U
)
```

Sites must be adjacent and increasing. The two-site target is formed before
any deliberate truncation, without an embedded global gate MPO. The local
Pauli transfer matrix is real for conjugation in the Hermitian Pauli basis;
real coefficient cores remain real.

For repeated updates, `pepsy.optimizers.PauliDMT` owns the canonical centre and
lazily updated identity environments, avoiding whole-chain preparation per
gate. It supports explicit gates and second-order even/odd/even evolution:

```python
import numpy as np
from pepsy.optimizers import PauliDMT

L = 8
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
I = np.eye(2)
bond_h = np.kron(Z, Z) + 0.3 * (np.kron(X, I) + np.kron(I, X))
initial = PauliMPO.from_terms(L, [(1.0, (L // 2,), "Z")])
run = PauliDMT(
    initial, max_bond=32, variant="operator", mode="heisenberg",
    diagnostics=True,
)
evolved = run.evolve([bond_h] * (L - 1), dt=0.05, steps=20)
correlation = initial.inner(evolved, normalized=True)
# Or: run.apply_gate(two_qubit_gate, (3, 4))
```

The Hamiltonian is exactly `sum_j bond_hamiltonians[j]`, with one Hermitian
4-by-4 matrix per bond. Assign onsite fields to these matrices without double
counting your intended Hamiltonian. Gates are prepared once per `evolve` call
with backend-native Hermitian eigendecomposition. Shared Hamiltonian objects
reuse that decomposition and their half/full-step transfers within the call.
They are prepared again on the next call, allowing changed parameters and
fresh autograd graphs. No process-global linear-algebra registration is
performed. `run.state` is an
independent snapshot. `run.report` retains only the latest gate or final
timestep, so diagnostic memory does not grow with elapsed time. Successful
gates are committed as they execute; a failed evolution call can leave a
partially completed timestep. Standalone operations instead own disposable
workspaces.

The workspace carries its canonical centre and normalized identity-prefix/
suffix caches across updates, invalidating only affected ranges. Each parity
layer sweeps toward the opposite end; the next timestep starts near the
current centre. This avoids a full return sweep between timesteps. Gates
within a layer commute exactly, but finite-bond DMT compression is nonlinear:
the centre-aware layer order is part of this approximation. The same order
continues when evolution is split across calls. Prefer this persistent driver
over repeatedly invoking standalone `apply_gate_dmt`, which must prepare its
own canonical workspace each time.

For fixed-rank Torch differentiation, use the existing
`TorchLinalgConfig(stabilized=True)` policy and verify objective gradients.
Adaptive support/rank decisions are eager; end-to-end JIT, degenerate-spectrum
derivatives, GPU and native symmetry support are not established by CPU forward
checks. Near-singular support or tied spectra can make the selected global
approximation sensitive to roundoff and internal gauge choices. The local
partial-trace checks have the numerical limitations described above. DMT
preserves local data at each split in exact arithmetic; it does not guarantee global
positivity, exact later local dynamics or zero Trotter error. Check timestep
and bond convergence independently. Implementation evidence and limitations
are in the [DMT design note](../../development/notes/2026-11-13-pauli-dmt-design.md).

## DAOE weight filter

`apply_daoe(max_weight=..., gamma=...)` applies the artificial dissipation
in [DAOE, Eq. (1)](https://arxiv.org/abs/2004.05177). For a Pauli string $P$,

$$
\mathcal D_{\ell_*,\gamma}(P)=
{\color{blue}e^{-\gamma\max(w(P)-\ell_*,0)}}P,
\qquad
{\color{purple}w(P)=\#\{j:P_j\ne I\}}.
$$

`max_weight` is $\ell_*$, a nonnegative integer; `gamma` is a finite,
nonnegative real damping **per filter application**. Weight counts nonidentity
factors: `XIIIZ` has weight two despite spanning five sites. Strings of weight
at most `max_weight`, including the identity, are unchanged.

Sparse terms stay sparse. Native cores use a diagonal counting superoperator
with bond dimension at most `max_weight + 1`, multiplying the existing
virtual bonds by at most this factor. There is no expansion into Pauli words
and no SVD inside the filter. NumPy, Torch, JAX and CuPy dense core paths
retain dtype, device and coefficient gradients; integer cores are promoted
to floating point. `max_weight` and `gamma` are static, nondifferentiable
parameters. `inplace=True` replaces the operator and clears its old compression
report; a no-op filter preserves the report.

### Reusable damping MPO

Build `PauliDampingMPO` once when applying the same filter repeatedly:

```python
from pepsy.operators import PauliDampingMPO, PauliMPO

damping = PauliDampingMPO(8, max_weight=2, gamma=0.1)
observable = PauliMPO(8, [
    (1.0, "ZIIIIIII"), (0.4, "XXXIIIII"),
]).compress_pauli(cutoff=0.0)
filtered = observable.apply_damping(damping)  # also damping.apply(observable)
filtered, report = filtered.compress_pauli(
    max_bond=64, cutoff=1.0e-12, renorm=False, return_report=True,
)

superoperator = damping.to_mpo()  # explicit Quimb MPO, dimension 4 per site
diagonal_cores = damping.to_diagonal_cores()  # (left, right, 4), I/X/Y/Z
print(damping.bond_dimensions, damping.factor("XYZIIIII"))
```

The exported MPO has local input/output dimension **four**, acting on a
dimension-four MPS of Pauli coefficients. A physical qubit MPO has local
input/output dimension two. `superoperator` represents the linear map
$\mathcal D[O]$; physical multiplication `damping @ observable` is not its
application API.

For an observable core $A$ and damping core $W$, application contracts

$$
{\color{purple}B_{(l,a),(r,b),p}}=
{\color{blue}A_{l,r,p}}\,W_{a,b,p,p}.
$$

The implementation retains only $W_{a,b,p,p}$ and applies this diagonal
directly. Exported arrays are independent copies; changing them cannot alter
later uses of the same filter. `to_mpo(like=sample_array)` and
`to_diagonal_cores(like=sample_array)` select the sample's dense backend,
dtype and device. Parameters are frozen and static; coefficient gradients
pass through `apply`.

`apply_daoe(max_weight=..., gamma=...)` constructs this same filter for one
application. Reuse the object to amortize construction across steps or
observables. No-op filters return independent operator copies by default;
`inplace=True` returns the input object. Actual filtering clears an old
compression report. The full exported dense matrix would be
$4^L\times4^L$ and is suitable only for small reference checks.

A small reference trajectory composes the existing Heisenberg gate, the new
filter, and explicit compression:

```python
from scipy.linalg import expm
from pepsy.operators import PauliDampingMPO, PauliMPO

h = PauliMPO(2, [(0.8, "ZZ"), (0.4, "XI"), (0.3, "IX")])
gate = expm(-0.1j * h.to_dense())  # small-system reference only
initial = PauliMPO(2, [(1.0, "ZI")]).compress_pauli(cutoff=0.0)
observable = initial.copy()
damping = PauliDampingMPO(2, max_weight=1, gamma=0.1)
correlations = []
for _ in range(10):
    observable = observable.apply_gate(gate, (0, 1), mode="heisenberg")
    observable = observable.apply_damping(damping)
    observable, report = observable.compress_pauli(
        max_bond=4, cutoff=1.0e-12, renorm=False, return_report=True,
    )
    correlations.append(initial.inner(observable, normalized=True))
```

`inner(..., normalized=True)` evaluates $2^{-L}\operatorname{Tr}(A^\dagger B)$,
the infinite-temperature correlation convention. This is a finite-time
calculation. The filter action is exact up to floating-point arithmetic;
the resulting evolution at nonzero `gamma` approximates unitary dynamics.
Study damping, filter interval, weight threshold, time step and bond cutoff
separately. The filter is not a local Kraus noise channel and does not preserve
the Hilbert-Schmidt norm. Keep `renorm=False` when tracking amplitudes.

This is the first DAOE building block. Local gates currently embed a global
gate MPO and multiply native cores; a scalable local TEBD/TDVP driver,
transport benchmarks, extrapolation workflow and native Symmray support
remain future work. CuPy/device execution is implemented through shared
dispatch. JAX placement is checked on a second logical CPU device;
CuPy and accelerator execution remain unverified.

## Gates and channels

Small dense local gates can be brought into the same basis automatically. The
decomposition enumerates `4**k` local strings for a `k`-qubit gate, so this is
intended for one- and few-qubit gates:

```python
gate_op = PauliMPO.from_dense(gate)
# or inspect the local expansion directly
from pepsy.operators import decompose_pauli
terms = decompose_pauli(gate)
```

`apply_gate` transforms an existing Pauli expansion locally and combines the
result back into canonical Pauli strings. It supports non-contiguous sites:

```python
rotated = op.apply_gate(gate, where=(2, 5), mode="conjugate")
heisenberg = op.apply_gate(gate, where=2, mode="heisenberg")
left_product = op.apply_gate(gate, where=2, mode="left")
```

The modes mean `G P G†`, `G† P G`, `G P`, and `P G`, respectively. Kraus maps
are available through `apply_channel`, with either Heisenberg or Schrödinger
picture:

```python
observable_after_channel = op.apply_channel(kraus, where=2)
operator_after_channel = op.apply_channel(
    kraus, where=2, picture="schrodinger"
)
```

MPS and MPO operations use the established tensor-network boundary:

```python
value = op.expectation(mps)
new_mps = op.apply(mps, max_bond=128, cutoff=1.0e-10)
mpo = op.to_mpo()
compressed, report = op.compress(max_bond=64, return_report=True)
# compressed is a PauliMPO with I/X/Y/Z physical legs
numerical_mpo = op.compress(basis="mpo", max_bond=64)
```

Numerical compression through `basis="mpo"` returns a Quimb
`MatrixProductOperator`, because a generic low-rank MPO does not remain sparse
in the Pauli basis. Similarly,
`exp()` and `time_evolution()` return the existing semantic/numerical MPO
objects rather than claiming that a generally dense exponential is a sparse
Pauli expansion.

`partial_trace` traces out the supplied sites by default. Pass `keep=True` to
interpret the supplied sites as the sites retained in the reduced operator.
The default is the ordinary unnormalized partial trace; use
`normalized=True` for normalized Pauli coefficients.

The current implementation targets finite one-dimensional qubit MPOs. A
two-dimensional square-lattice operator should use a PEPO or a deliberate
2D-to-1D mapping; the Pauli basis does not remove the associated contraction
cost. Native 2D Pauli-PEPO support is intentionally a separate follow-up from
this 1D PauliMPO boundary.
