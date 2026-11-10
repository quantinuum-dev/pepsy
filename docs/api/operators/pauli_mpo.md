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
