# DMT for 1D chains and 2D lattice regions

Use `PauliMPO` for native coefficient storage and repeated gate evolution,
or [`compress_mpo_dmt`](../operators/mpo_dmt.md) for ordinary ket/bra MPO
compression with the same protected SVD kernel. **1D** protects every
contiguous three-site reduced operator by default. **2D** protects explicit
physical neighborhoods mapped into the same open MPO chain. This is a
geometry-aware MPO method; storage, gate positions and chain evolution retain
their 1D meaning.
Supported storage uses dense `float32`, `float64`, `complex64` or `complex128`
cores on an open chain.

For ordinary MPO gate-stream replay, use
[`MpoOptimizer(mode="dmt")`](mpo.md#dmt-gate-stream-replay), which applies each
gate exactly before protected compression.

Optional `truncation="optimized"` adds guarded fixed-border SVD and constrained
bond refinement; `region_selection="min_cover"` shares protection strings
across overlapping regions. Both retain the original defaults. See the
[controls and optimality limits](../operators/mpo_dmt.md#optional-improved-bond-truncation).
`region_selection="rank_aware"`, `constraint_mode="relaxed"`, and
`check_regions=False` have the same meaning for this persistent driver and
ordinary MPO compression; see the same guide for reporting limitations.

## Choose an operation

| Task | API | Result and mutation |
| --- | --- | --- |
| Compress an existing operator | `rho.compress_dmt(max_bond=..., regions=...)` | New `PauliMPO`; `inplace=True` commits after success |
| Compress ordinary MPO tensors | `compress_mpo_dmt(mpo, max_bond=..., regions=...)` | New Quimb MPO; optional transactional `inplace=True` |
| Apply one adjacent gate | `rho.apply_gate_dmt(U, (j, j+1), max_bond=..., regions=...)` | New `PauliMPO`; optional `inplace=True` |
| Reuse environments across adjacent gates | `run = PauliDMT(rho, max_bond=..., regions=...)`; `run.apply_gate(U, (j, j+1))` | Updates `run`; returns one `PauliDMTBondReport` |
| Evolve a nearest-neighbor chain | `run.evolve(bond_hamiltonians, dt=..., steps=...)` | Updates `run` and `run.time`; returns an independent `PauliMPO` |
| Choose physical square-lattice regions | `square_lattice_dmt_regions(lx, ly, mapper=..., kind=...)` | MPO-position tuples; no state mutation or gate schedule |

`run.state` copies the live operator. Read it when a snapshot is needed;
repeated snapshots inside a gate loop add allocation and copy costs.
Standalone methods accept `return_report=True` to return `(operator, report)`.
The persistent driver uses `diagnostics=True` and `run.report`; reports retain
the latest gate or final timestep. These flags collect extra diagnostics;
preservation checks remain active without them.

`apply_gate_dmt` and `PauliDMT.apply_gate` cap only the updated bond; they do
not compress other bonds of an initially oversized operator. If a global cap
is required, start with `rho.compress_dmt(max_bond=...)` before applying
gates. `MpoOptimizer(mode="dmt")` instead caps all bonds on its first gate.

## 1D evolution

```python
import numpy as np
from pepsy.operators import PauliMPO
from pepsy.optimizers import PauliDMT

L = 6
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
I = np.eye(2)
rho = PauliMPO.from_terms(L, [
    (2.0**-L, "I" * L),
    (0.1 * 2.0**-L, (0,), "Z"),
])
h = np.kron(Z, Z) + 0.2 * (np.kron(X, I) + np.kron(I, X))
run = PauliDMT(rho, max_bond=16, cutoff="auto", diagnostics=True)
rho_t = run.evolve([h] * (L - 1), dt=0.03, steps=3)
rho_t = run.evolve([h] * (L - 1), dt=0.03, steps=2)
print(run.time, rho_t.trace(), run.report.final_bond_dimensions)
```

The Hamiltonian is **exactly** the sum of the supplied `L-1` two-site matrices.
Here the interior onsite field is twice the endpoint field. Distribute fields
among bond terms according to the intended Hamiltonian. Evolution uses a
second-order even/odd/even schedule; its finite-bond order follows the
canonical centre. Explicit `apply_gate` calls do not advance `run.time`.

Use `variant="density", mode="conjugate"` for `U rho U†`. For a traceless
observable, use `variant="operator", mode="heisenberg"` for `U† O U`.
`variant` selects the protected truncation rule; `mode` selects gate direction.
Hamiltonian preparation retains the source backend and precision. Trainable
Torch Hamiltonians use native matrix exponentials to handle degeneracies.

## 2D regions and physically local gates

Use the same mapping to construct the operator and its protected regions:

```python
import numpy as np
from scipy.linalg import expm
from pepsy.operators import PauliMPO, square_lattice_dmt_regions
from pepsy.tensors import OneDMap

mapper = OneDMap(3, 2, mode="snake")
_, site = mapper.build()
regions = square_lattice_dmt_regions(3, 2, mapper=mapper, kind="plaquettes")
rho = PauliMPO.from_terms(6, [
    (2.0**-6, "IIIIII"),
    (0.1 * 2.0**-6, (site[(0, 0)],), "X"),
])
rho = rho.compress_dmt(max_bond=32, regions=regions)

Z = np.diag([1., -1.])
U = expm(-0.03j * np.kron(Z, Z))
where = (site[(0, 0)], site[(1, 0)])  # MPO positions (0, 3)
target = rho.apply_gate(U, where, mode="conjugate", atol=0.0)
rho, report = target.compress_dmt(
    max_bond=32, cutoff="auto", regions=regions, return_report=True,
)
print(rho.trace(), report.final_bond_dimensions)
```

The gate is applied exactly before DMT, then the sweep preserves the selected
reduced operators of that **gated target**. Nonadjacent gate embedding can
increase intermediate bonds substantially. For pairs adjacent in MPO storage,
`apply_gate_dmt` or persistent `PauliDMT.apply_gate` avoids global embedding.
`PauliDMT.evolve` still takes `L-1` adjacent chain terms with `regions=...`;
it does not construct the missing square-lattice edges or a 2D Trotter schedule.

`kind="l_shapes"` protects L-shaped triples; `"triples"` also includes straight
triples. `regions` **replaces** default 1D protection. Mapping modes, explicit
regions, and permuted storage via `site_order` are detailed in the
[region contract](../operators/pauli_mpo.md#physical-lattice-regions).

## Accuracy and performance controls

The protected data obeys, to numerical precision,

$$
\operatorname{Tr}_{\bar R} {\color{purple}\rho_{\mathrm{compressed}}}
=\operatorname{Tr}_{\bar R} {\color{blue}\rho_{\mathrm{target}}},
\qquad R\in\mathcal R.
$$

Here `target` is the pre-compression operator, after any gate. This preserves
energy through compression when every Hamiltonian term lies in a protected
region; it does not eliminate Trotter or accumulated evolution error.

| Control or cost | Practical meaning |
| --- | --- |
| `max_bond` | Conservative protected budget plus unprotected residual rank; a small cap can be rejected |
| `cutoff`, `cutoff_mode` | Paper mode applies these only to the free residual; optimized refinement can reuse rank after its SVD seeds; relaxed fallback uses zero-cutoff seeds |
| `cutoff="auto"` | `1e-12` in double precision, `1e-6` in single precision; a rank heuristic |
| 1D protection | At most four protected directions per side; local support/SVD costs grow with bond size |
| Explicit regions | More protected directions, full `4**k` output panels for `k` sites, and contractions along their MPO spans |
| Persistent driver | One initial canonicalization; subsequent gates reuse and invalidate environments locally |
| Nonadjacent lattice gates | Exact embedding followed by a full sweep; often a larger cost than adjacent updates |

Keep double precision for difficult long-chain problems. Guards can reject
unresolved traces or excessive marginal errors. Default 1D standalone sweeps
check every three-site marginal; gated updates use local constraints and the
resolved trace guard. Explicit-region paths additionally check every selected
region at each split when `check_regions=True`. With `check_regions=False`,
the bond guards remain but full-region defects are not measured. Positivity and accurate long-time dynamics require
separate convergence checks. Successful persistent gates survive a later
failure, which can leave a partial timestep.

Run the [reproducible CPU benchmark](../../../examples/pauli_dmt_benchmark.py)
from the repository root, with the project's environment active and optional
`threadpoolctl` installed:

```bash
python examples/pauli_dmt_benchmark.py --case all --repeats 5 --profile
python examples/pauli_dmt_benchmark.py --case 2d-compress --dtype complex128
```

Timings exclude independent readout checks and first-call setup. Evolution
cases include workspace creation and two chain timesteps, with lattice
protection on the 2D case. JSON rows report samples, trace error, actual
discarded residual norms in Pauli-coefficient units, and an independent region
oracle for compression.
Use profiles to separate centre movement, SVD/protected splitting, and region
verification. See [measured bottlenecks](../../development/notes/2026-11-16-dmt-performance.md)
and the [full numerical contract](../operators/pauli_mpo.md#protected-density-matrix-truncation-dmt).
