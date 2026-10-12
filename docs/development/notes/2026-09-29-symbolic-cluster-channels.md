# 2026-09-29 — Exact MPO transition sharing and chi convergence

This extends [fused compact assembly](2026-09-29-fused-cluster-channels.md).
Implementation stays in Pepsy; Gaugy remains a public-API caller. The scope
is spin/Pauli operators, including native bosonic charge sectors.

## Implemented

`prepare_cluster_channels(..., structural_reuse=True)` now performs an exact
MPO transition quotient before reference QR. Every entry of every residual
matrix is a separate formal atom. Alternating right-to-left and left-to-right
sweeps merge identical continuations or prefixes within the same charge sector.
Formal zero continuations can be removed; numerical reference zeros cannot.
The neutral singleton rail stays separate.

Merging uses selection at one endpoint and addition at the other. These are
separate maps, not an orthogonal projector: using one map and its conjugate
would change path multiplicity. The maps are composed with the chosen QR
bases and absorbed into the existing fused local contraction schedule.
Replay still constructs actual MPO/PEPO tensors without expanded history
buffers, unprojected collection products, SVD, QR or numerical rank decisions.
Traces measure these constructed operators.

This is conservative weighted-automaton sharing, motivated by
[Crosswhite and Bacon](https://arxiv.org/abs/0708.1221), not an implementation
of a globally minimal weighted automaton. Distinct residuals are independent
even when further Hamiltonian identities might relate them. Existing
coefficient-binding-aware exponential reuse remains separate. The package's
`SparseVirtualTensor.apply_axis_groups` supplies the gather/merge primitive;
`MPOAutomaton` itself does not provide minimization.

The one-dimensional quotient is MPO-only. Graph/square PEPOs keep their
connectivity-aware fused tree/routing implementation. Applying these prefix
rules directly to branching or loops would require a separate correctness
argument; see the [cluster-TNO construction](https://arxiv.org/html/2112.01507).
Ordinary source `exp()` policies remain unchanged. `structural_reuse=False`
retains the previous channel-plan gauge. Capped MPO approximations can change
under the new default gauge even though the uncompressed operator is preserved.

## Reproducible measurement

Run [the benchmark](../../../examples/compact_cluster_channels_benchmark.py):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python examples/compact_cluster_channels_benchmark.py --profile-memory
```

Shared Python 3.12, Torch 2.6.0+cu124, CPU complex128, Xeon W-3375. Six-site
open chain, spatial cluster cutoff four, ordered factors `a sum ZZ` followed
by `b sum X`, step `-.1j`. One reference `(a,b)=(.3,.2)`, held-out values
`(.31,.19)`, default QR `rtol=1e-12`. Timings are medians of seven calls after
two warmups, separately measuring constructed-operator normalized trace and
its backward. Preparation and profiler overhead are excluded from timings;
no compiler speedup is claimed. The benchmark source prints exact dimensions,
preparation times, projection operands and saved-entry counts as well.

Errors compare with the **uncompressed cluster expansion at the same spatial
order**, not the exact global exponential. Operator error is relative
Frobenius norm. Gradient error is the absolute Euclidean error of
`real(trace(U)/64) + .23 imag(trace(U)/64)` with respect to both parameters.

| chi cap | actual max bond | operator error | gradient error | forward / backward ms | peak / initial tensor bytes |
| --- | ---: | ---: | ---: | --- | --- |
| Unshared, uncapped | 37 | 0 | 8.7e-18 | 23.5 / 8.3 | 864,096 / 61,592 |
| 1 | 1 | 6.93e-2 | 1.55e-2 | 7.9 / 7.1 | 273,512 / 35,536 |
| 3 | 3 | 3.11e-2 | 1.24e-2 | 10.0 / 9.2 | 274,128 / 36,152 |
| 6 | 6 | 2.17e-3 | 1.24e-2 | 9.7 / 7.7 | 275,736 / 37,760 |
| 12 | 11 | 4.32e-5 | 7.08e-6 | 13.1 / 10.7 | 282,600 / 44,624 |
| 20 | 11 | 4.32e-5 | 7.08e-6 | 10.0 / 8.0 | 279,784 / 41,808 |
| 25 | 25 | 1.39e-16 | 6.94e-18 | 19.8 / 10.3 | 508,576 / 39,960 |
| Shared, uncapped | 25 | 1.39e-16 | 6.94e-18 | 16.4 / 8.3 | 505,376 / 36,760 |

Torch's public profiler memory timeline measures live tracked CPU tensor
allocations during forward and backward. Initial tracked allocations are
shown because cached/preexisting tensors vary across runs. These counts
exclude Python/NumPy preparation storage, opaque workspaces and process RSS;
they are not whole-process peak memory. The measurements support reduced
tensor storage on this bounded case, not universal throughput claims.

Exact sharing changes bond dimensions `(13,33,37,33,13)` to `(5,25,25,25,9)`:
129 to 89 total channels and 13,304 to 6,456 dense output entries, with no
approximation. Preparation rose from about 10 ms to 19–26 ms in this probe.
At cap six the largest planned projection operand/result is 256 entries,
versus 4,884 unshared/uncapped. The local cluster cost therefore remains visible
when the cap gets small.

Caps 12 and 20 select the same ranks because the single reference's remaining
QR directions are below threshold. The slight worsening of gradient error
from cap three to six also shows why operator error alone is insufficient.
This method promises neither monotonic global errors nor a global tolerance.
More references/tangent snapshots or full structural dimensions are separate
accuracy choices. Full dimensions recover the operator family exactly.

## Validation and limits

New deterministic tests use independent random complex residual entries,
compare full operators and residual adjoints, cover crossing graph paths,
U1/Z2/U1U1/Z2Z2 sectors and repeated charges, and check identical maps for equal, zero and unequal
reference bindings. Live parameter derivatives agree in the uncapped limit,
including zero initialization. Existing CPU/CUDA, JAX trace compilation and
Torch full-graph assembly checks exercise the new default.

Preparation still enumerates the reference topology and symbolic sparse
transitions. The quotient reduces channel dimensions but does not eliminate
all duplicate local contraction recipes or compile a minimal automaton
directly from interaction strings. Dense local exponential/residual costs
still grow with cluster order. Fixed positive caps approximate the target;
no claim of arbitrary exact small chi, scalable compact recursive preparation,
native fermionic/PEPO support, or environment-optimal projection is made.

**Upstream decision:** adopt Pepsy's existing sparse grouping primitive; no
new compatibility shim or dependency change. The same-task upstream audit
and installed versions from the preceding study remain applicable. See the
[validation and working-tree handoff](../../../history/2026-09-29-symbolic-cluster-channels.md).
