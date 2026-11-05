# Independent six-site cluster batches

The opt-in prototype batches environment contractions and ALS for consecutive
disjoint patches, with modest measured gains and higher GPU memory use.

## Implemented scope

`PreparedLoopClusterUpdate(..., batch_size=4)` accepts at most four consecutive
two-site updates with disjoint complete patches and matching contraction
programs, operand shapes and reduced dimensions. Its default is one.
One-site actions, overlaps, incompatible shapes and `apply` boundaries end a
batch. It does not search ahead, reorder gates or batch independent networks.

Both environment programs and ALS use a leading batch axis. Each ALS member
keeps its retained rank, Cholesky certificate/spectral fallback and complete
sweep stopping. Finished members leave the active solve batch. The public
`TorchLinalgConfig` dispatch owns QR/SVD; eager is required. Preparation and
final local gauge extraction remain sequential.

All fits in a batch read a common state snapshot. Only the active endpoint
tensors and gauge are committed for each member, in original order, together
with its RMS/exponent change and scalar tracker record. The gauge merge must
retain unrelated bonds committed by preceding members. Callbacks must be
observational because later members may already have been evaluated.

The footprint condition is sufficient for independence:

$$
{\color{blue}C_i}\cap{\color{purple}C_j}=\varnothing,
\qquad i\ne j,
$$

where each C includes both active sites and every spectator in its environment.
Shared cut-bond gauges are unchanged by either update. The prototype uses this
conservative condition even when overlapping spectator-only regions could
also permit independent updates. It preserves the sequential approximation;
it does not substitute stale environments for dependent gates.

## Bounded timing measurement

Synthetic complex128 PEPOs on 3x10, seed 931, with uniform virtual bond D.
Four horizontal interior updates at rows 1 and column pairs (1,2), (3,4),
(5,6), (7,8) have disjoint six-site patches. Gates alternate upper/lower and
use diagonal phases .11, .15, .19, .23. Identity initial SU gauges; max bond D;
Hermitian-only environments; ALS cap6/tolerance1e-8; local fidelity enabled.

Use one numerical host thread, native Torch linalg, CUDA `gesvd` and greedy
contraction. Construct each plan before timing. Alternate sequential/batch4
over five repetitions; discard two warm-up repetitions and report medians of
three. Include input copies, pair preparation, contractions, all ALS work,
local diagnostics, gauge extraction and writeback. Synchronize CUDA around
each timing. The workstation is shared; ratios are workload measurements.
Hardware is an Intel Xeon W-3375 CPU and NVIDIA RTX A5000 GPU.

| Device / D | Sequential ms | Batch4 ms | Ratio |
| --- | ---: | ---: | ---: |
| CPU / 2 | 82.561 | 59.702 | 1.383x |
| CUDA / 4 | 176.619 | 129.366 | 1.365x |
| CUDA / 6 | 234.833 | 195.587 | 1.201x |
| CUDA / 8 | 612.000 | 577.925 | 1.059x |

All matched pair iteration counts agree; normalized errors and local fidelities
agree within 1e-8, generally near floating-point roundoff. These are seeded
synthetic states, not evolved high-D results or a full T=6 trajectory.

Peak allocated Torch CUDA bytes, including returned results, are 19,386,368
versus 34,234,880 at D4; 89,132,032 versus 229,973,504 at D6; and 371,293,184
versus 1,170,529,792 at D8 (sequential versus batch4). These are process-local
allocations, not device-wide use. The batch preflight scales contraction
workspace and FLOP estimates by the member count; it does not bound every
ALS or result allocation.

## Opportunities in the existing Eli schedule

A read-only geometry scan of the complete homogeneous depth20 T=6/dt=.05
middle-out schedule, preserving all source boundaries, gives these upper
bounds before checking current shape compatibility:

| Geometry | Pair actions | Singleton groups | Groups of 2 | Groups of 3 |
| --- | ---: | ---: | ---: | ---: |
| 5x5 corner-removed | 5,040 | 4,100 | 440 | 20 |
| 10x10 corner-removed | 24,640 | 19,920 | 2,180 | 120 |

Thus approximately 19% of pair actions are even eligible for this consecutive
grouping policy; rank/shape differences can reduce that further. Synthetic
batch4 speedups cannot be applied to the whole trajectory. The existing
downstream runner does not yet pass this opt-in option. No simulation was
started or restarted for this scan.

## Upstream and validation

Classification: **prototype**. The existing public Cotengra array contractor
is lifted with public `torch.vmap`; registered Autoray QR/SVD handle leading
batch axes. No installed library, dependency or linalg registration changed.
Installed versions: Quimb `1.15.1.dev101+g1b64353d9`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3`, Symmray
`0.4.1.dev25+gdc377f033`, Torch `2.14.1+cu126`.

Checked the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra guide](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray array documentation URL was unavailable; native installed source
and the repository provide the fallback. Native Symmray remains outside the
new dense batching route. Installed Cotengra signatures and numerical batch
probes were checked; Torch built-in linalg callables lack inspectable Python
signatures, so their configured batch capabilities were checked by execution.

Focused tests cover CPU/CUDA, complex64/complex128, Hermitian/PSD environments,
physical reconstruction, per-member stopping, rank growth, overlaps, shape
fallback, one-site/source barriers, workspace preflight, and gate/gauge
derivatives with native/stabilized linalg. Mixed Cholesky/spectral gradient
checks exclude failed Cholesky factors from autograd.

The initial broad selection had one new test failure from asserting an
unnecessary cache miss on active-bond growth. Active rank growth need not
change an environment's operand shapes. The corrected regression checks
actual bond growth and the independent analytic physical matrix element.
A first gradient test also used a local core norm that omitted the RMS
exponent; the final check differentiates a full physical matrix element.
Neither correction changes the numerical implementation or loosens tolerances.
Final validation totals are recorded in the
[session handoff](../../../history/2026-11-05-independent-cluster-batches.md).

Reproduction script and final samples:
`/tmp/pepsy-cluster-batch-bench-20261105.py` and
`/tmp/pepsy-cluster-batch-bench-20261105-final.json`.
The script requires the selected Python environment and local `pepsy/src`
on `PYTHONPATH`. Schedule counts are in
`/tmp/pepsy-cluster-batch-schedule-20261105.json`.
