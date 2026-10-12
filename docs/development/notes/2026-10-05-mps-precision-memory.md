# 2026-10-05 — MPS ledger precision and trajectory memory planning

Scope: the user requested resolving the three outstanding complex64 ledger
comparisons and implementing automatic GPU budgeting for retained trajectory
states. Baseline: `develop` / `2039fbc`. These changes are uncommitted; unrelated
working-tree changes were preserved.

## Precision finding and adopted correction

Reproduced the original failures unchanged:

- `test_backend_ledger_matches_cpu_and_preserves_state_dtype[cuda-False]`
- `test_backend_ledger_matches_cpu_and_preserves_state_dtype[cuda-True]`
- `test_backend_ledger_matches_cpu_and_preserves_state_dtype[jax-True]`

Their infidelity deviations from NumPy were approximately +3.84e-6, +3.70e-6,
and -4.05e-6, respectively, against the existing 3e-6 bound. Double-precision
scalar bookkeeping and even temporary double-precision norm reductions did
not resolve the drift already present in compressed states. The previous
configured CUDA `gesvd` comparison passed, but installing a hidden global
SVD policy would violate the caller's numerical policy.

**Adopt:** prepare the small dense gate's MPO factors with a double-precision
operator workspace, then cast factors back on the same device before ordinary
MPS compression. This uses public `MatrixProductOperator.from_dense` and
`gate_with_submpo_`, with existing operator split defaults and state compression
options. It applies only to float32/complex64 dense operators with at most 256
entries. It does not promote MPS tensors, change state truncation or ledger
values, register SVD drivers, or download operator/state arrays. The original
ten ledger tests pass, including all three previously failing cases.

Torch gradients through casts and factorization remain intact. JAX's temporary
x64 scope is restored on success and failure. A traced SVD cannot safely use a
temporary x64 scope whose backward pass runs after that scope exits: an actual
probe raised a mixed-precision dot type error. Traced operators with x64
disabled therefore keep the previous factorization path. With x64 enabled,
the double workspace is supported through differentiation. Tests compare
complete gate application and gradients with dense references under the
optimizer's existing highest-matmul-precision contract. Dynamic Quimb SVD
truncation is not JIT-compatible; this change does not claim general replay
JIT support. Metal, larger operators and native symmetric routes retain their
prior factorization paths.

## Automatic memory policy

**Adopt:** `MpsOptimizer.run(..., memory_budget="auto")` queries the active
allocator once for local shots. Torch CUDA accounts for driver-free memory,
reserved-but-unused cache and the per-process allocation fraction. CuPy
accounts for the current default pool's reusable blocks and its limit; custom
allocators conservatively use driver-free bytes. JAX uses public per-device
allocator statistics when they expose a limit and usage. No allocator settings
or device defaults are modified. CPU/unknown-query paths remain unbudgeted.

Half the available bytes become the automatic allowance. A positive integer
sets a byte allowance explicitly; `None` disables it. The planner projects
state storage from physical dimensions and the greater of `chi` and initial
bond size, with a floor at initial storage. Exact replay uses a dense-vector
estimate, including a child `run_kwargs` mode override. Native charge blocks
use conservative dense dimension estimates without materializing them.

The planner reserves eight estimated states for active work and 32 MiB for
probability workspace. Remaining capacity allows two state copies per branch
for parent/child overlap. It tightens the existing branch cap and automatic
worker budget; explicit `chi`, cutoff, dtype/device and retention are unchanged.
Independent retained results must fit every shot; streaming `retain="none"`
must fit the active worker count. Insufficient capacity raises `MemoryError`.
Automatic coalesced overflow is caught outside the lower runner: its exception
traceback is released before checking/replaying independent shots, preserving
the seed and preventing the failed frontier from overlapping the retry.
Explicit coalesced runs preserve their branch-cap error and never prune.

Diagnostics expose the byte budget, per-state estimate, capacity and reason.
No trial replay is used to select policy. This is planning rather than a hard
allocation limit: large intermediate operators, unusual compression workspace,
autograd graphs, cached payload growth, allocator fragmentation and concurrent
processes can exceed the allowance. **Defer:** allocator reservations, graph
memory prediction, multi-rank GPU budgeting, and continuation from a coalesced
frontier. MPI rejects explicit byte budgets; its prior chunking remains intact.

## Installed capabilities and upstream audit

Python 3.12; NumPy 2.5.2; Torch 2.6.0+cu124; CuPy 14.1.1; JAX 0.10.2;
Quimb 1.15.1.dev79+gb5e316200; Autoray 0.11.1.dev9+g1291702f9;
Cotengra 0.8.3.dev7+g1d7fd333f; Symmray 0.4.1.dev11+g1a3481803.
CUDA execution and allocator queries succeeded on the RTX A5000. Temporary
Numba cache output was directed to `/tmp`; no installed packages were edited.

Inspected installed `Tensor.norm`, `MatrixProductState.gate_nonlocal`,
`MatrixProductOperator.from_dense`, direct compression, Autoray namespace
dispatch, `torch.cuda.mem_get_info(device)`, allocator fraction/cache queries,
CuPy `memGetInfo()` and memory pool methods, and JAX precision/memory APIs.

Rechecked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The Symmray abelian
array page remained unavailable; installed APIs and official source supplied
the fallback. The newer Quimb changelog reports changed decomposition defaults;
this patch uses installed public APIs and does not adopt those new defaults.

Read the [CuPy pool contract](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html)
and [JAX x64 context](https://docs.jax.dev/en/latest/_autosummary/jax.enable_x64.html).
Torch documentation endpoints redirected or were unavailable; inspected the
installed Torch 2.6 signatures/source rather than assuming newer capabilities.
The legacy JAX `experimental.enable_x64` name is a narrowly scoped compatibility
fallback when the current public context name is absent. No custom SVD kernel
or copied upstream implementation was introduced.

## Validation

The precision workspace has a measurable cost. On the RTX A5000, using a
12-site random complex64 MPS with initial/maximum bond 16 and eight copies of
one random two-qubit unitary on `(i, i+3)`, direct replay with `cutoff=0` took:

| Backend | Previous small-factor precision | Double operator workspace |
| --- | ---: | ---: |
| Torch CUDA | 49.83 ms | 64.39 ms |
| CuPy | 62.15 ms | 63.46 ms |

Medians of seven alternating measurements after two warmups per path, with
device synchronization and one OpenBLAS/OMP thread. State/optimizer creation
was outside the timed region. No other agent-owned GPU test ran during the
reported measurement; the GPU remains a shared resource. The approximately
29%/2% cost on this small workload is a correctness tradeoff, not a claim about
large-bond throughput. An earlier concurrent timing was discarded. No CUDA-Q
comparison, retained-state peak-memory measurement or universal speed claim.

Final results are recorded in the [session handoff](../../../history/2026-10-05-mps-precision-memory.md).
Memory tests exercise real Torch CUDA, CuPy and JAX GPU queries/replay, native
fermionic metadata, seeded distributions, retention errors, cap fallback,
failed-frontier lifetime, exact-mode size estimates, allocator quota accounting
and unsupported-query behavior. Gate tests cover Torch CPU/CUDA and JAX
gradients, dtype preservation and precision restoration.
