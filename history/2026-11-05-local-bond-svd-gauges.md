# 2026-11-05 — Local diagonal gauges after BP reduced updates

- Scope: remove whole-network `gauge_all_simple` from reduced cluster gate
  updates and optional SU output of reduced compression.
- Branch / baseline: `develop`, `ee22198`.
- Commit status: uncommitted working-tree changes. Existing November 4
  boundary/FU edits and running frozen-source campaigns were preserved.

## Implementation

Optimized reduced factors are QR/LQ factored, then their rank-sized central
matrix is split with Quimb SVD (`cutoff=0`, `renorm=0`). Its unnormalized
singular values become the active diagonal gauge. Only the endpoint core
tensors change; exterior gauges and spectator core tensors stay fixed.
Exterior square-root weights are removed with a support pseudoinverse.
Missing initial gauges are identities. The gate-stream wrapper also avoids
global SU initialization. Legacy `regauge_opts` iteration/tolerance controls
are deprecated and ignored with a warning; other options are rejected.

Native Symmray extraction preserves physical/reduced charge spaces when
decompositions omit empty sectors. Tensor operations remain in their backend.
The [BP API](../docs/api/bp.md#reduced-loop-cluster-compression) describes
reconstruction, amplitude handling and the local-spectrum interpretation.

## Upstream audit

Installed: Quimb `1.15.1.dev101+g1b64353d9`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3`, Symmray
`0.4.1.dev25+gdc377f033`, Torch `2.14.1+cu126`.
Inspected installed `Tensor.split`, `array_split`, `gauge_simple_insert`,
`gauge_simple_remove` and native BlockVector operations.
**Adopt:** public Quimb QR/LQ/SVD and diagonal-index multiplication; no
installed library edits or new compatibility shim.

Consulted the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray Abelian-array documentation URL was inaccessible; installed
source and native reconstruction tests supplied the capability check.

## Validation and limits

New regressions cover NumPy, Torch CPU and CUDA PEPS/PEPO reconstruction,
nonzero network exponents, zero exterior weights, unnormalized spectra,
unchanged spectators/exterior gauges, input mutation policies, sequential
gate replay and blocked global SU calls. Native U1 fermionic reconstruction
is checked independently for physical and SU-core inputs.

Final selected gate: **77 passed** (12.84 s), including the complete reduced
update suite, new local-gauge suite, native local extraction and the two
SymPEPS loop-cluster wrapper checks. Full `ruff check src tests` and
`git diff --check` passed. Existing NumPy deprecations and the intentional
legacy-option deprecations remain visible.
After tightening inplace writeback to preserve spectator array identities,
all **26 local-gauge/native-extraction tests passed** again (3.87 s), with
Ruff and whitespace checks passing again.

The existing `test_native_reduced_loop_compression_uses_graded_svd_adapter`
fails before gauge extraction: the native solver produces shape `(1,1,2,2)`
where `(1,2,2,2)` is required. Loading the unchanged `ee22198` implementation
reproduced that exact failure. It is separate from this change and was not
fixed here. The full suite was not run.

The exact six-site 3×2 cluster selector and a production χ6 stability study
remain separate work; public cluster controls still select a graph radius.
No simulation was launched, stopped, or restarted. Nothing was staged or pushed.
