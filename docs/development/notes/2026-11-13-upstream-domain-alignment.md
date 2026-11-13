# Symmray/Quimb development alignment — 2026-11-13

Selected Pepsy native fermionic and ordinary operator paths work with the
current upstream development revisions; the audit also fixes empty BP
correction traces. This is scoped CPU validation, not a full-suite or
all-layout compatibility certificate.

## Reviewed revisions

GitHub `main` was queried directly on 2026-11-13. Both installed revisions
match the heads returned by GitHub:

| Dependency | Installed version | Source revision |
| --- | --- | --- |
| Symmray | `0.5.1.dev2+g20035beb7` | [`20035beb79d97e64c9dbf2e16f4acb98b2c10cf2`](https://github.com/jcmgray/symmray/commit/20035beb79d97e64c9dbf2e16f4acb98b2c10cf2) |
| Quimb | `1.16.1.dev1+g3371ff40c` | [`3371ff40c25d52e6bf60cbf3766918c782dad9f4`](https://github.com/jcmgray/quimb/commit/3371ff40c25d52e6bf60cbf3766918c782dad9f4) |
| Autoray | `0.11.1.dev17+g7a474cab3` | `7a474cab367dd0be2acbb1b80e9279da5c8bd528` |
| Cotengra | `0.8.4.dev8+g646239731` | `646239731c1062e2a74becddfa4ed4ce576daab5` |

NumPy is `2.5.2`, Torch `2.14.1+cu126`, JAX `0.11.2`, and NetKet `3.22.4`.
No dependency was installed or modified during this audit.

The requested Symmray Read the Docs index and array pages were unavailable
through the browser. Reviewed their official source instead: the
[changelog](https://github.com/jcmgray/symmray/blob/20035beb79d97e64c9dbf2e16f4acb98b2c10cf2/docs/changelog.md),
[array guide](https://github.com/jcmgray/symmray/blob/20035beb79d97e64c9dbf2e16f4acb98b2c10cf2/docs/abelian_arrays.md),
[conjugation guide](https://github.com/jcmgray/symmray/blob/20035beb79d97e64c9dbf2e16f4acb98b2c10cf2/docs/fermionic_arrays.md),
[dummy-mode guide](https://github.com/jcmgray/symmray/blob/20035beb79d97e64c9dbf2e16f4acb98b2c10cf2/docs/dummy_modes.md),
and [layout guide](https://github.com/jcmgray/symmray/blob/20035beb79d97e64c9dbf2e16f4acb98b2c10cf2/docs/block_layout.md).
Also reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray), and
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html).

## Compatibility decisions

| Classification | Finding and Pepsy handling |
| --- | --- |
| **Adopt, already dispatched** | Quimb's positive fermionic D2BP messages, native `safe_inverse`, seeded SRC, and complex64 `svd:eig` pass installed capability probes. Existing compatibility adapters defer to those implementations. |
| **Adopt, already dispatched** | Modern 2D `method`/`route` signatures and explicit `max_bond` work through Pepsy's existing boundary adapters. Normalized scalars and numerator/norm conventions remain explicit. |
| **Adopt, already implemented** | Sparse native gates fill zero sectors before fusion/exponentiation; explicit charges protect constructors from changes to inferred-charge defaults. See the [gate/FIT repair](2026-11-13-native-fermion-gates-fit.md). |
| **Adopt, unchanged policy** | Pepsy's cutoff policies remain explicit. New upstream low-level `rel` defaults do not justify silently changing a caller's requested truncation policy. |
| **Compatibility shim** | An empty native BP rho correction has zero trace. Avoid dense conversion when `num_blocks == 0`; retain the native correction and existing configuration sum. |
| **Prototype** | New packed CPU arrays pass eight small MPS replay probes, described below. No automatic conversion or default change. |
| **Defer** | Broader packed VMC/PEPS/PEPO/tree support, GPU/performance certification, and removal of legacy capability adapters. Passing current-head tests does not establish those claims. |

Pepsy's Hubbard construction uses its native Symmray/model adapters, not
Quimb's newly changed default spin-orbital ordering. No Hubbard coefficient,
basis, gate schedule, canonicalization algorithm, or FIT schedule was changed
in this audit. Native TreeFIT's existing odd-parity restrictions remain.

## Reproduced and repaired BP defect

The two earlier failures in
`test_fermionic_open_rho_sweep_supports_native_symmetries` (`U1`, `U1U1`)
also reproduce against the unchanged implementation. A forbidden-charge
correction can have no stored blocks and an empty fused index map; calling
Symmray `to_dense()` on it raises an empty-concatenation error.

[`_rho_trace`](../../../src/pepsy/bp/series.py) now returns zero for precisely
that case. It does not densify the network, replace graded observables with
`trace(rho @ gate)`, discard nonzero corrections, or change normalization.
The regression now also covers `Z2Z2`, alongside `U1`, `U1U1`, and `Z2`.
All four symmetry cases pass. See the [open-rho API](../../api/bp.md).

## Measured coverage

All runs activate the shared development environment, use the local source,
and restrict numerical libraries to one CPU thread. CUDA is disabled and
JAX uses CPU. Every command below uses `python -m pytest -q -o addopts=''`
with `-m 'not slow'` unless stated otherwise.

| Selection | Result |
| --- | --- |
| `test_vmc_api`, `test_vmc_local_energy`, `test_vmc_torch_zero_log`, `test_vmc_transition_plan`, `test_vmc_importance`, `test_vmc_tdvp` | 95 passed |
| `test_vmc_torch_compile`, `test_vmc_tdvp_compile` | 14 passed; includes flat-Z2 amplitudes, vmap/export, AOT execution, and derivatives |
| `test_tree_native`, `test_tree_mpo`, `test_tree_pepo`, `test_tree_operator_representation`, `test_tree_symmray_factor`, `test_tree_mps_parity`, `test_tree_submpo`, `test_pauli_mpo`, `test_tree_zipup` | 257 passed, 25 skipped |
| `test_mps_fermions`, `test_native_fermion_pepo_2x3` | 166 passed, 31 deselected; additionally excludes `nonchain_edge_identity_sandwich_4x2` |
| `test_quimb_compat`, `test_bp_symmray`, `test_fermionic_boundary`, `test_peps_boundary_amplitudes`, `test_peps_pepo_steps` | 125 passed after the fix; the initial run had 122 passed and the two reproduced empty-block failures |
| `test_tree_peps`, `test_tree_peps_optimizer`, `test_native_cluster_mpo`, `test_netket_flat_z2`, `test_netket_boundary_compression`, `test_bp_symmray`, `test_quimb_compat` | 266 passed; includes native JAX/JIT and complex directional-derivative checks |
| Expanded open-rho symmetry regression | 4 passed |
| Default smoke profile (`python -m pytest -q`) | 112 passed |
| Ruff, local Markdown links, `git diff --check` | Passed |

The six domain selections cover **854 distinct passing cases** after removing
their BP/compatibility overlap. The 266-case selection collected the previous
three-case rho parameterization; the final 125-case selection includes the
added fourth symmetry. The focused four-case rerun and smoke selection overlap
the domain runs and are reported separately.

Native TTN/TreeMPO checks include fermionic ordering, charge sectors, operator
scales, native readout, and replay/compression. Pauli tree coverage uses
TreeMPO's compact Pauli construction; there is no separate `PauliTree` class.
The separate lattice-tree `TreePeps`/`TreePepo` constructors and their tests
are ordinary dense arrays; these results do not establish a native fermionic
implementation of that separate representation.
Native Symmray full-PEPO fitting remains explicitly unvalidated in the
[PEPS API](../../api/optimizers/peps.md); native PEPO simple-update/readout
coverage here does not establish full-update support.

Some tests intentionally issue approximation, deprecated-alias, and QR rank
warnings. NumPy 2.5 also warns about shape assignment inside upstream Quimb
optimization packing; this is upstream behavior, not a new Pepsy coercion.
Hardware skips and excluded slow tests are not validated coverage.

## Packed-layout probes

Four-site U1 and U1U1 native MPS seeds and hopping gates were explicitly
converted with `to_packed()` on private copies. Nonlocal `(0, 3)` and reversed
`(1, 0)` gates were replayed in `mpo` (direct), `dmrg`, `dmrg2`, and `dmrg3`
at cap 16 and zero cutoff. All eight runs retained packed arrays and matched
the sparse reference with fidelity between `0.9999999999999999` and
`1.0000000000000002`.

Only the comparison copy was converted back with `to_blocksparse()`; direct
mixed-layout contractions are unsupported upstream. This is not a runtime or
truncated-state accuracy benchmark. Packed storage is NumPy/CPU; sparse Torch
and flat traced arrays remain the relevant gradient paths.

Temporary probe source/results: `/tmp/pepsy_upstream_alignment_probe.py` and
`/tmp/pepsy_upstream_alignment_probe.json`. Test logs use
`/tmp/pepsy-upstream-audit-*.log`. These temporary artifacts may disappear;
the revisions, decisions, and measured results above are the durable record.
