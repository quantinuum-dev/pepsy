# API migration guide

The named import aliases listed below remain available during the 0.x
compatibility window. This does not imply that every optimizer mode, default,
or supported Python version is unchanged. New code should use owning
namespaces and the canonical imports below.

## Upgrading from 0.4.1 to 0.5.0

Version 0.5.0 includes incompatible changes under the
[pre-1.0 stability policy](../stability.md). Review these changes before
upgrading an existing environment or reproducing numerical results.

### Environment

Use Python 3.12 or newer. Python 3.10 and 3.11 are no longer supported,
including for optional extras. The direct core requirements are NumPy
`>=1.26`, Quimb `>=1.15`, Cotengra `>=0.8.0`, Autoray `>=0.9`, and
tqdm `>=4.65`. Optional floors also increased, including Torch `>=2.4` and
Symmray `>=0.4.0`.

For NetKet VMC, use the `vmc-netket` extra (or `vmc` for both backends): it
requires NetKet `>=3.22`, JAX `>=0.7,<0.11.1`, Flax `>=0.10.6`, and Optax
`>=0.2.2`. The JAX upper bound addresses a demonstrated upstream import
failure. Regenerate environment locks against the selected extras instead of
reusing a Python 3.10/3.11 lock. See [installation](../installation.md) and
the [dependency audit](notes/dependency_minimums_2026_09.md) for all optional
floors and the limits of minimum-version testing.

### Replay modes and diagnostics

| Existing usage or assumption | Migration |
| --- | --- |
| `MpsOptimizer(state, gates, chi=64)` implicitly selects DMRG | It now selects `direct`; pass `mode="dmrg"` or a named DMRG schedule for variational replay. |
| MPS `mode="su"` | Removed; use `pepsy.operators.gate_simple` or the dedicated PEPS simple-update API. |
| MPS `routing="perm"`, `perm-dmrg2`, `dmrg2-perm`, or similar combinations | Removed; use standalone `mode="perm"` for lazy swap-and-split SVD, or an explicit solver with a static layout. These are different algorithms. |
| `mode="mix"` with custom FIT block sizes or initialization | Mix fixes one-site FIT and `guess-direct`; use ordinary DMRG to configure another policy. |
| Runtime finite scans or MPI timing/rank summaries are implicit | Request `finite_check=True` or `collect_diagnostics=True` explicitly where needed. |
| A fixed seed reproduces earlier finite-rank tree results | Tree SRC environments, path routing, and FIT traversal changed; compare observables and truncation error, and record the selected algorithm and traversal. |

For example, explicitly retain the variational replay choice:

```python
from pepsy.optimizers import MpsOptimizer

opt = MpsOptimizer(state, gates, chi=64, mode="dmrg")
opt.run(n_iter=8, fit_rtol="auto")
```

This selects DMRG; it does not promise the exact numerical trajectory of an
older default schedule. Current MPS FIT uses an eight-sweep default budget,
larger-block warm-up, and one-site refinement; `dmrg3` includes a two-site
transition. Tree optimizers use `fit_traversal="auto"` for path-versus-branch
selection. Set convergence, initialization, sweep, and traversal controls
explicitly when comparing algorithms. See [MPS](../api/optimizers/mps.md),
[MPO](../api/optimizers/mpo.md), [tree](../api/optimizers/tree.md), and
[TreePEPS](../api/optimizers/tree_peps.md) for their distinct schedules.

MPS permutation replay keeps the physical reordering. Read `opt.logical_order`
or `opt.qubits`, and call `opt.restore_qubit_order()` before treating `opt.p`
as an MPS in conventional logical order. A persistent layout cannot be
combined with `mode="perm"`.

### Hamiltonian construction and tree state contracts

The four `ham_tn.to_*` operator builders now default to `compress="term"`,
which compresses after each added term. To request workload-selected
construction, pass `compress="auto"`; to force shared state diagrams, use
`compress="automaton"`. Pass `compress=False` to disable numerical
compression. Explicit `max_bond=None` or `False` also disables it; an omitted
cap inherits the builder setting. Specify the construction strategy, bond cap,
and cutoff when reproducing older operator approximations. See
[Hamiltonian builders](../api/operators/hamiltonians.md).

`TreePeps` now permits four virtual bonds per site (rank five including the
physical leg), including the central backbone of `span-middle`. Update code
that assumed a hard three-virtual-bond limit. Non-branching geometries must
request `topology="path"`. Tree compression defaults to live-rank scheduling;
`order="depth"` requests the farthest-first schedule. Native Symmray SVD may
retain a degenerate boundary beyond a requested cap, as documented in the
tree API; a bond cap alone is not a proof of exact output rank.

Dense tree SRC/SDC use directed environments on the original layered target.
Native symmetry tensors reject these routes; use a supported direct, zipup,
or FIT route as appropriate. Odd-parity native TreeFIT projections remain
unsupported and raise explicitly. These backend restrictions should not be
worked around by silently converting fermionic tensors to dense arrays.

For `TreeSampler`, `backend="auto"` retains the dense compatibility path for
Symmray input; request `backend="symmray"` or `"native"` for block-sparse
sampling. See [tree sampling](../api/sampling/tree.md).

## Deprecated aliases

| Deprecated import | Canonical import |
| --- | --- |
| `pepsy.tensors.backend_cupy` | `pepsy.backends.backend_cupy` |
| `pepsy.tensors.backend_jax` | `pepsy.backends.backend_jax` |
| `pepsy.tensors.backend_numpy` | `pepsy.backends.backend_numpy` |
| `pepsy.tensors.backend_torch` | `pepsy.backends.backend_torch` |
| `pepsy.tensors.build_backend` | `pepsy.backends.build_backend` |
| `pepsy.tensors.get_default_array_backend` | `pepsy.backends.get_default_array_backend` |
| `pepsy.tensors.get_default_grad_backend` | `pepsy.backends.get_default_grad_backend` |
| `pepsy.tensors.get_torch_linalg_config` | `pepsy.backends.get_torch_linalg_config` |
| `pepsy.tensors.register_jax_linalg` | `pepsy.backends.register_jax_linalg` |
| `pepsy.tensors.register_torch_linalg` | `pepsy.backends.register_torch_linalg` |
| `pepsy.tensors.reset_default_backends` | `pepsy.backends.reset_default_backends` |
| `pepsy.tensors.reset_linalg_registrations` | `pepsy.backends.reset_linalg_registrations` |
| `pepsy.tensors.set_default_array_backend` | `pepsy.backends.set_default_array_backend` |
| `pepsy.tensors.set_default_grad_backend` | `pepsy.backends.set_default_grad_backend` |
| `pepsy.tensors.TorchLinalgConfig` | `pepsy.backends.TorchLinalgConfig` |
| `pepsy.tensors.build_contraction` | `pepsy.tensors.build_optimizer` |
| `pepsy.tensors.SpinfulFermionHubbard` | `pepsy.tensors.SpinfulFermion` |
| `pepsy.tensors.hrps_to_mps` | `pepsy.tensors.hrs_to_mps` |
| `pepsy.tensors.hrps_to_peps` | `pepsy.tensors.hrs_to_peps` |
| `pepsy.tensors.hrps_to_ttn` | `pepsy.tensors.hrs_to_ttn` |
| `pepsy.boundary.normalize` | `pepsy.boundary.peps_normalize` |
| `pepsy.boundary.infidelity` | `pepsy.boundary.peps_infidelity` |
| `pepsy.optimizers.QMeraParametricEnergyOptimizer` | `pepsy.optimizers.QMeraEnergyOptimizer` |
| `pepsy.optimizers.MpsStabOptimizer` | `pepsy.optimizers.StabilizerMpsSimulator` |
| `pepsy.optimizers.TreeStabOptimizer` | `pepsy.optimizers.StabilizerTreeSimulator` |
| `pepsy.sampling.MpsStabSampler` | `pepsy.sampling.StabilizerMpsSampler` |
| `pepsy.optimizers.stabilizer_tn.StabilizerMps` | `pepsy.optimizers.stabilizer_tn.StabilizerMpsSimulator` |
| `pepsy.experimental.mera` | `pepsy.experimental.qmera` |
| `pepsy.optimizers.mera` | `pepsy.optimizers.qmera` |
| `ham_tn.build_mpo(...)` | `ham_tn.to_mpo(...)` |
| `ham_tn.build_pepo(...)` | `ham_tn.to_pepo(...)` |

These aliases emit `DeprecationWarning` when resolved. Applications can make
the transition visible in CI with:

```bash
python -W error::DeprecationWarning -m pytest
```

## Removal policy

No alias is scheduled for removal from the current 0.x line. Before a planned
breaking release, maintainers should review warning usage, publish release
notes with the table above, and remove only aliases that have completed the
deprecation window. The root-level compatibility facade is governed by
[`api-manifest.txt`](api-manifest.txt) and remains unchanged until that
review.

## Tree operator conversion

The canonical model-facing conversion methods are `ham_tn.to_mpo(...)`,
`ham_tn.to_pepo(...)`, `ham_tn.to_tree_mpo(...)`, and
`ham_tn.to_tree_pepo(...)`. The first two return Quimb chain/lattice operator
networks; the latter two factor directly over a supplied `TreePlan` or
`TreePepsPlan` without creating a chain MPO. `to_treempo` and
`to_treepepsmpo` are short compatibility aliases for the native tree methods.

`ham_tn.build_mpo(...)` and `ham_tn.build_pepo(...)` remain available as
deprecated wrappers and emit `DeprecationWarning`. The lower-level native
routes `TreePlan.build_tree_operator(...)` and
`Fermion.build_tree_operator(...)` remain available when a model object is
already in hand.
