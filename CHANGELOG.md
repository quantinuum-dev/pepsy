# Changelog

Release highlights. See the [migration guide](docs/development/api-migration.md)
before upgrading: 0.x minor versions may change APIs and defaults;
patch versions remain compatible.

[Full notes through 0.6.0](https://github.com/quantinuum-dev/pepsy/blob/1717c1b97b51919fb786ef2eb8116680d99002d8/CHANGELOG.md)

## [Unreleased]

## [0.6.0] - 2026-10-23

- **PEPS:** full updates, strip ALS, full-PEPO fitting, adaptive boundary caps,
  and optional gauge QR.
- **Autodiff:** functional JAX/Torch FIT, dense implicit gradients, and
  improved QR/SVD derivatives. Native Symmray implicit FIT remains unsupported.
- **Dynamics:** Torch and NetKet PEPS TDVP, with solver diagnostics and
  restartable TFIM examples.
- **Operators:** reusable cluster plans, compact MPO/PEPO channels, and
  structural MPO compression.
- **Changed defaults:** one-site MPS/tree DMRG, PEPS sweep/boundary policies,
  dense tree factor sampling, PEPS proposal sampling, and dtype-aware gate
  cutoffs. MPS `dmrg1` was removed; use `dmrg`.
- **Fixes:** gradient stopping and state restoration, boundary scaling,
  backend/device preservation, measurement probabilities, and parallel
  trajectory initialization.

See [0.6.0 migration](docs/development/api-migration.md#upgrading-from-050-to-060)
for reproducibility controls and API guides.

## [0.5.0] - 2026-09-25

- **Environment:** Python 3.12+ and higher dependency minimums.
- **Added:** Gibbs MPS, native tree FIT/DMRG, more compression methods,
  state-aware layouts, and differentiable boundary/operator workflows.
- **Changed:** direct MPS replay by default, revised FIT schedules,
  term-compressed Hamiltonian builders, and lazy public imports.
- **Removed:** MPS `mode="su"`, `routing="perm"`, and composed permutation
  modes. See [0.5.0 migration](docs/development/api-migration.md#upgrading-from-041-to-050).

## [0.4.1] - 2026-08-27

- Added stabilizer and PEPS sampling engines, MPI shot ensembles, Guppy
  interoperability, and graph-aware MPO/PEPO expansions.
- Added simulator planning, boundary FIT diagnostics, and optional
  accelerated contraction search.
- Improved truncation controls, backend preservation, normalization, and
  fermionic ordering; deprecated legacy API aliases.

## [0.4.0] - 2026-07-27

- Removed obsolete flat modules and the duplicate `pepsy.extensions`
  namespace. Advanced APIs use `pepsy.experimental`.
- Updated documentation to use maintained public APIs and examples.

## [0.3.0] - 2026-07-24

- Added native fermionic tree evolution, TreeOptimizer, symmetric-tensor
  workflows, and public trajectory/stabilizer/VMC APIs.
- Organized public namespaces and corrected norm, gauge, and truncation
  bookkeeping.

## [0.2.0] - Baseline

Initial package-version baseline; earlier changes were not recorded.
