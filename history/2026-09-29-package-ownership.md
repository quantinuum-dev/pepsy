# 2026-09-29 — Independent Pepsy and downstream Gaugy ownership

- Scope: user clarified Pepsy is the tensor-network/graph foundation and
  Gaugy owns its Pauli expansion and domain optimization code.
- Branch / baseline: `develop`, `b343ced`; existing numerical implementation
  edits and unrelated optimizer work remain in the working tree.
- Commit status: this Pepsy test/documentation cleanup is uncommitted.

## Findings and changes

Pepsy source and dependency metadata contain no Gaugy import/dependency.
The current Gaugy cluster adapters import public Pepsy APIs. Graph planning,
PEPO construction, routing, compression and reporting already live in Pepsy;
the Pauli engines and connected-log objectives live in Gaugy. No numerical
implementation move was needed and no public signature changed.

Moved the two-case Gaugy graph handoff/partial-vector gradient regression
from Pepsy's graph test module to Gaugy's binding tests. Added a package
boundary regression: a fresh isolated interpreter blocks all Gaugy imports
while running public graph/square PEPO construction, traces, compression,
and reports. The test also checks base/optional dependency declarations.
Documented one-way ownership in each package's existing architecture guides.

## Validation

Shared Python 3.12 environment, CPU, single-threaded BLAS/OpenMP:

- Pepsy public API, package layout, and graph autodiff selection: **65 passed**
  in 12.11 seconds, with two existing deprecation warnings.
- Gaugy binding/materialization selection: **19 passed** in 8.13 seconds,
  including the moved integration tests.
- Pepsy `ruff check src tests`, Gaugy changed-test Ruff, and diff whitespace
  checks passed. Local documentation links checked before completion.

This is a test ownership and documentation change, not a new numerical
validation claim for the entire package. No full suite or dependency change.
The previously rejected Gaugy push remains pending explicit approval; this
task did not authorize retrying publication.

See [package ownership](../docs/development/package_layout.md#pepsy-and-downstream-packages).
