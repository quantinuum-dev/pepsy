# Automaton plus numerical delinearisation, 2026-09-30

Classification: **adopt** the existing public API composition. No package
algorithm, default, or compatibility shim changed. The single-exponential
`gaugy_examples/pauli_gaugy/cluster_1d.ipynb` now calls
`delinearize_mpo(automaton_mpo, ...)` alongside the frontier version. Both
use `rtol=1e-12`, `preserve_zeros=True`, `max_sweeps=4`, and comparison time
0.06. Uncapped automaton preparation remains exact Pauli/state reduction;
the subsequent QR stage selects dependencies numerically at that time.

## Measurements

Six sites, NN + NNN Ising, J1=1, J2=0.5, h=1, complex128 NumPy.
Entries list maximum stored bonds across orders p=1 through 5.

| Boundary | Automaton | Automaton + QR | Frontier + QR |
| --- | --- | --- | --- |
| OBC | 1, 25, 81, 161, 209 | 1, 24, 54, 51, 50 | 1, 24, 54, 51, 50 |
| PBC | 1, 125, 281, 281, 329 | 1, 64, 64, 64, 64 | 1, 64, 64, 64, 64 |

All five OBC bonds coincide between the QR variants. PBC's fourth stored
bond differs: automaton + QR gives 17, 16, 20 at p=2, 3, 4, versus
18, 17, 18 for frontier + QR. Thus neither variant uniformly improves the
other. These are representation-dependent numerical reductions, not
minimum-rank guarantees.

Worst relative Frobenius error against independent explicit Cp:
2.42e-14 OBC, 5.91e-14 PBC. Worst change introduced by automaton QR relative
to its input: 9.69e-15 OBC, 3.03e-14 PBC (rounded upward).
Every operator also passes the independent cut-rank tail consistency check.

## Validation and environment

- Complete single-exponential notebook executions passed separately for OBC
  and PBC. Only affected OBC outputs were refreshed in the source notebook;
  all unrelated cells/outputs and metadata were preserved by snapshot check.
- `tests/test_mpo_delinearize.py`: 27 passed.
- Ruff on Pepsy `src tests` and the notebook, notebook schema, and whitespace
  checks passed. Both updated numerical figures were visually inspected.
- No full package suite or joint-notebook rerun. No staging or publication.

Installed versions: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev7+g83fb22865`, NumPy `2.5.2`, SciPy `1.17.1`.
Inspected installed signatures and local sources of `prepare_cluster_channels`
and `delinearize_mpo`. The latter accepts dense NumPy open-chain Quimb MPOs,
uses SciPy GELSY QR solves, and rejects native/autodiff arrays explicitly.
Periodic interactions use open-chain MPO storage here.

Consulted the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray abelian-arrays documentation returned an internal error; used
the official repository and installed dense-array guards for scope.
No upstream implementation was adopted or modified; no dependency upgrade.
