# Integrated cluster-MPO automaton and delinearisation API

Date: 2026-09-30. Working-tree implementation, not committed or published.
The user clarified that the requested integration belongs in Pepsy itself;
the preceding notebook composition alone did not fulfill that request.

## Implemented

- `exp_mpo_cluster` and `exp_mpo_cluster_product` accept explicit
  `preparation="frontier"` or `"automaton"`. They reuse term parsing,
  coordinate mapping, parameter/vector binding and fixed cluster planning,
  then evaluate the channel plan instead of the ordinary assembled MPO.
- `delinearize=True` and `delinearize_opts` integrate NumPy QR dependency
  removal after channel evaluation. Options are rtol, preserve_zeros and
  max_sweeps. Fixed uncapped residuals, exact graph targets, and Quimb return
  are required; incompatible controls reject rather than being ignored.
- `ClusterChannelPlan.exp` and `trace_exp` support the same reduction.
  Repeated evaluations keep fresh numerical QR decisions outside unchanged
  symbolic maps and array/autodiff kernels. PEPO/native and autodiff arrays
  reject explicit QR use, without conversion or detaching.
- Channel-plan evaluation reports describe final stored dimensions and
  include a separate delinearisation report plus original channel dimensions.
  One-shot channel calls return these dictionaries rather than the legacy
  facade's analytical dataclass; the ordinary default path is unchanged.
- Fixed a static report-classification bug exposed by an existing automaton
  regression: comparing effective bases with the original topology confused
  exact structural reductions with reference QR projection. The report now
  records whether the numerical QR branch actually ran. No numerical result
  or test tolerance was changed to resolve this failure.

Owning guides: [one-shot API](../../api/operators/mpo_cluster.md#integrated-automaton-plus-qr-construction),
[channel plans](../../api/operators/cluster_channels.md#optional-numerical-delinearisation-on-evaluation),
and [numerical delinearisation](../../api/operators/mpo_delinearize.md).

## Validation

The final affected selection passed **297 tests**, with two existing public
alias deprecation warnings, in 111.96 seconds. It included:

`test_mpo_cluster_channels_api.py`, `test_mpo_delinearize.py`,
`test_mpo_cluster.py`, `test_mpo_cluster_recursive.py`,
`test_cluster_fixed_factorization.py`, `test_cluster_channels.py`,
`test_cluster_channel_frontier.py`, `test_cluster_channel_automaton.py`,
`test_cluster_channel_structure.py`, `test_cluster_channel_pepo.py`,
`test_public_api.py`, and `test_package_layout.py`.

New regressions cover independent full exponentials and partial cluster sums,
disjoint/crossing/nested supports, ordered factors with runtime coefficients,
coordinate mapping, parameter rebinding including zero, traces, source-plan
preservation, final report dimensions, and backend/option guards. Selected
construction tests explicitly forbid SVD and expanded source assembly.
Existing channel suites separately exercise NumPy, native symmetry and
Torch/JAX differentiation without numerical delinearisation.

The initial broader run had 246 passes and one failure in
`test_no_decomposition_no_collection_expansion_and_zero_parameter_gradients`;
its method-label assertion exposed the report bug described above. The final
297-test result follows that fix. Ruff (`src tests`), edited-document links
and `git diff --check` passed. No full repository suite was run.

The integrated facade was also probed on the notebook's six-site Ising model
(J1=1, J2=0.5, h=1, t=0.06, p=5) for OBC/PBC. Stored bonds agree with the
earlier standalone composition: OBC `(5,33,209,129,13)` becomes
`(4,16,50,16,4)`; PBC `(5,41,329,233,21)` becomes `(4,16,64,20,4)`.
Independent explicit-Cp relative Frobenius errors are 2.42e-14 (OBC) and
5.03e-14 (PBC), rounded upward, using the existing examples reference helper.
These are stored-dimension measurements, not optimality or memory guarantees.

## Upstream audit and limits

Classification: **adopt** existing Pepsy composition at the public API layer;
no new upstream algorithm, shim, dependency change, or installed-library edit.
Reused the [same-task upstream audit](2026-09-30-automaton-delinearisation-example.md#validation-and-environment)
and rechecked installed versions/signatures. They are unchanged: Quimb
`1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev7+g83fb22865`, NumPy `2.5.2`,
SciPy `1.17.1`. The earlier Symmray documentation fetch was unavailable;
the official repository and installed guards were inspected instead.

Numerical QR still selects ranks at the evaluated parameters, has no global
error bound or minimum-rank guarantee, and supports dense NumPy MPOs only.
No notebook source/output edits were made during this package-integration turn.
