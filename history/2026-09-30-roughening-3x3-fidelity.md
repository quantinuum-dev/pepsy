# 2026-09-30 — Roughening fidelity, BP norm, and 1,200-sample checks

- Scope: verify the roughening SU exponent flags and compare 3×3 D=2/4
  evolution with exact evolution, C=0/4/6 BP norms, boundary χ=2D local Z
  and imbalance, and 1,200 Z-basis samples at t=4/6.
- Baselines: Pepsy develop / 69b8294; examples main / 40cadd7.
- Commit status: new evidence notes are uncommitted; no staging or publishing.
  Existing implementation edits and unrelated work were preserved.

The runner already passes `renorm=False, strip_exponent=True`. No additional
implementation change was necessary. CPU complex128 diagnostics at dt=.25,
Δθ=0 with no periodic gauge refresh completed through t=6 without overflow.
Full-contraction norms agree with dense amplitudes within 6e−16.

Fidelity against exact full-Hamiltonian evolution is only 0.0159–0.0477 at
the requested snapshots; comparison with exact Trotter evolution confirms
that timestep error alone does not explain this. C=4 improves BP norms;
C=6 is not monotonic. Boundary χ=2D has measurable local-Z errors.

Strict sampling failed only for D=4,t=4 at χ=χ′=8. The diagnostic explicitly
enabled existing absolute-eigenvalue proposal repair for that snapshot and
retained importance weights. All four proposals were subsequently checked
over all 512 configurations for normalization, full support, and agreement
with batch probabilities; all sample amplitudes match exact PEPS amplitudes.
Aggregate sampled observables agree with the PEPS but not exact evolution.

See [detailed numerical evidence and artifact paths](../docs/development/notes/2026-09-30-roughening-3x3-fidelity.md).
Scripts, PDF/PNG figures, JSON/CSV, and samples are under
`/tmp/roughening_3x3_fidelity_bp_samples_20260930/`.
No GPU job was launched or disturbed; no numerical suite was rerun in this
follow-up, and package/runner defaults were not changed.
