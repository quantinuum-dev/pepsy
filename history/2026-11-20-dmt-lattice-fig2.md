# 2026-11-20 — Fig. 2 lattice-DMT transport benchmark

- Scope: user requested Fig. 2 of arXiv:2610.40776 after the 6×6 quench tests.
- Branch / baseline: `develop`, `eaedf789`; preceding working-tree edits
  preserved. No core library API changed in this follow-up.
- Publication: uncommitted, unstaged; no commit/push requested or performed.

Implemented `examples/dmt_lattice_fig2.py` using public `PauliDMT` transfer
updates in traceless operator mode. Match the length-21 uncoupled-chain
Hamiltonian, 1.4/0.9045 fields, column-major interleaving, energy packet and
initial-normalized MSD. Compare chain triples to all physical straight/L-shaped
triples at chi128. Real float64 Torch tensors use explicit CPU SciPy GESDD/
pivoted QR; no CuPy or positive-factor fitting.

Keep the important difference: our integrator is protected-SVD Strang TEBD,
while the paper uses TDVP. Independent small dense/spectral and sparse bit-spin
oracles validate normalization, conjugation and timestep order. Added actual
lossy traceless nonconsecutive-gate preservation checks. An independent N=21
sparse exponential/typicality reference uses one random-phase state; it is
unbiased but has unestimated sampling uncertainty and is not labeled exact.

Added a provenance-labeled Figure 2 vector-coordinate fixture/extractor.
The source archive has TeX/PDFs, no raw numerical series; its Fig. 2 timestep
is unspecified. Plotted coordinate precision and typicality errors are separate.
The duplicate published black references agree; colored curves start at the
analytic MSD=0.13234094, while the plotted black reference starts at 0.17456.

Completed width-one controls at dt0.1/0.2 to t20: MSD31.66191/31.62494.
Width-two lattice DMT dt0.2 reaches t20, MSD31.63558, without preservation
warnings; matched-dt width-one difference is only 0.01064 at the endpoint.
The independent N21 typicality calculation finishes t20 at MSD31.63836.
At common t7, width-four chain/lattice MSDs are 13.73422/14.32772 versus
14.27268 for the independent reference. Chain overlap is only 0.93214,
versus 0.99872 for lattice protection. Chain4 hits its declared 1500-second
budget at t7.4; this is a partial-time result, not a full-width reproduction.
Lattice4 stops at its 1800-second budget at t9, MSD19.64849, overlap0.998696;
no preservation failures are recorded. Keep both partial endpoints visible.
The final width-two chain control finishes t20 at MSD31.62926 and overlap
0.988817; compare lattice2 overlap0.998596 and the complete MSD curves rather
than only their close endpoints. All six DMT runs and the independent
reference have finished; four full-time DMT runs and two budget-limited runs
are explicitly labeled. No benchmark processes are intentionally left running.
See the [evidence record](../docs/development/notes/2026-11-20-dmt-lattice-fig2.md)
and linked aggregate/plots for final measured endpoints, uncertainty and budgets.

Validation: 49 focused tests passed; Ruff and whitespace passed. Full suite
not rerun because only examples/tests/docs were added; prior known BP namespace
and installed-version failures remain recorded in the earlier handoff.
This is a selected-width physical comparison, not a full reproduction of
all width2–20 published curves or a thermodynamic diffusion calculation.
