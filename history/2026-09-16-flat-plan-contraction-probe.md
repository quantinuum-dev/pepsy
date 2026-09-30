# 2026-09-16 — Flat-plan contraction probe

- Probed `pepsy.contract_flat` on networks built by Gaugy's
  `FlatPEPSPlan.from_color_ansatz`.
- Confirmed exact, DMRG/FIT, direct MPS, CTMRG, and HOTRG paths execute on
  flat HRPS networks.
- Direct MPS schedules (`bottom-up`, `top-down`, four-sided, and middle-out)
  matched exact contraction on a small 3x3 probe at `chi=16`.
- CTMRG modes `projector`, `projector2d`, and `l2bp` execute. Small dense
  probes validated `projector2d` and `l2bp`; the default projector showed
  finite-chi error. On the 4x4, 52x16 notebook-sized network, CTMRG was slower
  and substantially more chi-sensitive than direct MPS.

Environment probe: Pepsy 0.4.1, Quimb 1.15.1.dev55, Cotengra
0.8.3.dev7, Autoray 0.11.1.dev3, Symmray 0.3.2.dev8.
