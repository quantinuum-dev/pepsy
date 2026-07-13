# Leaning on quimb (and cotengra / autoray)

pepsy already builds on the `quimb` ecosystem — `quimb`, `cotengra`, and
`autoray` are hard dependencies (`pyproject.toml`). The BP workstream should
**wrap quimb where possible** rather than reimplement message passing, gauging,
and contraction from scratch.

## What quimb already provides

- `quimb.tensor.TensorNetwork` — the network object pepsy already uses (pepsy's
  `build_bra_ket` returns quimb-tagged networks; `BdyMPS` consumes them).
- **Belief propagation** — `quimb.tensor` ships BP variants (dense and "loopy"
  1-norm / 2-norm message passing) plus BP-based **gauging** of tensor networks
  (the Tindall–Fishman gauge). These cover the BP fixed point, the gauge, and
  basic environment extraction.
- **Contraction** — `cotengra` path optimization, which pepsy already wraps via
  `build_optimizer(...)` / `contraction_opt="auto-hq"`. Reuse this for the small
  loop-excitation sub-networks in the loop expansion.
- **Backends** — `autoray` dispatch lets the same code run on numpy / torch /
  jax / cupy, and is exactly how `symmray` arrays plug in (see `symmray.md`).

## "pepsy concept → quimb API" map (to fill in during M1)

| pepsy need | quimb surface | notes |
| --- | --- | --- |
| Build double-layer network | `build_bra_ket` → `qtn.TensorNetwork` | already done |
| BP fixed point | `quimb.tensor` BP message-passing classes | audit exact names/signatures in M1 |
| BP gauge | quimb BP gauging helpers (`gauge_all`-family) | doubles as simple-update gauge / initializer |
| BP environments / RDMs | quimb message → local env contraction | may need a thin pepsy adapter |
| Sub-network contraction | `cotengra` via pepsy `build_optimizer` | do **not** add seed kwargs (tests assert absence) |

> The exact class/function names are intentionally left to the M1 audit so this
> doc records *verified* API, not guesses. Update the table with the concrete
> symbols and the pinned minimum quimb version once confirmed.

## Integration guidelines

- **Adapters, not forks:** accept pepsy-tagged networks, return pepsy result
  dataclasses (`BPResult`), so users never have to context-switch APIs.
- **Version guarding:** quimb is already required; if BP needs a newer quimb,
  guard the import and document the minimum version. Keep behavior graceful if
  an older quimb lacks a feature.
- **Reuse pepsy's optimizer builders** for contraction; don't introduce parallel
  contraction config.
- For MPS gate streams, treat Quimb's `info`/`cur_orthog` metadata as part of
  the algorithm state. Reuse a known canonical range for local expectations
  and one-site norms; do not replace it with a full-network norm contraction.
  When building an uncapped diagnostic target from `p.copy()`, use a separate
  info dictionary so the target cannot corrupt the live optimizer's center.
- A persistent site layout is a bookkeeping permutation over an MPS. It can be
  relabelled without SVD only for `p.max_bond() == 1`; otherwise make the
  one-time reorder explicit and caller-controlled. Keep logical readout as an
  axis/sample remap rather than restoring the physical MPS order every step.
- Exact MPS replay is a separate contracted-TensorNetwork path. It does not
  consume canonical metadata, and returning to an MPS backend requires
  rebuilding and canonicalizing an MPS first.
- **Decision to record (M1):** how much BP logic lives in pepsy vs. is delegated
  to quimb. Write the outcome into `history/` and into `PLAN.md` §7.

## References

- quimb docs: https://quimb.readthedocs.io/ (tensor + belief propagation).
- cotengra: https://cotengra.readthedocs.io/.
- autoray: https://github.com/jcmgray/autoray.
