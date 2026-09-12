# MPS layout replay and scheduling

## 2026-09-11 implementation note

The MPS layout finder now has two opt-in additions:

- `objective="replay"` uses static compression candidates as a bounded
  shortlist, schedules each barrier-separated ordinary gate segment for every
  candidate, then replays each candidate on a copied optimizer and records the
  transient `MPS.max_bond()` and `bond_sizes()` after every event. The selected
  plan carries `replay_event_order` and `scheduled_stream`, and
  `run(layout=plan)` executes that joint layout/order result.
- `compile_schedule(..., strategy="mountain")`, exposed through
  `MpsOptimizer.gate_stream_schedule` and
  `MpsOptimizer.current_gate_stream_schedule`, builds a stable dependency DAG.
  Events sharing a logical site retain their input order; only disjoint
  ordinary gate/sub-MPO events can move. Direct caps are fixed barriers whose
  mapped physical positions are removed before later events are emitted.
- `replay_schedule="measure-early"` is an opt-in safe relaxation of the
  barrier policy: measurements and resets may move left across immediately
  preceding ordinary events on disjoint supports. Shared-site gates,
  feed-forward, and cap events remain fixed dependencies.

The replay objective is deliberately state-aware and explicit. It does not
change the existing `locality` or `compression` scores, and it never mutates
the source optimizer. It requires a fixed-layout MPS mode and supports direct
caps, while rejecting conditional caps, trajectory streams, and preinstalled
persistent layouts. Measurements, resets, and conditional events are retained
as fixed barriers by default; the
measure-early policy only crosses immediately preceding disjoint ordinary
events. A state reorder is
allowed only through the caller's explicit `replay_allow_lossy_reorder=True`
choice, following the normal persistent-layout contract.

The compiled schedule emits physical positions plus a position-to-logical
`site_order`, matching `MpsOptimizer.set_gate_schedule`. Direct cap events are
emitted as lifetime barriers, and the schedule metadata exposes the final
shortened `final_site_order`. It does not move measure/reset/feed-forward
events; those remain in the stateful replay path.

## Upstream compatibility audit

The active Python 3.12 environment was probed on this date:

- Quimb `1.15.1.dev51+g2e99c793e`; `MatrixProductState.max_bond()`,
  `bond_sizes()`, and the existing MPS gate/compression APIs are available.
- Autoray `0.11.1.dev3+g1b476b305`.
- Cotengra `0.8.3.dev7+g1d7fd333f`.
- Symmray `0.3.2.dev8+g6c6dd34b5`.

The upstream changelog audit classified this change as **adopt** for the
existing Quimb MPS public surfaces: no private upstream implementation is
vendored or patched. The scheduler itself is **prototype** infrastructure in
Pepsy and is independent of Quimb's optional path optimizers. Quimb's
`gate_with_auto_swap_` contract retains physical positions after a non-swap
back operation, while Pepsy's `set_gate_schedule` already owns the required
fixed-layout stream boundary.

Focused regression coverage is in `tests/test_mps_layout_upgrade.py`.
