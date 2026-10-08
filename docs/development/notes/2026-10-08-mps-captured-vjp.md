# MPS fixed-shape Torch VJP capture

The [session record](../../../history/2026-10-08-mps-captured-vjp-workflow.md)
documents the FX compatibility changes and focused validation. The
[Gaugy workflow record](../../../../gaugy/history/2026-10-08-mps-eager-overlap-workflow.md)
contains end-to-end compiler checks and measured CPU costs.

Capturing the joint forward and first-order derivative preserves Pepsy's custom
SVD/QR policy. Capturing only the forward array graph does not. Adaptive QR
capture uses fixed-shape masks and runtime assertions; singular-chart
regularization remains an explicit extension rather than a differentiability
guarantee. No Inductor or GPU performance claim follows from CPU aot_eager
success. The measured small direct CPU example was faster with eager Torch.
