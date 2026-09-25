# `pepsy.boundary.states`

`BdyMPS` is the reusable boundary-environment store. Its `mps_b` map uses
`Y{i}_l`/`Y{i}_r` for vertical cuts and `X{i}_l`/`X{i}_r` for horizontal cuts.
By default construction is eager for backward compatibility. Pass
`lazy=True` to defer each boundary chain until its key is accessed; requesting
`Y2_l`, for example, materializes only the missing left chain through step 2.
`ensure_boundaries(direction="y", upto_step=0)` is available when a caller
wants explicit materialization without touching a key.

`tn_flat` with `flat=True` means one already-flattened effective lattice layer.
`tn_double` with `flat=False` is the multi-layer path for tagged BRA--KET or
BRA--PEPO--KET networks. The `flat` flag does not flatten a stack of layers.

> API details are maintained as handwritten Markdown in this page.
