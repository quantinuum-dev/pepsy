# 2026-09-25: retained-register spin 1D qMERA

The main unmoded 1D `QMeraBuilder` now builds a unitary-completion preparation
circuit. `system_size=N` is the preferred constructor spelling for a 1D
chain of N spatial sites; `shape=N` remains accepted, while 2D uses a tuple
shape. At each fine-to-coarse scale, it groups adjacent child registers into
blocks of configured length, merges an odd binary tail into a ternary block,
and retains up to `bond_qubits` child wires. A disentangler window straddles
each block boundary. Gate execution runs coarse-to-fine: isometry-block
unitaries, then boundary disentanglers. With periodic boundaries the ordered
site list starts at site 1 and ends at site 0, matching the ml4mb 1D
reference's block convention. The L=7, bond-width-2 schedule has a ternary
bottom block and two RG scales; the generated pair order and a nonzero dense
statevector agree with the ml4mb reference to numerical precision.
For unmoded spin 1D blocks, `structure="brickwall"` retains that ml4mb
pair order, while `structure="ladder"` visits consecutive block wires and
closes the last wire to the first. `circuit_depth` repeats the complete
sequence. Two-wire ladders have one edge. Isometry and boundary
disentangler structures are independent; 2D and explicit-mode fermion
layouts reject ladder rather than silently changing their gate placement.
Ladder is a Pepsy extension to the ml4mb reference, which uses brickwall
pair rounds.

The spin default is the global-X-preserving four-rotation pair family ZZ, YY,
XI, IX. Every angle remains an independent parameter in the existing Pepsy
parameter dictionary. The ml4mb reference has five pair templates and **all
five** preserve global-X Z2. Pepsy now exposes these through
`pair_ansatz=` (with `ansatz=` retained as an alias) and adds a spin-only
`spin_symmetry="z2"|"unrestricted"` selector that chooses the default
compatible template or checks a named one. Descriptive preferred names
identify the symmetry and Pauli rotations: `z2_zz_yy_rx`,
`z2_rx_zz_yy_xx_rx`, `z2_zz_rx`, `z2_yz_zy`, `z2_all_paulis`, and
`all_paulis`. The six old names remain accepted aliases; the old listing
function retains its previous output. The explicit `all_paulis` 15-angle
template uses every nonidentity Pauli word. The
ml4mb `minimal` initialization policy is retained: its four separately
trainable angles start at a shared value; other presets start with independent
values. Custom `QMeraPairSpec` values must declare
`symmetry="unrestricted"` before using a generator such as ZI that breaks
global-X parity.

The public `QMeraPairSpec.rotation_sequence` reports each chronological
logical rotation and pair-local wires, including repeated angles in
`extended`. Its order is RX(0), RX(1), RZZ, RYY, RXX, RX(0), RX(1). Pepsy
composes these into one differentiable pair tensor per placement. The logical
YZ/ZY rotations use direct Pauli matrices here; ml4mb lowers them into RZZ
gates with fixed RX basis changes.

The input is named separately by `initial_state="zero"|"plus"`; the default is
`|0...0>` and the plus option produces the `|+>` input used in the ml4mb
reference. These are independent choices: commutation of
the circuit with global X does not put a `|0...0>` input in a definite
symmetry sector. Explicit-mode fermions keep the native Symmray path, and 2D
keeps its existing schedule.

Upstream compatibility audit for this task used Quimb 1.15.1.dev66,
Autoray 0.11.1.dev3, Cotengra 0.8.3.dev7, Symmray 0.4.1.dev7,
NumPy 2.5.2, Torch 2.6.0, and JAX 0.10.2 in the shared Python 3.12
environment. We inspected the installed `TensorNetwork.gate_inds` and
`cotengra.array_contract_expression` signatures and the official
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray). The
[Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable during the audit; the installed implementation and official
repository were used instead. Classification: **adopt** the existing public
Quimb, Autoray, and Cotengra APIs; no compatibility shim or dependency
modification was needed for the spin path. The native Symmray path is
**deferred** from this change, so its representation is preserved.

Validation is recorded in the
[session handoff](../../../history/2026-09-25-qmera-1d-retained-registers.md).
The focused regressions compare the 1D gate stream to a dense Pauli circuit, direct and
compiled local energies, and Torch gradients, alongside the existing 2D and
native fermion tests.
