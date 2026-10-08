# Full-update gate conversion, normalization, and scheduler corrections

The [current-commit review](../../../history/2026-10-08-full-update-current-commit-review.md)
reproduced two correctness failures and a scheduler performance issue at
`ad8ed05`. The following corrections remain working-tree changes.

## Implemented behavior

- Convert incoming full-update gates to the state's backend, dtype, and device
  before either pair reduction or exact strip-target construction. This uses
  Pepsy's existing converter; PEPS arrays are not transferred to the host.
- Honor target normalization after single-site absorption independently of
  final normalization. An active exact strip target receives the same checked
  scalar normalization, without an extra norm contraction. Its own requested
  target normalization still runs when the block closes.
- Track dependency counts and ready pairs incrementally. A single-site node
  becomes logically available once its predecessors are released, but is
  emitted only when a chosen pair needs it. Traverse single-site ancestors
  only for the chosen pair. Preserve the old strip/bond priority and gate IDs.

The dependency graph still takes quadratic storage/work for heavily
overlapping queues, and selecting among many simultaneously ready pairs can
still take quadratic work. This removes the cubic predecessor-copy behavior;
it does not claim a linear-time compiler or a globally optimal schedule.

## Measured scheduler evidence

CPU, shared NumPy gate objects, alternating XX/ZX rotations of angle 0.1 on
one bond, compilation only. The circuit forces the original execution order.
These are single-run measurements on the same environment, not production
speed guarantees.

| Gates | Reviewed implementation | Corrected implementation |
| --- | ---: | ---: |
| 100 | 0.021 s | 0.018 s |
| 200 | 0.086 s | 0.066 s |
| 400 | 0.378 s | 0.255 s |
| 800 | 1.877 s | 0.995 s |
| 1600 | 10.100 s | 3.965 s |

One old orientation pass copied 84,575 / 671,650 / 5,353,300 predecessor
indices at 100 / 200 / 400 gates. The corrected pass copies zero indices in
this case, because all pair predecessors have been emitted before selection.
An independent differential probe of 200 seeded, mixed 60-gate queues
returned exactly the same schedules as `ad8ed05`, including local gates,
mixed Pauli axes, and opaque overlapping matrices.

## Upstream audit

The installed-version/signature audit from the preceding review applies to
this continuation in the unchanged `py312` environment: Quimb
1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a, Cotengra
0.8.3.dev8+g8954240f2, Symmray 0.4.1.dev15+g0374aaa3c, Torch 2.6.0+cu124,
CuPy 14.1.1. Public ALS accepts the explicit overlap-network arguments;
gate conversion and scalar normalization reuse existing Pepsy capabilities.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The Symmray
abelian-array documentation endpoint returned an error. Classification:
**defer** upstream changes; these corrections address Pepsy control flow,
with no compatibility shim, dependency installation, or native-symmetry
behavior change. Full-update still explicitly excludes native symmetry arrays.

Focused numerical results and remaining validation limits are recorded in the
[fix handoff](../../../history/2026-10-08-full-update-review-corrections.md).
