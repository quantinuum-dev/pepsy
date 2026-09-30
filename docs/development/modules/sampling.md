# Sampling implementation

Public sampling APIs live in `pepsy.sampling`. Choose the engine from the
state representation; the direct PEPS and BP proposal algorithms are distinct.

| Module | Responsibility |
| --- | --- |
| `mps.py` | `MpsSampler`, dense/native conditional sampling, symmetric prefix environments |
| `vector.py` | `VecSampler`, computational/Pauli bases, dense categorical sampling |
| `peps.py` | `PepsSampler`, exact conditionals and conditioned boundary-MPS proposals |
| `bp.py` | `PepsBpSampler`, BP proposal probabilities and original-state amplitudes |
| `results.py` | Sample records, batch conversions, fermionic configuration encodings |
| `_common.py` | Site-map validation, array conversion, fermionic code ordering |
| `tree.py` | Tree conditional sampling and tree-specific records |
| `stabilizer.py` | Physical-state sampling through stabilizer frame projectors |
| `samplers.py` | Compatibility imports and old serialized class paths |

The public namespace resolves each class directly from its owner. Importing
`VecSampler` does not load the MPS or PEPS engines. Historical classes imported
from `sampling.samplers` are the same objects, and old pickle globals resolve
through that facade. Tests that replace private module globals patch the
owning implementation module.

Keep proposal probabilities, physical amplitudes, and importance weights
distinct. Shared result records do not own sampler environments. No engine
may reuse a sampled-prefix environment as an unconditioned future boundary.

See the [sampling API guide](../../api/sampling/samplers.md) for supported
backends, result shapes, seeds, and approximation controls.
