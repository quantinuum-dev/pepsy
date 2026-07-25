# API stability policy

Pepsy is organized into two user-facing layers.

## Stable core

The stable core is intended for normal application code:

- `pepsy.boundary`
- `pepsy.backends`
- `pepsy.fitting`
- `pepsy.operators`
- `pepsy.sampling`
- `pepsy.solvers`
- `pepsy.tensors`

Stable APIs follow semantic versioning, include regression tests, and receive
deprecation warnings before removal whenever practical.

## Advanced and experimental domains

BP, MERA, stabilizer tensor networks, tree tensor networks, Symmray-specific
workflows, and VMC integrations are advanced domains. They are available from
their explicit modules and through `pepsy.experimental`:

```python
from pepsy.experimental import bp, symmetry, vmc
from pepsy.vmc.torch import TorchVMCDriver
```

These domains may evolve faster and can have additional dependency or backend
requirements. Their public entry points are documented, but implementation
details are not compatibility guarantees.

## Deprecations

Old flat module paths such as `pepsy.core` and `pepsy.optimize_mps` remain as
warning-emitting compatibility facades for the current deprecation window.
New code should use the responsibility-based paths, for example
`pepsy.tensors.core` and `pepsy.optimizers.mps`.
