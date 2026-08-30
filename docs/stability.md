# API stability policy

PePsY is organized into two user-facing layers.

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

`pepsy.interop` and the high-level `pepsy.optimizers` namespace are stable
orchestration surfaces. Their advanced subdomains—such as QMERA, tree and
stabilizer tensor networks, noisy trajectories, and Symmray workflows—remain
explicit domain APIs and may require optional dependencies.

## Top-level compatibility facade

The top-level `pepsy` namespace is a frozen compatibility facade. Existing
root-level names remain available through lazy aliases, but new public names
should be added to their responsibility-based namespace instead of expanding
`pepsy.__init__`. Advanced functionality belongs in its explicit domain module
or under `pepsy.experimental`.

Any proposed root-level addition requires an API-stability review and a
regression test. This policy prevents the root namespace from becoming a
second, eager import surface while preserving existing user code.

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

## Removed compatibility modules

Old flat module paths such as `pepsy.core` and `pepsy.optimize_mps` were
removed in the 0.4 package-layout cleanup. Use the responsibility-based paths,
for example `pepsy.tensors` and `pepsy.optimizers.mps`.
