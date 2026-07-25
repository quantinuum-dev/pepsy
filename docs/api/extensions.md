# Optional and advanced extensions

Pepsy keeps optional or heavier domains behind explicit namespaces. New code
should use the canonical implementation modules or the discovery-friendly
`pepsy.experimental` namespace:

```python
from pepsy.experimental import bp, symmetry, stabilizer, vmc
from pepsy.vmc import TorchVMCDriver
```

The compatibility extension paths remain available:

```python
from pepsy.extensions.vmc import TorchVMCDriver
from pepsy.extensions.mera import QMeraBuilder
from pepsy.extensions.bp import gauge_all
from pepsy.extensions.stabilizer import STNState
```

The extension namespaces delegate to the established implementations in
`pepsy.vmc`, `pepsy.optimizers.mera`, `pepsy.bp`, and
`pepsy.optimizers.stabilizer_tn`. Existing imports remain supported; the
extension namespace simply makes the dependency and maturity boundary visible
to new callers.


> API details are maintained as handwritten Markdown in this page.
