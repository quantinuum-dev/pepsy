# Optional and advanced extensions

Pepsy keeps optional or heavier domains behind explicit lazy entry points:

```python
from pepsy.extensions.vmc import TorchVMCDriver
from pepsy.extensions.mera import QMeraBuilder
from pepsy.extensions.bp import gauge_all
from pepsy.extensions.stabilizer import STNState
```

The compatibility namespaces delegate to the established implementations in
`pepsy.vmc`, `pepsy.optimizers.mera`, `pepsy.bp`, and
`pepsy.optimizers.stabilizer_tn`. Existing imports remain supported; the
extension namespace simply makes the dependency and maturity boundary visible
to new callers.

```{eval-rst}
.. automodule:: pepsy.extensions
   :members:
   :undoc-members:
```
