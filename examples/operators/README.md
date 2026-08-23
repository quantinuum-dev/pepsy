# Operator API smoke examples

These examples demonstrate the four canonical construction families from the
[operator inventory](../../docs/development/modules/operators.md). Run them
from the Pepsy repository root after activating the shared Python 3.12
environment:

```bash
source ~/envs/py312/bin/activate
python examples/operators/higher_order_mpo.py
python examples/operators/fixed_channel_pepo.py
python examples/operators/dense_cluster_pepo.py
python examples/operators/ordered_pepo_product.py
```

They are deliberately small and use public `pepsy.operators` imports. They
check representation construction and explicit materialization; they are not
benchmark scripts.
