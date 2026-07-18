"""Tree tensor-network optimizer package.

Public entry point :class:`TreeOptimizer` replays a bundled gate stream on a
rooted tree tensor network (Seitz et al., Quantum 7, 964, 2023;
arXiv:2206.01000).  :class:`TreeLayoutFinder` / :class:`TreePlan` choose and
describe the tree structure.
"""

from .layout import TreeLayoutFinder, TreePlan
from .optimizer import TreeOptimizer

__all__ = [
    "TreeOptimizer",
    "TreeLayoutFinder",
    "TreePlan",
]
