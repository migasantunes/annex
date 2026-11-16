"""annex: a vector search engine (approximate nearest neighbour) written in C++20.

The index classes are implemented in C++ (the ``annex._core`` extension) and
take and return NumPy arrays::

    import numpy as np
    import annex

    index = annex.FlatIndex(dim=128, metric="l2")
    index.add(np.random.rand(1000, 128).astype(np.float32))
    labels, distances = index.search(queries, k=10)

Every metric is a distance: smaller means more similar.
"""

# The version comes from the extension, which gets it from project() in the
# root CMakeLists.txt, so the importable package and the compiled code cannot
# disagree. A test checks it also matches the version in pyproject.toml.
from annex._core import FlatIndex, __version__, active_kernel_name

__all__ = ["FlatIndex", "__version__", "active_kernel_name"]
