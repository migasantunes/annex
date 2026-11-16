# Type stubs for the compiled annex._core module (python/src/bindings.cpp).
# Keep in sync by hand; docstrings live in the C++ source.

from typing import Literal

import numpy as np
import numpy.typing as npt

__version__: str

def active_kernel_name() -> str: ...

class FlatIndex:
    def __init__(self, dim: int, metric: Literal["l2", "ip", "cosine"] = "l2") -> None: ...
    def add(
        self,
        vectors: npt.ArrayLike,
        labels: npt.ArrayLike | None = None,
    ) -> None: ...
    def search(
        self, queries: npt.ArrayLike, k: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.float32]]: ...
    def __len__(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def metric(self) -> str: ...
