"""Data reduction by simple sampling"""
from .base import Reducer

import numpy as np
import warnings

class DownSampler(Reducer):
    type = "Downsampling"
    
    def reduce(self, x: np.ndarray, n: int) -> np.ndarray:
        if n >= x.shape[0]:
            warnings.warn("n is greater than the number of points, no reduction will be performed")
            return x
        if n <= 2:
            raise ValueError('n must be greater than 2')
        indices = np.linspace(0, x.shape[0] - 1, n, dtype=int)
        return x[indices]