"""Base class for data reduction"""

from abc import ABC, abstractmethod
import numpy as np

class Reducer(ABC):
    type = 'None'
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def reduce(self, x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        pass
    
class Reducer2D(Reducer):
    type = '2D'
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def reduce(self, x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        if x.shape[1] != 2:
            raise ValueError('The input must be a 2-column array')
        pass