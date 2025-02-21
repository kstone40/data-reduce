"""Visvalingam-Wyatt line simplification algorithm."""
# Based on 'Line generalisation by repeated elimination of points' by Visvalingam and Whyatt, 1993. #
from .base import Reducer2D

import numpy as np
import warnings

class VWReducer(Reducer2D):
    """
    Visvalingam-Whyatt line simplification algorithm
    
    Methods:
        area_by_det(x1, x2): Compute the area of a triangle given its vertices
        point_importance(index, x1, x2): Calculate the importance of a point based on its effective area
        all_importances(x1, x2): Calculate the importance of all points in the dataset
        reduce(x, n, importances): Reduce the dataset to n points using the VVW algorithm
    """    
    
    type = 'Visvalingam-Whyatt'
    
    def area_by_det(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the area of a triangle given its vertices"""
        if (len(x1) != 3) or (len(x2) != 3):
            raise ValueError('The input must be a list of 3 elements')
        Z = np.vstack((x1, x2, np.ones((len(x1),)))).T
        return np.abs(np.linalg.det(Z))/2
    
    def point_importance(self, index: int, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate the importance of a point based on its effective area"""
        if (index == 0) or (index == len(x1)-1):
            # The first and last points are always kept
            return np.inf
        if (index < 0) or (index >= len(x1)):
            raise ValueError('Index out of bounds')
        _x1 = x1[index-1:index+2]
        _x2 = x2[index-1:index+2]
        area = self.area_by_det(_x1, _x2)
        return area
    
    def all_importances(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Calculate the importance of all points in the dataset"""
        if len(x1) != len(x2):
            raise ValueError('x1 and x2 must have the same length')
        importances = []
        for i in range(len(x1)):
            importances.append(self.point_importance(i, x1, x2))
        return np.array(importances)


    def reduce(self, x: np.ndarray, n: int, importances: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Reduce the dataset to n points using the VVW algorithm
        
        Args:
            x (np.ndarray): The 2D dataset to reduce
            n (int): The number of points to keep
            importances (np.ndarray, optional): The pre-computed importances
                of the points. By default is ``None`` which will compute the
                importances from scratch.
            
        Returns:
            np.ndarray: The reduced x1 dataset
            np.ndarray: The reduced x2 dataset
            
        """
        if x.shape[1] != 2:
            raise ValueError('The input must be a 2-column array')
        
        x1, x2 = x[:, 0], x[:, 1]        
        
        if n >= len(x1):
            warnings.warn("n is greater than the number of points, no reduction will be performed")
            return x1, x2
        if n <= 2:
            raise ValueError('n must be greater than 2')
        
        importances = self.all_importances(x1, x2)

        cull_idx = []
        m_reduce = len(importances) - n

        x1_culled = x1.copy()
        x2_culled = x2.copy()

        while len(cull_idx) < m_reduce:
            # Find the lowest importance point
            min_idx = np.argmin(importances)

            # Remove min_idx from all arrays: importance, x1, x2
            cull_idx.append(min_idx)
            importances = np.delete(importances, min_idx)
            x1_culled = np.delete(x1_culled, min_idx)
            x2_culled = np.delete(x2_culled, min_idx)
            
            # Re-compute the importance of the neighboring points
            if (min_idx > 0):
                importances[min_idx-1] = self.point_importance(min_idx-1, x1_culled, x2_culled)
            # Note: with min_idx removed, the index of min_idx+1 is now min_idx
            importances[min_idx] = self.point_importance(min_idx, x1_culled, x2_culled)
        
        X_culled = np.vstack((x1_culled, x2_culled)).T
        
        return X_culled
