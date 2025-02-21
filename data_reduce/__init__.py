"""Data reduction algorithms"""
__version__ = "0.1.0"

from .base import Reducer, Reducer2D
from .vw import VWReducer
from .sampler import DownSampler
from .plotting import show_reduction
