import numpy as np 

from numba import njit, prange

@njit("Tuple((float64[:], int32[:], float64[:], int32[:], int32[:]))(int64, int64)")
def 
