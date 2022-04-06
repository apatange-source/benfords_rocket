import numpy as np 

from numba import njit, prange

@njit("Tuple((float64[:], int32[:], float64[:], int32[:], int32[:]))(int64, int64)")
def generate_kernels(input_length, num_kernels, random_state=42,):
    np.random.seed(random_state)    
    candidate_lengths = np.array((7, 9, 11), dtype= np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0

    for i in range(num_kernels):

        _length = lengths[i]
        _weight = np.random.normal(0, 1, _lenth)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length-1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0

        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

