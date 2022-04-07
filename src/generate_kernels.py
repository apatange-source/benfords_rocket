import numpy as np

from numba import njit, prange

from benfords_rocket.src.benfords_generator import BenfordRandom

import random


@njit(
    "Tuple((float32[:], int32[:], float32[:], int32[:], int32[:], int32[:], "
    "int32[:]))(int32, int32, int32, optional(int32))",
    cache=True,
)
def _generate_kernels(
    n_timepoints,
    num_kernels,
    n_columns,
    random_state,
):

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels).astype(np.int32)

    num_channel_indices = np.zeros(num_kernels, dtype=np.int32)

    for i in range(num_kernels):
        limit = min(n_columns, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))
        ),
        dtype=np.float32,
    )

    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.float32)
    paddings = np.zeros(num_kernels, dtype=np.float32)

    a1 = 0
    a2 = 0  # for channel_indices

    for i in range(num_kernels):
        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        _benfords_list = [
            BenfordRandom().random()
            for _ in range((_num_channel_indices * _length) // 2)
        ]
        _benfords_list.extend(
            [
                -1 * BenfordRandom().random()
                for _ in range((_num_channel_indices * _length) // 2)
            ]
        )
        random.shuffle(_benfords_list)
        _weights = np.array(_benfords_list,).astype(
            np.float32,
        )

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 0  # For weights per channel

        for _ in range(_num_channel_indices):
            b3 = a3 + _length
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        channel_indices[a2:b2] = np.random.choice(
            np.arange(0, n_columns),
            _num_channel_indices,
            replace=False,
        )

        biases[i] = np.random.uniform(-1, 1)

        dilation = np.int32(
            (
                2
                ** np.random.uniform(
                    0,
                    np.log2((n_timepoints - 1) / (_length - 1)),
                )
            )
        )

        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0

        paddings[i] = padding

        a1 = b1
        a2 = b2

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )
