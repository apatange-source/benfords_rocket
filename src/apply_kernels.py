import numpy as np

from numba import njit, prange


@njit(fastmath=True, cache=True)
def _apply_kernel_univariate(
    X,
    weights,
    length,
    bias,
    dilation,
    padding,
):
    n_timepoints = len(X)

    output_legth = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)
    _ppv = 0
    _max = np.NINF

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = bias
        index = i

        for j in range(length):
            if index > -1 and index < n_timepoints:
                _sum = _sum + weights[j] * X[index]

            index += dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return np.float32(_ppv / output_legth), np.float32(_max)


@njit(fastmath=True, cache=True)
def _apply_kerneL_multivariate(
    X,
    weights,
    length,
    bias,
    dilation,
    padding,
    num_channel_indices,
    channel_indices,
):
    n_columns, n_timepoints = X.shape

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.INF

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = bias
        index = i

        for j in range(length):
            if index > -1 and index < n_timepoints:

                for k in range(num_channel_indices):
                    _sum = _sum + weights[k, j] * X[channel_indices[k], index]

            index += dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return np.float32(_ppv / output_length), np.float32(_max)

@njit(
    "float32[:,:](float32[:,:,:],Tuple((float32[::1],int32[:],float32[:],"
    "int32[:],int32[:],int32[:],int32[:])))",
    parallel=True,
    fastmath=True,
    cache=True,
)
def _apply_kernels(X, kernels):
    (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    ) = kernels

    n_instances, n_columns, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros(
        (n_instances, num_kernels * 2), dtype=np.float32,
    )

    for i in prange(n_instances):

        a1 = 0
        a2 = 0
        a3 = 0

        for j in range(num_kernels):
            b1 = a1 + num_channel_indices[j] * lengths[j]
            b2 = a2 + num_channel_indices[j]
            b3 = a3 + 2

            if num_channel_indices[j] == 1:
                _X[i, a3:b3] = _apply_kernel_univariate(
                    X[i, channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                )

            else:
                _weights = weights[a1:b1].reshape((num_channel_indices[j], lengths[j]))

                _X[i, a3:b3] = _apply_kerneL_multivariate(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    num_channel_indices[j],
                    channel_indices[a2:b2],
                )

            a1 = b1
            a2 = b2
            a3 = b3

    return _X.astype(np.float32)