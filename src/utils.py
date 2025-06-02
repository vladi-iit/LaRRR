from math import sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import torch


def topk(vec: ArrayLike, k: int):
    assert np.ndim(vec) == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"
    sort_perm = np.flip(np.argsort(vec))  # descending order
    indices = sort_perm[:k]
    values = vec[indices]
    return values, indices


def fuzzy_parse_complex(vec: ArrayLike, tol: float = 10.0):
    assert issubclass(
        vec.dtype.type, np.complexfloating
    ), "The input element should be complex"
    rcond = tol * np.finfo(vec.dtype).eps
    pdist_real_part = pdist(vec.real[:, None])
    # Set the same element whenever pdist is smaller than eps*tol
    condensed_idxs = np.argwhere(pdist_real_part < rcond)[:, 0]
    fuzzy_real = vec.real.copy()
    if condensed_idxs.shape[0] >= 1:
        for idx in condensed_idxs:
            i, j = row_col_from_condensed_index(vec.real.shape[0], idx)
            avg = 0.5 * (fuzzy_real[i] + fuzzy_real[j])
            fuzzy_real[i] = avg
            fuzzy_real[j] = avg
    fuzzy_imag = vec.imag.copy()
    fuzzy_imag[np.abs(fuzzy_imag) < rcond] = 0.0
    return fuzzy_real + 1j * fuzzy_imag


def row_col_from_condensed_index(d, index):
    # Credits to: https://stackoverflow.com/a/14839010
    b = 1 - (2 * d)
    i = (-b - sqrt(b**2 - 8 * index)) // 2
    j = index + i * (b + i + 2) // 2 + 1
    return (int(i), int(j))

def report(_test_sample, predictions = None, dt = None, start = 0., label_= " ", quantiles = None, idx_plotted=None, figsize = (10,2)):
    if idx_plotted is None:
        idx_plotted = np.arange(_test_sample.shape[1])

    fig, axes = plt.subplots(len(idx_plotted), 1, figsize=(figsize[0], figsize[1] * len(idx_plotted)))
    fig.suptitle(label_)
    auxT = _test_sample.shape[0]

    if dt is not None:
        time = (start+np.arange(0, auxT)) * dt
    else:
        time = start+np.arange(0, auxT)

    
    if len(idx_plotted) == 1:
        axes = [axes]  # Ensure axes is iterable

    if predictions is not None:
        predictions = predictions.reshape(_test_sample.shape)
    
    for i, (ax, idx) in enumerate(zip(axes, idx_plotted)):
        ax.plot(time, _test_sample[0:auxT, idx], 'r', label='true state' if i == 0 else "")
        if predictions is not None:
            ax.plot(time, predictions[0:auxT, idx], 'g--', label='estimate' if i == 0 else "")
        if quantiles is not None:
            ax.fill_between(np.arange(0, auxT), quantiles[0:auxT, idx, 0], quantiles[0:auxT, idx, 1], color='green', alpha=0.25)
        if i == 0 and predictions is not None:  # Add legend only to the first subplot to avoid repetition
            ax.legend(loc="upper left")
    if predictions is not None:
        print(f"{label_} RMSE: {np.linalg.norm(predictions-_test_sample,'fro')/np.sqrt(_test_sample.shape[0])}")
    return fig, axes
    
    
def tonp(x):
    return x.detach().cpu().numpy()

def frnp(x, device=None):
    return torch.Tensor(x).to(device)

def sqrtmh(A: torch.Tensor):
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

def center_matrix(A):
    """
    Center the matrix A by subtracting the mean of each row and column.
    """
    r_mean = np.mean(A,axis=0, keepdims=True)
    c_mean = np.mean(A, axis=1, keepdims=True)
    A = A - r_mean
    A = A - c_mean
    A = A + r_mean.mean()
    return A