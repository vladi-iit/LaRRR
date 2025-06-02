import logging

import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Literal
from scipy.linalg import toeplitz
from scipy.linalg import matmul_toeplitz

from LaRRR.src.utils import topk

logger = logging.getLogger("LaRRR")


def add_diagonal_(M: ArrayLike, alpha: float):
    """
    Add alpha to the diagonal of a matrix M in-place.

    Parameters
    ----------
    M : ArrayLike
        The matrix to modify.
    alpha : float
        The value to add to the diagonal of M.
    """
    np.fill_diagonal(M, M.diagonal() + alpha)


def rank_reveal(
    values: np.ndarray,
    rank: int,  # Desired rank
    rcond: Union[float, None] = None,  # Threshold for the singular values
    ignore_warnings: bool = True,
):
    if rcond is None:
        rcond = 10.0 * values.shape[0] * np.finfo(values.dtype).eps

    top_values, top_idxs = topk(values, rank)

    if all(top_values > rcond):
        return top_idxs
    else:
        valid = top_values > rcond
        # In the case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
        first_invalid = np.argmax(np.logical_not(valid))
        _first_discarded_val = np.max(np.abs(values[first_invalid:]))

        if not ignore_warnings:
            logger.warning(
                f"Warning: Discarted {rank - values.shape[0]} dimensions of the {rank} requested due to numerical instability. Consider decreasing the rank. The largest discarded value is: {_first_discarded_val:.3e}."
            )
        return top_idxs[valid]

def weighted_norm(A: ArrayLike, M: Union[ArrayLike, None] = None):
    r"""Weighted norm of the columns of A.

    Args:
        A (ndarray): 1D or 2D array. If 2D, the columns are treated as vectors.
        M (ndarray or LinearOperator, optional): Weigthing matrix. the norm of the vector :math:`a` is given by
        :math:`\langle a, Ma\rangle`. Defaults to None, corresponding to the Identity matrix. Warning: no checks are
        performed on M being a PSD operator.

    Returns:
        (ndarray or float): If ``A.ndim == 2`` returns 1D array of floats corresponding to the norms of
        the columns of A. Else return a float.
    """
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    if M is None:
        norm = np.linalg.norm(A, axis=0)
    else:
        _A = np.dot(M, A)
        _A_T = np.dot(M.T, A)
        norm = np.real(np.sum(0.5 * (np.conj(A) * _A + np.conj(A) * _A_T), axis=0))
    rcond = 10.0 * A.shape[0] * np.finfo(A.dtype).eps
    norm = np.where(norm < rcond, 0.0, norm)
    return np.sqrt(norm)

def toeplitz_generator(exp_decay: float, npts: int, context_length: int, dt: float = 1., 
                        symmetric: bool = False
                        )->ArrayLike:
    if context_length==0:
        toep = np.diag(npts*np.ones(npts-1)/(npts-1),1)
    else:     
        tau_ = (npts* dt* np.exp(-np.arange(0, context_length) * dt * exp_decay)/(npts - np.arange(0, context_length)))
        tau_[0] *= 0.5
        tau_[-1] *= 0.5
        tau = np.concatenate((tau_, np.zeros(npts - context_length)))
        toep = toeplitz(tau, np.zeros(npts)).T
    if symmetric:
        toep += toep.conj().T
        toep /=2
    return toep


def toeplitz_integrator(matrix: ArrayLike, exp_decay: float, npts: int, context_length: int, dt: float = 1., 
                        symmetric: bool = False
                        )->ArrayLike:
    tau_ = (npts* dt* np.exp(-np.arange(0, context_length) * dt * exp_decay)/(npts - np.arange(0, context_length)))
    tau_[0] *= 0.5
    tau_[-1] *= 0.5
    tau = np.concatenate((tau_, np.zeros(npts - context_length)))
    if symmetric:
        tau_[0] *=  2
        tau /= 2
        toep = matmul_toeplitz((tau,tau), matrix)
    else:
        toep = matmul_toeplitz((tau,np.zeros_like(tau)), matrix)
    return toep

import numpy as np


