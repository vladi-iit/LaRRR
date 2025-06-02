import logging
from math import sqrt
from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve, eig, eigh, lstsq, qr
from scipy.sparse.linalg import eigs

from LaRRR.src.structs import EigResult
from LaRRR.src.linalg import add_diagonal_, rank_reveal, weighted_norm, toeplitz_generator
from LaRRR.src.utils import fuzzy_parse_complex, center_matrix

logger = logging.getLogger("LaRRR")


def fit(
    K : ArrayLike,  # Kernel matrix for equaly spaced trajectory data
    dt: float,  # Time step
    shift: float,  # Shift for resolvent operator
    context_length: int,  # Number of context points, for TO is 1, for IG at least 2
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    symmetric: bool = False, # Whether the generator is self-adjoint or not
    svd_solver: Literal[
        "arnoldi", "full"
    ] = "arnoldi",  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
) -> EigResult:
    # Number of data points
    npts = K.shape[0]
    eps = 1000.0 * np.finfo(K.dtype).eps
    penalty = max(eps, tikhonov_reg) * npts

    toep = toeplitz_generator(exp_decay=shift, npts=npts, context_length = context_length, dt=dt, symmetric=symmetric)

    A = np.linalg.multi_dot([toep, K / sqrt(npts), toep.T,K / sqrt(npts)])
    
    if context_length==0: #TO for time step dt is learned 
        K[-1] = 0
        K[:,-1] = 0
        npts -=1
    
    penalty = max(eps, tikhonov_reg) * npts

    add_diagonal_(K, penalty)
    # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
    # Prefer svd_solver == 'randomized' in such a case.
    if svd_solver == "arnoldi":
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        num_arnoldi_eigs = min(rank + 5, npts)
        values, vectors = eigs(A, k=num_arnoldi_eigs, M=K)
    elif svd_solver == "full":  # 'full'
        values, vectors = eig(A, K, overwrite_a=True, overwrite_b=True)
    else:
        raise ValueError(f"Unknown svd_solver: {svd_solver}")
    # Remove the penalty from K (inplace)
    add_diagonal_(K, -penalty)

    numerically_nonzero_values_idxs = rank_reveal(values, rank, ignore_warnings=False)
    values = values[numerically_nonzero_values_idxs]
    vectors = vectors[:, numerically_nonzero_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        logger.warning(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )

    # Eigenvector normalization
    K_vecs = np.dot(K / sqrt(npts), vectors)
    vecs_norm = np.sqrt(
        np.sum(
            K_vecs**2 + tikhonov_reg * K_vecs * vectors * sqrt(npts),
            axis=0,
        )
    )
    if context_length==0:
        npts +=1

    stable_values_idxs = rank_reveal(
        vecs_norm, rank, rcond=1000.0 * np.finfo(values.dtype).eps
    )
    U = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]
    values = values[stable_values_idxs]

    # Ordering the results
    V = K @ U
    svals = np.sqrt(np.abs(values))

    W_YX = np.linalg.multi_dot([V.T, toep, V]) / npts

    if context_length==0:
        npts -=1
    W_X = np.linalg.multi_dot([U.T, K, U]) / npts
    if context_length==0:
        npts +=1

    values, vl, vr = eig(
        W_YX, left=True, right=True
    )  # Left -> V, Right -> U
    values = fuzzy_parse_complex(values)
    r_perm = np.argsort(values)
    vr = vr[:, r_perm]
    l_perm = np.argsort(values.conj())
    vl = vl[:, l_perm]
    values = values[r_perm]

    # transforming the eigenvalues 
    if context_length==0: #TO for time step dt is learned, i.e. exp(L dt) 
        values = np.log(values)/dt
    else: # resovlent (shift - L)^(-1) is learned by numerical integration over context_length steps of size dt
        values = shift - 1 / values

    rcond = 1000.0 * np.finfo(U.dtype).eps
    # Normalization in RKHS
    norm_r = weighted_norm(vr, W_X)
    norm_r = np.where(norm_r < rcond, np.inf, norm_r)
    vr = vr / norm_r

    # Bi-orthogonality of left eigenfunctions
    norm_l = np.diag(np.linalg.multi_dot([vl.T, W_YX, vr]))
    norm_l = np.where(np.abs(norm_l) < rcond, np.inf, norm_l)
    vl = vl / norm_l
    idx = np.argsort(values)
    result: EigResult = {"values": values[idx], "left": np.linalg.multi_dot([toep.T,V,vl[:,idx]]), "right": U @ vr[:,idx]}
    return result


def fit_to(
    K : ArrayLike,  # Kernel matrix for equaly spaced trajectory data
    dt: float,  # Time step
    step: int = 1, # time step for learning, time scale of the learning algorithm is step dt   
    tikhonov_reg: float = 0.,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: Union[int,None] = None,  # Rank of the estimator
    centered : bool = False,  # Whether we learn only non-trivial eigenvalues (True) or not (False)
    symmetry: Literal[
        "symmetric", None
    ] = None,
    # Whether the generator is self-adjoint or not
    svd_solver: Literal[
        "arnoldi", "full"
    ] = "arnoldi",  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
) -> EigResult:
    #if symmetry == "antisymmetric":
    #    centered = True

    if centered:
        K = center_matrix(K)

    npts = K.shape[0] - step   # Number of data points
    eps = 1000.0 * np.finfo(K.dtype).eps
    penalty = max(eps, tikhonov_reg)*npts

    if symmetry=='symmetric':
        M =  np.diag(np.ones(npts),step)
        M +=  M.T
        M /= 2
    else:
        M =  np.diag(np.ones(npts),step)

    if rank is not None:
        # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
        A = np.linalg.multi_dot([M, K / sqrt(npts), M.T, K / sqrt(npts)])
     
        add_diagonal_(K, penalty)
        if svd_solver == "arnoldi":
            # Adding a small buffer to the Arnoldi-computed eigenvalues.
            num_arnoldi_eigs = min(rank + 5, npts)
            values, vectors = eigs(A, k=num_arnoldi_eigs, M=K)
        elif svd_solver == "full":  # 'full'
            values, vectors = eig(A, K, overwrite_a=True, overwrite_b=True)
        else:
            raise ValueError(f"Unknown svd_solver: {svd_solver}")
        # Remove the penalty from K (inplace)
        add_diagonal_(K, -penalty)

        numerically_nonzero_values_idxs = rank_reveal(values, rank+1, ignore_warnings=False)
        bias_sigma_sq = np.abs(values[numerically_nonzero_values_idxs][-1])
        values = values[numerically_nonzero_values_idxs][:-1]
        vectors = vectors[:, numerically_nonzero_values_idxs][:,:-1]
        # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
        if not np.all(np.abs(values) >= tikhonov_reg):
            logger.warning(
                f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
            )

        # Eigenvector normalization
        K_vecs = np.dot(K / sqrt(npts), vectors)
        vecs_norm = np.sqrt(
            np.sum(
                K_vecs**2 + tikhonov_reg * K_vecs * vectors * sqrt(npts),
                axis=0,
            )
        )

        stable_values_idxs = rank_reveal(
            vecs_norm, rank, rcond=1000.0 * np.finfo(values.dtype).eps
        )
        U = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]
        values = values[stable_values_idxs]

        V = K @ U
    else:
        logger.error(
                f"Error: Full rank kernel method not supported."
            )
    
    # Eigenvalue decomposition
    if symmetry == 'symmetric':
        W = np.linalg.multi_dot([V.T, M, V]) / npts
        values, vr_ = eigh(W, overwrite_a=True, overwrite_b=True)
        vl_ = vr_
    else:
        W = np.linalg.multi_dot([V.T, M, V]) / npts
        values, vl_, vr_ = eig(W, left=True, right=True)  
        values = fuzzy_parse_complex(values)
    r_perm = np.argsort(values)
    vr_ = vr_[:, r_perm]
    # l_perm = np.argsort(values.conj())
    vl_ = vl_[:, r_perm]
    values = values[r_perm]
    
    rcond = 1000.0 * np.finfo(U.dtype).eps
    ## Normalization in RKHS norm
    vr = U @ vr_ 
    r_norm =weighted_norm(vr, K / npts)
    r_norm = np.where(np.abs(r_norm) < rcond, np.inf, r_norm)
    vr = vr / r_norm
    vr_ = vr_ / r_norm 
    
    
    # Biorthogonalization
    if symmetry=='symmetric':
        vl = np.linalg.multi_dot([M, V, vl_]) /sqrt(npts)
        vl_ /= sqrt(npts)
        l_norm = np.sum(vl_.conj() * (W@vr_), axis=0).conj() 
    else: #TO for time step dt is learned, i.e. exp(L dt), without symmetry constrains
        vl = np.linalg.multi_dot([M.T, V, vl_]) /sqrt(npts)
        vl_ /= sqrt(npts)
        l_norm = np.sum(vl_.conj() * (W@vr_), axis=0).conj()
    #l_norm = np.where(np.abs(values) < rcond, np.inf, values.conj() / r_norm)
    l_norm = np.where(np.abs(l_norm) < rcond, np.inf, l_norm)
    vl = vl / l_norm

    # Spectral bias
    Kvr = K@vr/sqrt(npts)
    bias =  np.sqrt( npts * bias_sigma_sq / np.sum( Kvr*Kvr.conj(), axis=0).real)

    # transforming the eigenvalues 
    if symmetry=='symmetric': #hyperbolic cosine of the generator L with time dt is learned, i.e. cosh(L dt), which is the same as exp(L dt) for L^*=L
        values = np.log(values)/(step*dt)
    else: #TO for time step dt is learned, i.e. exp(L dt), without symmetry constrains
        values = np.log(values)/(step*dt)

    # Correcting the normalization of eigenfunctions
    vr *=sqrt(1+step/npts)
    vl *=sqrt(1+step/npts)

    result: EigResult = {"values": values, "left": vl , "right": vr, "bias": bias, "type": "dual", "centered": centered}
    return result