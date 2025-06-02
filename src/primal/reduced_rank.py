import logging
from math import sqrt
from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve, eig, eigh, lstsq, qr
from scipy.sparse.linalg import eigsh

from LaRRR.src.structs import EigResult
from LaRRR.src.linalg import add_diagonal_, rank_reveal, weighted_norm,toeplitz_integrator, toeplitz_generator
from LaRRR.src.utils import fuzzy_parse_complex ,center_matrix

logger = logging.getLogger("LaRRR")


def fit(
    Z : ArrayLike,  # Feature matrix for equaly spaced trajectory data of the shape [num_training_points, features]
    dt: float,  # Time step
    shift: float,  # Shift for resolvent operator
    context_length: int,  # Number of context points, for TO is 1, for IG at least 2
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    symmetric: bool = False,  # Whether the generator is self-adjoint operator or not
    svd_solver: Literal[
        "arnoldi", "full"
    ] = "full",  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
) -> EigResult:
    # Number of data points
    npts = Z.shape[0]
    eps = 1000.0 * np.finfo(Z.dtype).eps
    penalty = max(eps, tikhonov_reg)

    Z_int = toeplitz_integrator(matrix = Z, exp_decay=shift, npts=npts, context_length = context_length, dt=dt, symmetric=symmetric)

    H =  np.linalg.multi_dot([Z_int.T, Z]) / npts
    if context_length==0: #TO for time step dt is learned 
        C = Z[:-1].T @ Z[:-1] / (npts-1)
    else:
        C = Z.T @ Z / npts
    add_diagonal_(C, penalty)

    # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
    # Prefer svd_solver == 'randomized' in such a case.
    if svd_solver == "arnoldi":
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        num_arnoldi_eigs = min(rank + 5, npts)
        values, vectors = eigsh(H@H.T, k=num_arnoldi_eigs, M=C)
    elif svd_solver == "full":  # 'full'
        values, vectors = eigh(H@H.T, C, overwrite_a=True, overwrite_b=True)
    else:
        raise ValueError(f"Unknown svd_solver: {svd_solver}")
    if rank == Z.shape[1]: 
        #logger.warning(
        #    f"Warning: Full rank estimator is chosen, hence rank reduction bias is not estimated."
        #)
        numerically_nonzero_values_idxs = rank_reveal(values, rank, ignore_warnings=False)
        values = np.sqrt(values[numerically_nonzero_values_idxs])
        #bias_sigma = 1
        vectors = vectors[:, numerically_nonzero_values_idxs]
    else:
        numerically_nonzero_values_idxs = rank_reveal(values, rank+1, ignore_warnings=False)
        values = np.sqrt(values[numerically_nonzero_values_idxs[:-1]])
        #bias_sigma = np.sqrt(values[numerically_nonzero_values_idxs[0]])
        vectors = vectors[:, numerically_nonzero_values_idxs[:-1]]

    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        logger.warning(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )

    # Eigenvector normalization
    vecs_norm = weighted_norm(vectors, C)

    stable_values_idxs = rank_reveal(
        vecs_norm, rank, rcond=1000.0 * np.finfo(values.dtype).eps
    )
    V = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]
    values = values[stable_values_idxs]

    W_YX = np.linalg.multi_dot([V.T, H, V])

    if symmetric:
        values, vr = eigh(W_YX, overwrite_a=True, overwrite_b=True)
        vl = vr
    else:
        values, vl, vr = eig(W_YX, left=True, right=True)  
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
    
    ## Normalization in RKHS norm
    rcond = 1000.0 * np.finfo(Z.dtype).eps
    vr = V @ vr
    r_norm = np.linalg.norm(vr, axis=0)
    r_norm = np.where(np.abs(r_norm) < rcond, np.inf, r_norm)
    vr = vr / r_norm
    #bias = bias_sigma / weighted_norm(vr,C) 
    # Biorthogonalization
    vl = np.linalg.multi_dot([Z_int, V, vl]) / sqrt(npts)
    l_norm = np.sum((Z.T@vl) * vr, axis=0) / sqrt(npts)
    l_norm = np.where(np.abs(l_norm) < rcond, np.inf, l_norm)
    vl = vl / l_norm

    result: EigResult = {"values": values, "left": vl , "right": vr, "bias": None, "type": "primal", "centered": False}
    return result

def fit_to(
    Z : ArrayLike,  # Feature matrix for equaly spaced trajectory data of the shape [num_training_points, features]
    dt: float,  # Time step
    step: int = 1, # Multiple of time-step  
    tikhonov_reg: float = 0.,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: Union[int,None] = None,  # Rank of the estimator
    centered : bool = False,  # Whether we learn only non-trivial eigenvalues (True) or not (False)
    symmetry: Literal[
        "symmetric", None
    ] = None,
    # Whether the generator is self-adjoint, skew-adjoint or neither
    svd_solver: Literal[
        "arnoldi", "full"
    ] = "arnoldi",  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
) -> EigResult:
    #if symmetry == "antisymmetric":
    #    centered = True

    if centered:
        #centering = lambda X : X - np.mean(X, axis=0,keepdims=True)
        Z = Z - np.mean(Z, axis=0,keepdims=True)
    
    # Number of data points
    npts = (Z.shape[0] - step)//step
    eps = 1000.0 * np.finfo(Z.dtype).eps
    penalty = max(eps, tikhonov_reg)
    
    C = Z[:-step:step].T @ Z[:-step:step] / npts
    if symmetry=='symmetric':
        H =  Z[:-step:step].T @ Z[step::step] / npts
        H +=  H.T
        H /= 2
    elif symmetry=='antisymmetric':
#        step = 2
#        npts -= (step-1)
        T =  Z[:-step:step].    T @ Z[step::step] / npts
        Hsin = (T-T.T)/2
        Hcos = (T+T.T)/2
        H = T
    else:
        H =  Z[:-step:step].T @ Z[step::step] / npts

    add_diagonal_(C, penalty)

    if rank is not None:
        # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
        # Prefer svd_solver == 'randomized' in such a case.
        if svd_solver == "arnoldi":
            # Adding a small buffer to the Arnoldi-computed eigenvalues.
            num_arnoldi_eigs = min(rank + 5, Z.shape[1])
            values, vectors = eigsh(H@H.T, k=num_arnoldi_eigs, M=C)
        elif svd_solver == "full":  # 'full'
            values, vectors = eigh(H@H.T, C, overwrite_a=True, overwrite_b=True)
        else:
            raise ValueError(f"Unknown svd_solver: {svd_solver}")
        
        if rank == Z.shape[1]: 
            #logger.warning(
            #    f"Warning: Full rank estimator is chosen, hence rank reduction bias is not estimated."
            #)
            numerically_nonzero_values_idxs = rank_reveal(values, rank, ignore_warnings=False)
            values = np.sqrt(values[numerically_nonzero_values_idxs])
            bias_sigma = 1
            vectors = vectors[:, numerically_nonzero_values_idxs]
        else:
            numerically_nonzero_values_idxs = rank_reveal(values, rank+1, ignore_warnings=False)
            bias_sigma = sqrt(np.abs(values[numerically_nonzero_values_idxs][-1]))
            values = values[numerically_nonzero_values_idxs][:-1]
            vectors = vectors[:, numerically_nonzero_values_idxs][:,:-1]

        # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
        if not np.all(np.abs(values) >= tikhonov_reg):
            logger.warning(
                f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
            )

        # Eigenvector normalization
        vecs_norm = weighted_norm(vectors, C)

        stable_values_idxs = rank_reveal(
            vecs_norm, rank, rcond=1000.0 * np.finfo(values.dtype).eps
        )
        V = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]
        values = values[stable_values_idxs]
    else:
        bias_sigma = 1
        values, vectors = eigh(C, overwrite_a=True, overwrite_b=True)
        numerically_nonzero_values_idxs = rank_reveal(values, Z.shape[1], ignore_warnings=False)
        values = np.sqrt(values[numerically_nonzero_values_idxs])
        vectors = vectors[:, numerically_nonzero_values_idxs]
        V = vectors/values

    if symmetry == 'symmetric':
        W = np.linalg.multi_dot([V.T, H, V])
        values, vr = eigh(W, overwrite_a=True, overwrite_b=True)
        vl = vr
    else:
        W = np.linalg.multi_dot([V.T, H, V])
        values, vl, vr = eig(W, left=True, right=True)  
        values = fuzzy_parse_complex(values)

    r_perm = np.argsort(values)
    vr = vr[:, r_perm]
    # l_perm = np.argsort(values.conj())
    vl = vl[:, r_perm]
    values = values[r_perm]
    
    ## Normalization in RKHS norm
    rcond = 1000.0 * np.finfo(Z.dtype).eps
    vr = V @ vr
    r_norm = np.linalg.norm(vr, axis=0)
    r_norm = np.where(np.abs(r_norm) < rcond, np.inf, r_norm)
    vr = vr / r_norm
    #bias = bias_sigma / weighted_norm(vr,C) 
    
    # Biorthogonalization
    ZZ = np.zeros_like(Z) 
    if symmetry=='symmetric':
        ZZ[step:] += Z[:-step]
        ZZ[:-step] += Z[step:]
        ZZ /=2
        vl = np.linalg.multi_dot([ZZ, V, vl]) /sqrt(npts)
    else: #TO for time step dt is learned, i.e. exp(L dt), without symmetry constrains
        ZZ[step:] += Z[:-step]
        vl = np.linalg.multi_dot([ZZ, V, vl]) /sqrt(npts)

    #l_norm = np.where(np.abs(values) < rcond, np.inf, values.conj() / r_norm)
    #vl = vl / l_norm

    l_norm = np.sum((Z.T @ vl).conj() * vr / sqrt(npts), axis=0).conj()
    l_norm = np.where(np.abs(l_norm) < rcond, np.inf, l_norm)
    vl = sqrt((npts*step+step)/npts) * vl / l_norm

    # transforming the eigenvalues 
    if symmetry=='symmetric': # exp(L dt) is learned for L^*=L
        print(f'values {values}')
        values = np.log(values)/(step*dt)
    elif symmetry=='antisymmetric': # when L^*=-L hyperbolic sine of the generator L with time dt is learned, i.e. evals of sinh(L step dt)/step are 1j*values
        # #print(f'values {values}')
        # values = np.where(values < -1., -1., values)
        # values = np.where(values > 1., 1., values)  
        # #values = 1j*np.arcsin(values)/(step*dt)
        # #print(values)
        # omegas_ = np.arcsin(values)
        # print(f'corrections: {np.diag(vr.conj().T @ Hcos @ vr) > -rcond}')
        #values = 1j*np.where(np.diag(vr.conj().T @ Hcos @ vr) > -rcond, omegas_/(step*dt), 
        #                      np.where(omegas_ > 0., (np.pi - omegas_)/(step*dt), (-np.pi - omegas_)/(step*dt) ))
        print(f'values {values}')
        print(f'cos: {values_cos}')
        print(f'sin: {values_sin}')
        values = 1j*np.arctan2(values_sin, values_cos)/(step*dt)
        
    else: #TO for time step dt is learned, i.e. exp(L dt), without symmetry constrains
        values = np.log(values)/(step*dt)

    # Spectral bias
    bias = bias_sigma / weighted_norm(vr,C) 

    result: EigResult = {"values": values, "left": vl , "right": vr, "bias": bias, "type": "primal", "centered": centered}
    return result