import logging
from math import ceil, pi, sqrt
from typing import Literal

import numpy as np
import scipy.linalg
import torch

from numpy.typing import ArrayLike 
from typing import Union, Callable, Optional

from LaRRR.src.linalg import weighted_norm, toeplitz_integrator, toeplitz_generator
from LaRRR.src.structs import EigResult,  ModeResult
from LaRRR.src.utils import fuzzy_parse_complex, tonp, frnp, sqrtmh

from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KernelDensity
from scipy.integrate import cumulative_trapezoid

import torch
import torch.nn as nn


logger = logging.getLogger("LaRRR")


def evaluate_eigenfunction(
    eig_result: EigResult,
    which: Literal["left", "right"],
    data: ArrayLike,  # Feature matrix of the shape [num_evaluation_points, features] or kernel matrix of the shape [num_evaluation_points, num_training_points] 
    dual: bool = False  # Whether the algorithm is dual or primal
):
    vr_or_vl = eig_result[which]
    if dual | (which == "left"):
        rsqrt_dim = (data.shape[1]) ** (-0.5)
        return np.linalg.multi_dot([rsqrt_dim * data, vr_or_vl])
    else:       
        return data @ vr_or_vl

def modes(
    eig_result: EigResult,
    initial_conditions: Union[None,ArrayLike] = None,  # Feature matrix of the shape [num_initial_conditions, features] or kernel matrix of the shape [num_initial_conditions, num_training_points]
    obs_train: Union[None,ArrayLike] = None  # Observable to be predicted evaluated on the trajectory data, shape [num_training_points, obs_features]
) -> ModeResult:
    evals = eig_result["values"]
    levecs = eig_result["left"]

    if eig_result["type"] == "dual":
        dual = True
    else:
        dual = False

    if initial_conditions is None:
        conditioning_ = None

    if obs_train is None:
        modes_ = None
    else:
        npts = obs_train.shape[0]

    if eig_result["centered"]:
        if obs_train is not None:
            if obs_train.ndim == 1:
                obs_train = np.expand_dims(obs_train, axis=1)
            obs_train_mean = np.mean(obs_train, axis=0, keepdims=True)
            obs_train = obs_train - obs_train_mean
        if eig_result['type'] == 'dual':
            if initial_conditions is not None:
                initial_conditions_mean = initial_conditions.mean(axis=1, keepdims=True)
                initial_conditions = initial_conditions - initial_conditions_mean        
        evals = np.append(evals, 0.)

    if obs_train is not None:
        str = 'abcdefgh' # Maximum number of feature dimensions is 8
        einsum_str = ''.join([str[k] for k in range(obs_train.ndim - 1)]) # string for features
        modes_ = np.einsum('nr,n'+einsum_str+'->r'+einsum_str,  levecs.conj(), obs_train) /sqrt(npts) # [rank, features]
        if eig_result["centered"]:
            modes_ = np.append(modes_, obs_train_mean, axis = 0)

    if initial_conditions is not None:
        if initial_conditions.ndim==1:
            initial_conditions = np.expand_dims(initial_conditions,axis = 0)    
        conditioning_ = evaluate_eigenfunction(eig_result, "right", initial_conditions, dual=dual).T  # [rank, num_initial_conditions]
        if eig_result["centered"]:
            conditioning_ = np.append(conditioning_, np.expand_dims(np.repeat(1., initial_conditions.shape[0]),axis=0),axis=0)

    result: ModeResult = {"decay_rates": -evals.real, "frequencies": evals.imag/(2*np.pi), "modes": modes_, "conditioning": conditioning_ }    
    return result

def kmd_filter(kmd: ModeResult,
           indices: Union[ArrayLike,None] = None,
           decay_rates: Union[Callable[ArrayLike,ArrayLike],None] = None, 
           frequencies: Union[Callable[ArrayLike,ArrayLike],None] = None, 
           eps: float = 1e-6,
           real:bool = True, # If the states and observables are real valued, return only positive frequencies 
)-> ModeResult:
    modes_normsq = np.sqrt(np.sum(np.abs(kmd['modes']*kmd['modes'].conj()), axis = tuple(range(1, kmd['modes'].ndim))))
    relevant_modes = modes_normsq > eps*np.max(modes_normsq)
    filter = relevant_modes
    if indices is not None:
        indices_ = np.isin(np.arange(kmd["modes"].shape[0]), indices)
        filter = filter*indices_
    if decay_rates is not None:
        filter = filter*decay_rates(kmd["decay_rates"])
    if frequencies is not None:
        filter = filter*frequencies(kmd["frequencies"])
    if real:
        freq = kmd["frequencies"][filter]
        rates = kmd["decay_rates"][filter]
        if kmd['modes'] is not None:
            modes = kmd["modes"][filter]
        if kmd["conditioning"] is not None:
            conditioning = kmd["conditioning"][filter]
        idx_p = freq > 0
        #corr_ = np.expand_dims(np.exp(-2*freq[idx_p]*2*np.pi*1j),axis=tuple(range(-(modes.ndim-freq.ndim), 0)))
        #modes[idx_p] = modes[idx_p] + corr_ * modes[idx_p].conj()
        modes[idx_p] *= 2
        filter_ = np.logical_not(freq < 0)

        if kmd['modes'] is not None:
            modes_ = modes[filter_]
        else:
            modes_ = None
        if kmd["conditioning"] is not None:
            conditioning_ = conditioning[filter_]
        else:
            conditioning_ = None

        result: ModeResult = {"decay_rates": rates[filter_], "frequencies": freq[filter_], "modes": modes_, "conditioning": conditioning_}
    else:

        if kmd['modes'] is not None:
            modes_ = kmd["modes"][filter]
        else:
            modes_ = None
        if kmd["conditioning"] is not None:
            conditioning_ = kmd["conditioning"][filter]
        else:
            conditioning_ = None

        result: ModeResult = {"decay_rates": kmd["decay_rates"][filter], "frequencies": kmd["frequencies"][filter], "modes": modes_, "conditioning": conditioning_}
    return result


def predict(
    t: Union[float,ArrayLike],  # time in the same units as dt
    mode_result: ModeResult,
    real: bool = True  # If the states and observables are real valued, return real predictions
) -> ArrayLike: # shape [num_init_cond, features] or if num_t>1 [num_init_cond, num_time, features]
    if type(t) == float:
        t = np.array([t])
    evals = -mode_result["decay_rates"]+2*np.pi*1j*mode_result["frequencies"]
    to_evals = np.exp(evals[:,None]*t[None,:]) # [rank,time_steps]

    modes_ = mode_result["modes"]
    conditioning_ = mode_result["conditioning"]
    modes_ = np.expand_dims(modes_, axis = 1)
    dims_to_add = modes_.ndim - conditioning_.ndim
    conditioning_ = np.expand_dims(conditioning_, axis=tuple(range(-dims_to_add, 0)))
    modes = conditioning_*modes_ # [rank, num_init_cond, obs_features]

    str = 'abcdefgh' # Maximum number of feature dimensions is 8
    einsum_str = ''.join([str[k] for k in range(modes.ndim - 2)]) # string for features
    predictions = np.einsum('rs,rm'+einsum_str+'->ms'+einsum_str, to_evals, modes)
    if predictions.shape[0]==1 or predictions.shape[1]==1: # If only one time point or one initial condition is requested, remove unnecessary dims
        predictions = np.squeeze(predictions)
    if real:
        return predictions.real
    else:
        return predictions

def delay_embedding(X: ArrayLike, memory_length:int = 1, backend: Union['torch','numpy'] = 'numpy')->ArrayLike:
    n = X.shape[0]
    window_shape=n-memory_length
    if backend=='numpy':
        X = np.lib.stride_tricks.sliding_window_view(X, window_shape=window_shape, axis=0).T.reshape(window_shape,-1,order='F')
    elif backend=='torch':
        X = X.unfold(0,window_shape,1).reshape(window_shape,-1).T.reshape(window_shape,-1,order='F')
    else:
        logger.warning(
            f"Warning: Backend {backend} is not supported."
        )
    return X

def smooth_cdf(values, cdf): # Moved smooth_cdf here from NCP/utils.py
    scdf = IsotonicRegression(y_min=0., y_max=cdf.max()).fit_transform(values, cdf)
    if scdf.max() <= 0:
        return np.zeros(values.shape)
    scdf = scdf/scdf.max()
    return scdf

def find_best_quantile(x, cdf, alpha):
    x = x.flatten()
    t0 = 0
    t1 = 1
    best_t0 = 0
    best_t1 = -1
    best_size = np.inf

    while t0 < len(cdf):
        # stop if left border reaches right end of discretisation
        if cdf[t1] - cdf[t0] >= 1-alpha:
            # if x[t0], x[t1] is a confidence interval at level alpha, compute length and compare to best
            size = x[t1] - x[t0]
            if size < best_size:
                best_t0 = t0
                best_t1 = t1
                best_size = size
            # moving t1 to the right will only increase the size of the interval, so we can safely move t0 to the right
            t0 += 1
        
        elif t1 == len(cdf)-1:
            # if x[t0], x[t1] is not a confidence interval with confidence at least level alpha, 
            #and t1 is already at the right limit of the discretisation, then there remains no more pertinent intervals
            break
        else:
            # if moving x[t0] to the right reduces the level, we need to increase t1
            t1 += 1
    return x[best_t0], x[best_t1]


def predict_quantiles(eig_result: EigResult,
                        initial_conditions: ArrayLike,  
                        observables: ArrayLike, 
                        support:ArrayLike,
                        time : Union[ArrayLike, float] = 1., 
                        alpha: float = 0.05,
                        dual: bool = False,
                        get_cdfs = False)->ArrayLike:
    n_samples = observables.shape[0]
    n_features = observables.shape[1]
    support_shape = support.shape
    if len(support_shape) == 1:
        support = support[None,:].reshape(1, -1)
        if n_features > 1:
            support = support.repeat(n_features, axis=0)
    if type(time) == float:
        time = np.array([time])
    
    # Reshape obs to (n_samples, n_features, 1) for broadcasting
    observables_reshaped = observables.reshape(n_samples, n_features, 1)

    # Reshape support to (1, n_features, n_points) for broadcasting
    support_reshaped = support[None, :, :]

    # Perform the comparison
    # This will broadcast to shape (n_samples, n_features, n_points)
    cdf_obs = (observables_reshaped <= support_reshaped).astype(int)
    

    kmd_cdf = modes(eig_result=eig_result, initial_conditions=initial_conditions, obs_train=cdf_obs, dual=dual)
    predicted_cdf = predict(t=time, mode_result=kmd_cdf).real

    if initial_conditions.shape[0] == 1:
        predicted_cdf = predicted_cdf[None, :, :]
    if len(time) == 1:
        predicted_cdf = predicted_cdf[:, None, :]
    shape_ = predicted_cdf.shape[:-1]
    quantiles = np.zeros((*shape_, 2))
    for i in range(shape_[0]):
        for t in range(shape_[1]):
            for f in range(n_features):
                predicted_cdf[i,t,f,:] = smooth_cdf(support[f],predicted_cdf[i,t,f,:])
                quantiles[i,t, f, :] = find_best_quantile(support[f], predicted_cdf[i,t,f,:], alpha)
    if get_cdfs:
        return quantiles.squeeze(), predicted_cdf.squeeze()
    else:
        return quantiles.squeeze()


class OrthogonalRandomFeatures(torch.nn.Module):
    def __init__(
        self,
        # Possibly just pass the environment specs here.
        input_dim: int,
        num_random_features: int = 1024,
        length_scale: Union[float, torch.Tensor] = 1.0,
        rng_seed: Optional[int] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_random_features = num_random_features
        if torch.is_tensor(length_scale):
            assert length_scale.ndim == 0 or (
                length_scale.ndim == 1 and length_scale.shape[0] == self.state_dim
            )
        else:
            length_scale = torch.tensor(length_scale)

        W = self._sample_orf_matrix(self.input_dim, self.num_random_features, rng_seed)
        self.register_buffer("rff_matrix", W)
        self.register_buffer(
            "random_offset", torch.rand(self.num_random_features) * 2 * pi
        )
        self.register_buffer("length_scale", length_scale)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(
                self.input_dim, bias=False, elementwise_affine=False
            )
        else:
            self.layer_norm = nn.Identity()

    def _sample_orf_matrix(self, input_dim, num_random_features, rng_seed):
        num_folds = ceil(num_random_features / input_dim)
        rng_torch = rng_seed if rng_seed is None else torch.manual_seed(rng_seed)

        G = torch.randn(
            num_folds,
            input_dim,
            input_dim,
            generator=rng_torch,
        )
        Q, _ = torch.linalg.qr(
            G, mode="complete"
        )  # The _columns_ in each batch of matrices in Q are orthonormal.

        Q = Q.transpose(
            -1, -2
        )  # The _rows_ in each batch of matrices in Q are orthonormal.

        S = torch.tensor(
            scipy.stats.chi.rvs(
                input_dim,
                size=(num_folds, input_dim, 1),
                random_state=rng_seed,
            ),
        )

        W = Q * S  # [num_folds, input_dim, input_dim]
        W = torch.cat(
            [W[fold_idx, ...] for fold_idx in range(num_folds)], dim=0
        )  # Concatenate columns [num_folds*input_dim, input_dim]
        W = W[:num_random_features, :]
        return W.T

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.layer_norm(inputs)
        Z = torch.tensordot(
            inputs
            / self.length_scale,  # Length_scale should broadcast to the batch dimension
            self.rff_matrix.to(inputs.dtype),
            dims=1,
        )
        assert self.random_offset.shape[0] == Z.shape[-1]
        Z = torch.cos(Z + self.random_offset) / sqrt(0.5 * self.num_random_features)
        return Z  # [batch_size, num_random_features]