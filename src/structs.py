from typing import TypedDict, Union, Callable

from numpy.typing import ArrayLike
from torch import Tensor


class FitResult(TypedDict):
    U: Union[ArrayLike, Tensor]
    V: Union[ArrayLike, Tensor]
    svals: Union[Union[ArrayLike, Tensor], None]


class EigResult(TypedDict):
    values: Union[ArrayLike, Tensor]
    left: Union[Union[ArrayLike, Tensor],None]
    right: Union[ArrayLike, Tensor]
    bias: Union[ArrayLike, Tensor]
    type: str
    centered: bool

class ModeResult(TypedDict):
    decay_rates : Union[ArrayLike, Tensor]
    frequencies : Union[ArrayLike, Tensor]
    modes : Union[ArrayLike, Tensor]
    conditioning : Union[ArrayLike, Tensor]
