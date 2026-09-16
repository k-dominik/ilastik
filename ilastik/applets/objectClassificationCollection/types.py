import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from lazyflow.base import ItemId
from lazyflow.operators.ioOperators.types import RowBase


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class EmbeddingVector(RowBase):
    embedding_vector: npt.NDArray[np.floating]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LabelRow(RowBase):
    label: int


@dataclass
class AdaptionParameters:
    n_unlabeled: int
    labeled_fraction: float


LabelTable = dict[ItemId, LabelRow]
