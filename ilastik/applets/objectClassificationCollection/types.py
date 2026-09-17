###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2026, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#          http://ilastik.org/license.html
###############################################################################
from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
from pydantic.dataclasses import dataclass

from lazyflow.base import ItemId
from lazyflow.operators.ioOperators.types import RowBase
from torch import Tensor, from_numpy


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LabelRow(RowBase):
    label: int


@dataclass
class AdaptionParameters:
    n_unlabeled: int
    labeled_fraction: float


@dataclass
class UmapRow(RowBase):
    x: float
    y: float


def dump_umap_table(table: "UmapTable") -> dict[str, Any]:
    ids = np.array([x.id for x in table.values()], dtype="uint64")
    umap = np.array([[row.x, row.y] for row in table.values()])
    return dict(indices=ids, umap=umap)


def restore_umap_table(data_dict: dict[Literal["indices", "umap"], Any]) -> "UmapTable":
    return {ItemId(id): UmapRow(id=id, x=row[0], y=row[1]) for id, row in zip(data_dict["indices"], data_dict["umap"])}


UmapTable = Annotated[dict[ItemId, UmapRow], PlainSerializer(dump_umap_table), BeforeValidator(restore_umap_table)]


def dump_label_table(table: "LabelTable") -> dict[str, Any]:
    ids = np.array([x.id for x in table.values()], dtype="uint64")
    labels = np.array([row.label for row in table.values()], dtype="uint8")
    return dict(indices=ids, labels=labels)


def restore_label_table(data_dict: dict[Literal["indices", "labels"], Any]) -> "LabelTable":
    return {ItemId(id): LabelRow(id=id, label=label) for id, label in zip(data_dict["indices"], data_dict["labels"])}


LabelTable = Annotated[dict[ItemId, LabelRow], PlainSerializer(dump_label_table), BeforeValidator(restore_label_table)]


def to_np(tensor_dict: dict[str, Any]) -> dict[str, Any]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in tensor_dict.items()}


def from_np(np_dict: dict[str, Any]) -> dict[str, Any]:
    return {name: from_numpy(np.atleast_1d(nparray)) for name, nparray in np_dict.items()}


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProjectorData:
    state_dict: dict[str, Tensor]
    adaptation_parameters: AdaptionParameters
    input_dim: int
    hidden_dim: int
    output_dim: int
    n_epochs: int
    # label_table
    # unlabeled_items


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProjectorDataS:
    state_dict: Annotated[dict[str, Tensor], PlainSerializer(to_np), BeforeValidator(from_np)]
    adaptation_parameters: AdaptionParameters
    input_dim: int
    hidden_dim: int
    output_dim: int
    n_epochs: int
    # label_table
    # unlabeled_items
