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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class EmbeddingVector(RowBase):
    embedding_vector: npt.NDArray[np.floating]


def dump_embedding_table(table: "EmbeddingTable") -> dict[str, Any]:
    ids = np.array([x.id for x in table.values()], dtype="uint64")
    embeddings = np.stack([x.embedding_vector for x in table.values()], axis=0)
    return dict(indices=ids, embeddings=embeddings)


def restore_table(data_dict: dict[Literal["indices", "embeddings"], Any]) -> "EmbeddingTable":
    return {
        ItemId(id): EmbeddingVector(id=id, embedding_vector=embedding)
        for id, embedding in zip(data_dict["indices"], data_dict["embeddings"])
    }


EmbeddingTable = Annotated[
    dict[ItemId, EmbeddingVector], PlainSerializer(dump_embedding_table), BeforeValidator(restore_table)
]
