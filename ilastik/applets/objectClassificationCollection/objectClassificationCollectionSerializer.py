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
from typing import TYPE_CHECKING

from ilastik.applets.base.appletSerializer import AppletSerializer, SerialClassifierSlot
from ilastik.applets.base.appletSerializer.slotSerializer import (
    SerialDataclassDictSlot,
    SerialLabelTableSlot,
    SerialListSlot,
    SerialSlot,
)
from ilastik.applets.objectClassificationCollection.opObjectClassificationCollection import EmbeddingSource, UmapRow

from .types import EmbeddingVector, LabelRow

if TYPE_CHECKING:
    from .opObjectClassificationCollection import OpOCC
    from lazyflow.slot import Slot
    import h5py


class EmbeddingSelectSerializer(SerialSlot[EmbeddingSource]):
    @staticmethod
    def _saveValue(group: "h5py.Group", name: str, value: EmbeddingSource):
        group.create_dataset(name, data=int(value))

    @staticmethod
    def _getValue(subgroup: "h5py.Group", slot: "Slot[EmbeddingSource]"):
        val = subgroup[()]
        slot.setValue(EmbeddingSource(val))


class ObjectClassificationCollectionSerializer(AppletSerializer):
    # FIXME: predictions can only be saved, not loaded, because it
    # would call setValue() on a connected slot

    def __init__(self, topGroupName, operator: "OpOCC"):
        self.VERSION = 1  # Make sure to bump the version in case you make any changes in the serialization
        serialSlots = [
            SerialListSlot(operator.LabelNames),
            SerialListSlot(operator.LabelColors, transform=lambda x: tuple(x.flat)),
            SerialListSlot(operator.PmapColors, transform=lambda x: tuple(x.flat)),
            EmbeddingSelectSerializer(operator.SelectEmbeddingSource),
            SerialDataclassDictSlot(
                operator.AdaptedEmbeddings, EmbeddingVector, operator.embedding_cache, operator.EmbeddingCacheInput
            ),
            SerialLabelTableSlot(
                operator.AnnotationsTable, LabelRow, operator.op_grid_labels, operator.AnnotationsTableCacheInput
            ),
            SerialClassifierSlot(
                operator.Classifier, operator.classifier_cache_original_embedding, name="ClassifierForests"
            ),
            SerialClassifierSlot(
                operator.ClassifierAdapted,
                operator.classifier_cache_adapted_embedding,
                name="ClassifierForests_adapted_features",
            ),
            SerialDataclassDictSlot(operator.Umap, UmapRow, operator.umap_cache, operator.UmapCacheInput),
            SerialDataclassDictSlot(
                operator.AdaptedUmap, UmapRow, operator.umap_adapted_cache, operator.AdaptedUmapCacheInput
            ),
        ]

        super().__init__(topGroupName, slots=serialSlots, operator=operator)
