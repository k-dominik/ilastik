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

from ilastik.applets.base.appletSerializer import (
    AppletSerializer,
)
from ilastik.applets.base.appletSerializer.slotSerializer import SerialDataclassDictSlot
from ilastik.applets.objectFeatureCollection.opObjectFearturesCollection import EmbeddingTable

if TYPE_CHECKING:
    from .opObjectFearturesCollection import OpObjectFeaturesCollection


class ObjectFeatureCollectionSerializer(AppletSerializer):
    # FIXME: predictions can only be saved, not loaded, because it
    # would call setValue() on a connected slot

    def __init__(self, topGroupName, operator: "OpObjectFeaturesCollection"):
        self.VERSION = 1  # Make sure to bump the version in case you make any changes in the serialization
        serialSlots = [
            SerialDataclassDictSlot[EmbeddingTable](
                operator.Embedding, EmbeddingTable, operator.embedding_cache, operator.EmbeddingCacheInput
            ),
        ]

        super().__init__(topGroupName, slots=serialSlots, operator=operator)
