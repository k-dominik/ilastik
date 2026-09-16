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
# 		   http://ilastik.org/license.html
###############################################################################
from ilastik.applets.base.standardApplet import StandardApplet

from ilastik.applets.objectFeatureCollection.objectFeatureCollectionSerializer import ObjectFeatureCollectionSerializer
from ilastik.applets.objectFeatureCollection.opObjectFearturesCollection import OpObjectFeaturesCollection


class ObjectFeatureCollectionApplet(StandardApplet):
    """Calculate features for each image in an image collection table"""

    def __init__(
        self,
        name="Object Feature Calculation",
        workflow=None,
        projectFileGroupName="ObjectFeaturesCollection",
        interactive=True,
    ):
        super().__init__(name=name, workflow=workflow, interactive=interactive)
        self._topLevelOperator = OpObjectFeaturesCollection(parent=workflow)
        self._serializableItems = [ObjectFeatureCollectionSerializer("oc_collection_features", self.topLevelOperator)]

        self._topLevelOperator.progress_signal.subscribe(self.progressSignal)

    @property
    def topLevelOperator(self):
        return self._topLevelOperator

    @property
    def broadcastingSlots(self):
        return []

    @property
    def singleLaneGuiClass(self):
        from ilastik.applets.objectFeatureCollection.objectFeatureCollectionGui import ObjectFeatureCollectionGui

        return ObjectFeatureCollectionGui

    @property
    def dataSerializers(self):
        return self._serializableItems
