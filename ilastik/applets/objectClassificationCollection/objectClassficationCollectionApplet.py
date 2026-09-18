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
from ilastik.applets.base.standardApplet import StandardApplet

from ilastik.applets.objectClassificationCollection.objectClassificationCollectionSerializer import (
    ObjectClassificationCollectionSerializer,
)

from .opObjectClassificationCollection import OpOCC


class ObjectClassificationCollectionApplet(StandardApplet):

    def __init__(self, workflow, projectFileGroupName):

        self._topLevelOperator = OpOCC(parent=workflow)
        super().__init__(name=projectFileGroupName, workflow=workflow)
        self._projectFileGroupName = projectFileGroupName
        self._serializers = [ObjectClassificationCollectionSerializer("oc_collection", self.topLevelOperator)]
        self._topLevelOperator.progress_signal.subscribe(self.progressSignal)

    def getMultiLaneGui(self):
        """
        Override from base class. The label that is initially selected needs to be selected after volumina knows
        the current layer stack. Which is only the case when the gui objects LayerViewerGui.updateAllLayers run at
        least once after object init.
        """
        from .objectClassificationCollectionGui import ObjectClassificationCollectionGui

        multi_lane_gui = super().getMultiLaneGui()
        guis = multi_lane_gui.getGuis()
        for gui in guis:
            if isinstance(gui, ObjectClassificationCollectionGui) and gui.labelListData.selectedIndex().row() < 0:
                gui.selectLabel(0)
                gui.isInitialized = True
        return multi_lane_gui

    @property
    def dataSerializers(self):
        return self._serializers

    @property
    def topLevelOperator(self) -> OpOCC:
        return self._topLevelOperator

    @property
    def singleLaneGuiClass(self):
        from .objectClassificationCollectionGui import (
            ObjectClassificationCollectionGui,
        )  # Prevent imports of QT classes in headless mode

        return ObjectClassificationCollectionGui
