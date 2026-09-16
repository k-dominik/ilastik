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
from ilastik.applets.labeling.labelingApplet import LabelingApplet

from ilastik.applets.objectClassificationCollection.objectClassificationCollectionSerializer import (
    ObjectClassificationCollectionSerializer,
)

from .opObjectClassificationCollection import OpOCC


class ObjectClassificationCollectionApplet(LabelingApplet):

    def __init__(self, workflow, projectFileGroupName, hintOverlayFile=None, pmapOverlayFile=None):
        if hintOverlayFile is not None:
            assert isinstance(hintOverlayFile, str)

        if not hasattr(self, "_topLevelOperator"):
            self._topLevelOperator = OpOCC(parent=workflow)

        super().__init__(workflow, projectFileGroupName)
        self._projectFileGroupName = projectFileGroupName
        self._serializers = [ObjectClassificationCollectionSerializer("oc_collection", self.topLevelOperator)]
        self._topLevelOperator.progress_signal.subscribe(self.progressSignal)
        self._topLevelOperator.progress_signal.subscribe(lambda x: print(f"progress {x=}"))

    def getMultiLaneGui(self):
        """
        Override from base class. The label that is initially selected needs to be selected after volumina knows
        the current layer stack. Which is only the case when the gui objects LayerViewerGui.updateAllLayers run at
        least once after object init.
        """
        from .objectClassificationCollectionGui import (
            ObjectClassificationCollectionGui,
        )  # Prevent imports of QT classes in headless mode

        multi_lane_gui = super(LabelingApplet, self).getMultiLaneGui()
        guis = multi_lane_gui.getGuis()
        if len(guis) > 0 and isinstance(guis[0], ObjectClassificationCollectionGui) and not guis[0].isInitialized:
            guis[0].selectLabel(0)
            guis[0].isInitialized = True
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
