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
from ilastik.applets.base.applet import Applet
from ilastik.applets.base.appletSerializer import AppletSerializer, SerialSlot

from .opObjectClassificationCollectionExport import OpObjectClassificationCollectionExport


class ObjectClassificationCollectionExportApplet(Applet):
    def __init__(self, workflow, title="Prediction Export", projectFileGroupName="Collection Export"):
        super().__init__(title, syncWithImageIndex=False)
        self._op = OpObjectClassificationCollectionExport(parent=workflow)
        self._gui = None
        self._serializers = [
            AppletSerializer(projectFileGroupName, slots=[SerialSlot(self._op.OutputFilename)], operator=self._op)
        ]
        self._op.FileList.notifyReady(self._state_changed)
        self._op.FileList.notifyUnready(self._state_changed)

    def _state_changed(self, *args):
        self.appletStateUpdateRequested()

    @property
    def topLevelOperator(self):
        return self._op

    @property
    def dataSerializers(self):
        return self._serializers

    def getMultiLaneGui(self):
        if self._gui is None:
            from .objectClassificationCollectionExportGui import ObjectClassificationCollectionExportGui

            self._gui = ObjectClassificationCollectionExportGui(self, self._op)
        return self._gui
