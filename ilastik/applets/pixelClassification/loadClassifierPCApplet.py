###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
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
import logging
from ilastik.applets.base.standardApplet import StandardApplet
from ilastik.applets.pixelClassification.opLoadClassifierPC import OpLoadClassifierPixelClassification

logger = logging.getLogger('__name__')


class LoadClassifierPixelClassificationApplet(StandardApplet):
    """
    Simple applet for loading a pre-trained from file and predicting supplied
    images.
    """
    def __init__(self, workflow, guiName, projectFileGroupName):
        super(self.__class__, self).__init__(guiName, workflow)
        self._serializableItems = []

    @property
    def singleLaneOperatorClass(self):
        """
        from `StandardApplet`:
            Return the operator class which handles a single image.
            Single-lane applets should override this property.
            (Multi-lane applets must override ``topLevelOperator`` directly.)
        """
        return OpLoadClassifierPixelClassification

    @property
    def broadcastingSlots(self):
        """
        from `StandardApplet`:
            Slots that should be connected to all image lanes are referred to as
            "broadcasting" slots. Single-lane applets should override this
            property to return a list of the broadcasting slots' names.
            (Multi-lane applets must override ``topLevelOperator`` directly.)
        """
        return []

    @property
    def singleLaneGuiClass(self):
        from loadClassifierPCGui import LoadClassifierPixelClassificationGui
        return LoadClassifierPixelClassificationGui

    @property
    def dataSerializers(self):
        return self._serializableItems
