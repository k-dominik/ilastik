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
from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui


class LoadClassifierPixelClassificationGui(LayerViewerGui):
    def __init__(self, *args, **kwargs):
        """
        from LayerViewerGui:
            Constructor.  **All** slots of the provided *topLevelOperatorView*
            will be monitored for changes.
            Changes include slot resize events, and slot ready/unready status
            changes.
            When a change is detected, the `setupLayers()` function is called,
            and the result is used to update the list of layers shown in the
            central widget.

            :param topLevelOperatorView: The top-level operator for the applet
              this GUI belongs to.
            :param additionalMonitoredSlots: Optional.  Can be used to add
              additional slots to the set of viewable layers (all slots from the
              top-level operator are already monitored).
            :param centralWidgetOnly: If True, provide only a central widget
              without drawer or viewer controls.
        """
        super(LoadClassifierPixelClassificationGui, self).__init__(*args, **kwargs)
