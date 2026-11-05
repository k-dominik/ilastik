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
from .opDataSelection import OpMultiLaneDataSelectionGroup


class OpDataSelectionFileList(OpMultiLaneDataSelectionGroup):
    """Keep the shell/serializer lane interface, limiting it to lane zero.

    The lane is created when a collection is first configured. Individual files
    are rows of its FileListTable and never cause additional lanes to be added.
    """

    def _insertInnerOperator(self, index, length):
        if index != 0 or length != 1:
            raise ValueError("File-list data selection supports only one lane.")
        return super()._insertInnerOperator(index, length)

    def addLane(self, laneIndex):
        if laneIndex != 0:
            raise ValueError("File-list data selection supports only lane zero.")
        return super().addLane(laneIndex)
