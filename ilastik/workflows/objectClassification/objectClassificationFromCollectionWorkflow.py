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
import enum
import itertools
from typing import TYPE_CHECKING

from ilastik.applets.dataSelection.dataSelectionApplet import DataSelectionApplet
from ilastik.applets.objectClassificationCollection.objectClassficationCollectionApplet import (
    ObjectClassificationCollectionApplet,
)
from ilastik.applets.objectFeatureCollection.objectFeaturesCollectionApplet import ObjectFeatureCollectionApplet
from ilastik.applets.objectFeatureCollection.opObjectFearturesCollection import OpObjectFeaturesCollection
from ilastik.utility.slot_name_enum import SlotNameEnum
from ilastik.workflow import Workflow
from lazyflow.graph import Graph

if TYPE_CHECKING:
    from ilastik.applets.base.applet import Applet


class OcFromCollection(Workflow):
    workflowName = "Embedding assisted Object Classification for Image Collection"
    workflowDescription = "Leverage feature embeddings to classify objects faster."
    defaultAppletIndex = 0  # show DataSelection by default

    @enum.unique
    class Roles(SlotNameEnum):
        RAW_DATA = enum.auto()

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName

    def __init__(self, shell, headless, workflow_cmdline_args, project_creation_args, *args, **kwargs):
        graph = Graph()
        super().__init__(shell, headless, workflow_cmdline_args, project_creation_args, graph=graph, *args, **kwargs)
        self.stored_classifier = None
        self._applets: list[Applet] = []
        self._workflow_cmdline_args = workflow_cmdline_args

        self.dataSelectionApplet = self.createDataSelectionApplet()

        self.objectFeatureCollectionApplet = ObjectFeatureCollectionApplet(
            workflow=self, projectFileGroupName="ObjectFeatureCollection"
        )

        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.objectFeatureCollectionApplet)

    def createDataSelectionApplet(self):
        data_instructions = "Load a folder of objects in separate images using the 'Raw Data' tab shown on the right."
        c_at_end = ["yxc", "xyc"]
        for perm in itertools.permutations("zyx", 3):
            c_at_end.append("".join(perm) + "c")

        applet = DataSelectionApplet(
            self,
            "Input Data",
            "Input Data",
            supportIlastik05Import=False,
            instructionText=data_instructions,
            forceAxisOrder=c_at_end,
        )
        applet.topLevelOperator.DatasetRoles.setValue(self.Roles.asDisplayNameList())
        return applet

    def connectLane(self, laneIndex: int):
        assert laneIndex == 0
        # Get a handle to each operator
        opData = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opFeatures: OpObjectFeaturesCollection = self.objectFeatureCollectionApplet.topLevelOperator.getLane(laneIndex)
        # # assert opData.Table.ready()
        opFeatures.FileList.connect(opData.FileListTable)
