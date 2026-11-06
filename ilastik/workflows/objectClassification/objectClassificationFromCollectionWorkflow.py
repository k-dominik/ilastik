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

from ilastik.applets.dataSelection.dataSelectionFileListApplet import DataSelectionFileListApplet
from ilastik.applets.objectClassificationCollection.objectClassficationCollectionApplet import (
    ObjectClassificationCollectionApplet,
)
from ilastik.applets.objectClassificationCollection.objectClassificationCollectionExportApplet import (
    ObjectClassificationCollectionExportApplet,
)
from ilastik.applets.objectClassificationCollection.opObjectClassificationCollection import OpOCC
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
        self._state_update_subscriptions = []
        self._workflow_cmdline_args = workflow_cmdline_args

        self.dataSelectionApplet = self.createDataSelectionApplet()

        self.objectFeatureCollectionApplet = ObjectFeatureCollectionApplet(
            workflow=self, projectFileGroupName="ObjectFeatureCollection"
        )

        self.objectClassificationApplet = ObjectClassificationCollectionApplet(
            workflow=self, projectFileGroupName="ObjectClassificationCollection"
        )
        self.dataExportApplet = ObjectClassificationCollectionExportApplet(self)
        self.dataExportApplet.topLevelOperator.WorkingDirectory.connect(
            self.dataSelectionApplet.topLevelOperator.WorkingDirectory
        )
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.objectFeatureCollectionApplet)
        self._applets.append(self.objectClassificationApplet)
        self._applets.append(self.dataExportApplet)

    def _after_init(self):
        super()._after_init()
        opFeatures = self.objectFeatureCollectionApplet.topLevelOperator
        opClassify = self.objectClassificationApplet.topLevelOperator
        # Observe both invalidation and completed computation/restoration of caches.
        # Readiness alone only says that a value can be requested.
        for slot in (
            self.dataExportApplet.topLevelOperator.FileList,
            opFeatures.embedding_cache.Output,
            opClassify.classifier_cache_original_embedding.Output,
            opClassify.classifier_cache_adapted_embedding.Output,
        ):
            for notify in (slot.notifyReady, slot.notifyUnready, slot.notifyDirty, slot.notifyValueChanged):
                self._state_update_subscriptions.append(notify(self._request_applet_state_update, defer=True))

    def _request_applet_state_update(self, *args):
        self.handleAppletStateUpdateRequested()

    def cleanUp(self):
        for unsubscribe in self._state_update_subscriptions:
            unsubscribe()
        self._state_update_subscriptions.clear()
        super().cleanUp()

    def createDataSelectionApplet(self):
        data_instructions = "Add the individual object images to the file collection shown on the right."
        c_at_end = ["yxc", "xyc"]
        for perm in itertools.permutations("zyx", 3):
            c_at_end.append("".join(perm) + "c")

        applet = DataSelectionFileListApplet(
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
        opClassify: OpOCC = self.objectClassificationApplet.topLevelOperator.getLane(laneIndex)
        # # assert opData.Table.ready()
        opFeatures.FileList.connect(opData.FileListTable)

        opClassify.FileList.connect(opData.FileListTable)
        opClassify.Embedding.connect(opFeatures.Embedding)

        opExport = self.dataExportApplet.topLevelOperator
        opExport.FileList.connect(opData.FileListTable)
        opExport.Predictions.connect(opClassify.Predictions)
        opExport.LabelNames.connect(opClassify.LabelNames)

    def handleAppletStateUpdateRequested(self):
        files = self.dataExportApplet.topLevelOperator.FileList
        input_ready = files.ready() and bool(files.value)

        feature_cache = self.objectFeatureCollectionApplet.topLevelOperator.embedding_cache
        features_ready = input_ready and feature_cache.Output.ready() and feature_cache.hasCacheValue()

        opClassify = self.objectClassificationApplet.topLevelOperator
        classifier_ready = features_ready and any(
            cache.Output.ready() and cache.hasCacheValue()
            for cache in (
                opClassify.classifier_cache_original_embedding,
                opClassify.classifier_cache_adapted_embedding,
            )
        )

        data_busy = self.dataSelectionApplet.busy
        features_busy = self.objectFeatureCollectionApplet.busy
        classification_busy = self.objectClassificationApplet.busy
        export_busy = self.dataExportApplet.busy

        self._shell.setAppletEnabled(
            self.dataSelectionApplet, not (features_busy or classification_busy or export_busy)
        )
        self._shell.setAppletEnabled(
            self.objectFeatureCollectionApplet,
            input_ready and not (data_busy or classification_busy or export_busy),
        )
        self._shell.setAppletEnabled(
            self.objectClassificationApplet,
            features_ready and not (data_busy or features_busy or export_busy),
        )
        self._shell.setAppletEnabled(
            self.dataExportApplet,
            classifier_ready and not (data_busy or features_busy or classification_busy),
        )
        self._shell.enableProjectChanges(not (data_busy or features_busy or classification_busy or export_busy))
