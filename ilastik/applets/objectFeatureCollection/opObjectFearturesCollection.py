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
from typing import TYPE_CHECKING, Optional

from ilastik.applets.objectFeatureCollection.types import EmbeddingTable

from ilastik.applets.fileCollection.fileCollectionOps import FileList as FileListT
from ilastik.applets.fileCollection.fileCollectionOps import OpGrid, OpGridView
from ilastik.plugins import plugin_manager
from lazyflow.operator import InputSlot, Operator, OutputSlot, Slot
from lazyflow.operators.ioOperators.types import FileListDataRow
from lazyflow.operators.valueProviders import OpValueCache
from lazyflow.utility.orderedSignal import OrderedSignal

if TYPE_CHECKING:
    from ilastik.plugins.types import ObjectFeaturesPlugin
    from lazyflow.base import ItemId
    from lazyflow.graph import Graph
    from lazyflow.rtype import Roi


class OpEmbedding(Operator):
    FileList = InputSlot[FileListT]()
    SelectedPlugin = InputSlot[str](value="dinov2:small-reg")
    Embedding = OutputSlot[EmbeddingTable](stype="object")

    def __init__(
        self,
        parent: Optional[Operator] = None,
        graph: Optional["Graph"] = None,
        FileList: Optional[InputSlot[FileListT] | OutputSlot[FileListT]] = None,
        SelectedPlugin: Optional[str | InputSlot[str] | OutputSlot[str]] = None,
    ):
        super().__init__(parent=parent, graph=graph)
        self.FileList.setOrConnectIfAvailable(FileList)
        self.SelectedPlugin.setOrConnectIfAvailable(SelectedPlugin)
        self._table: dict["ItemId", FileListDataRow] = {}
        self._plugin: "ObjectFeaturesPlugin" | None = None
        self._progress_signal = OrderedSignal()

    @property
    def progress_signal(self):
        return self._progress_signal

    def setupOutputs(self):
        self._table = dict(self.FileList.value)
        assert not self._table or all(
            x.dtype == next(iter(self._table.values())).dtype for x in self._table.values()
        ), "All dtypes are required to be the same"
        # @output_axiskeys = "".join(_OUTPUT_AXIS_KEYS)
        # assert not self._table or all(
        #     x. == output_axiskeys for x in self._table
        # ), f"All Embedding axis keys must be {_OUTPUT_AXIS_KEYS}"

        selected_plugin = self.SelectedPlugin.value
        self._plugin = plugin_manager.get_object_feature_plugin_by_name(selected_plugin)

        self.Embedding.meta.shape = (1,)
        self.Embedding.meta.dtype = object

    def execute(self, slot, subindex, roi, result):
        assert slot == self.Embedding
        return [self._plugin.compute_batchwise(self._table, progress_callback=self._progress_signal)]

    def propagateDirty(self, slot, subindex, roi):
        self._embedding = {}


class OpObjectFeaturesCollection(Operator):
    FileList = InputSlot[FileListT]()
    SelectedPlugin = InputSlot[str](value="dinov2:small-reg")

    Output = OutputSlot()
    EmbeddingCacheInput = InputSlot[EmbeddingTable](optional=True)
    Embedding = OutputSlot[EmbeddingTable](stype="object")

    def __init__(
        self,
        parent: Optional[Operator] = None,
        graph: Optional["Graph"] = None,
        FileList: Optional[Slot[FileListT]] = None,
    ):
        super().__init__(parent=parent, graph=graph)
        self._got_lane = False
        self.FileList.setOrConnectIfAvailable(FileList)
        self._op_grid = OpGrid(parent=self, FileList=self.FileList)
        self._op_grid_view = OpGridView(parent=self, FileList=self.FileList, Grid=self._op_grid.Grid)
        self.Output.connect(self._op_grid_view.Output)

        self.op_embedding = OpEmbedding(parent=self, FileList=self.FileList, SelectedPlugin=self.SelectedPlugin)
        self.progress_signal = self.op_embedding.progress_signal

        self.embedding_cache = OpValueCache[EmbeddingTable](parent=self)
        self.embedding_cache.Input.connect(self.op_embedding.Embedding)

        self.Embedding.connect(self.embedding_cache.Output)

    def addLane(self, laneIndex: int):
        assert laneIndex == 0
        assert not self._got_lane

        self._got_lane = True

    def removeLane(self, laneIndex: int, finalLength: int):
        assert laneIndex == 0
        assert finalLength == 0
        assert self._got_lane

        self._got_lane = False

    def getLane(self, laneIndex: int):
        assert laneIndex == 0
        assert self._got_lane
        return self

    def _setInSlot(self, slot: InputSlot, subindex: int, roi: "Roi", value: EmbeddingTable):
        if slot == self.EmbeddingCacheInput:
            self.embedding_cache.forceValue(value)
