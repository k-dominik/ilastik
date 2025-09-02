###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2025, the ilastik developers
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
from functools import partial
from typing import Dict, List, Tuple

import vigra
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QShowEvent
from PyQt5.QtWidgets import QAbstractItemView, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from ilastik.utility.gui import silent_qobject
from lazyflow.request.request import Request, RequestPool
from lazyflow.roi import getIntersectingBlocks, roiToSlice
from lazyflow.slot import OutputSlot
from lazyflow.utility.io_util.write_ome_zarr import SPATIAL_AXES
from lazyflow.utility.timer import timeLogged

from .connectBlockedLabels import Block, Neighbourhood, SpatialAxesKeys, connect_regions, extract_annotations

logger = logging.getLogger(__name__)


class LookupDelegate(QStyledItemDelegate):
    def __init__(self, parent, lookup_func):
        super().__init__(parent)
        self._lookup_fun = lookup_func

    def displayText(self, value, locale):
        return self._lookup_fun(value)


class LabelExplorer(QWidget):
    positionRequested = pyqtSignal(dict)
    closed = pyqtSignal()

    display_text = "Label Explorer"

    def __init__(
        self, nonzero_blocks_slot: OutputSlot, label_slot: OutputSlot, block_shape_slot: OutputSlot, parent=None
    ):
        super().__init__(parent)
        self._lookup_table: dict[str, str] = {}
        self.setWindowTitle("Label Explorer")
        self._block_shape_slot = block_shape_slot
        self._nonzero_blocks_slot = nonzero_blocks_slot
        self._axistags = label_slot.meta.getAxisKeys()
        self._display_axistags = [x for x in self._axistags if x != "c"]
        self._item_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemNeverHasChildren
        self.setupUi()
        self.label_slot = label_slot

        self._block_cache: Dict[Tuple[int, ...], Block] = {}
        self._shown: bool = False

        self.unsubscribe_fns = []
        self.unsubscribe_fns.append(label_slot.notifyDirty(self.update_table))

        def _sync_viewer_position(currentRow, _currentColumn, _previousRow, _previousColumn):
            position = self.tableWidget.item(currentRow, 0).data(Qt.UserRole)
            self.positionRequested.emit(position)

        self.tableWidget.currentCellChanged.connect(_sync_viewer_position)

    def _item_lookup(self, item):
        return self._lookup_table.get(item, "NOT FOUND")

    def showEvent(self, a0: "QShowEvent") -> None:
        super().showEvent(a0)
        self.sync_state()

    def setupUi(self):
        layout = QVBoxLayout()
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(len(self._display_axistags) + 1)
        self.tableWidget.setHorizontalHeaderLabels(self._display_axistags + ["label"])
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.setItemDelegateForColumn(
            len(self._display_axistags), LookupDelegate(self.tableWidget, self._item_lookup)
        )
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)

    def update_table(self, slot, roi, **kwargs):
        """ """
        blockShape = self._block_shape_slot.value
        block_starts = getIntersectingBlocks(blockshape=blockShape, roi=(roi.start, roi.stop))
        block_rois = [(bl, bl + blockShape) for bl in block_starts]
        block_slicings = [roiToSlice(*br) for br in block_rois]
        self.populateTable(block_slicings)

    def update_blocks(self, block_slicings):
        def extract_single(roi):
            labels_data = vigra.taggedView(self.label_slot[roi].wait(), "".join(self._axistags))

            spatial_axes = [SpatialAxesKeys(x) for x in self._axistags if x in SPATIAL_AXES]

            labels_data = labels_data.withAxes("".join(spatial_axes))
            block_regions = extract_annotations(labels_data)
            block = Block(
                axistags="".join(self._axistags), slices=roi, regions=block_regions, neigbourhood=Neighbourhood.SINGLE
            )
            self._block_cache[block.block_start] = block

        # only go through the overhead of a requestpool if there are many blocks to request
        if len(block_slicings) > 1:
            pool = RequestPool()
            for roi in block_slicings:
                pool.add(Request(partial(extract_single, roi)))

            pool.wait()
            pool.clean()
        elif len(block_slicings) == 1:
            extract_single(block_slicings[0])

    def initialize_table(self):
        if not self._shown:
            return
        non_zero_slicings: List[Tuple[slice, ...]] = self._nonzero_blocks_slot.value
        self.populateTable(non_zero_slicings)
        for column in range(self.tableWidget.columnCount() - 1):
            self.tableWidget.resizeColumnToContents(column)

    @timeLogged(logger, logging.INFO, "populateTable")
    def populateTable(self, block_slicings):
        if not self._shown:
            # No need to update the table if not shown
            return
        self.update_blocks(block_slicings)
        self._regions_dict = connect_regions(self._block_cache)
        self.update_table_data()

    def update_table_data(self):
        annotation_anchors: List[Tuple[Dict[str, float], int]] = []
        for k, v in self._regions_dict.items():
            if k == v:
                annotation_anchors.append((k.tagged_center, k.label))

        at_non_c = [x for x in self._axistags if x != "c"]

        with silent_qobject(self.tableWidget):

            self.tableWidget.setRowCount(len(annotation_anchors))
            for row, (roi, label) in enumerate(annotation_anchors):
                roi_center = roi
                for i, at in enumerate(at_non_c):
                    position_item = QTableWidgetItem()
                    position_item.setFlags(self._item_flags)
                    position_item.setData(Qt.DisplayRole, int(roi_center[at]))
                    position_item.setData(Qt.UserRole, roi_center)
                    self.tableWidget.setItem(row, i, position_item)

                label_item = QTableWidgetItem(str(label))
                label_item.setFlags(self._item_flags)
                self.tableWidget.setItem(row, len(at_non_c), label_item)

    def sync_state(self, _a0=None):
        """Update internal "ready" state on gui events

        This widget is shown in a splitter and we want to know if the widget is  currently
        visible or not. If not, we don't need to do any updates.
        """
        shown_before = self._shown
        self._shown = not self.visibleRegion().isEmpty()
        if self._shown and not shown_before:
            self.initialize_table()
        self.tableWidget.viewport().update()

    def cleanup(self):
        for fn in self.unsubscribe_fns:
            fn()
