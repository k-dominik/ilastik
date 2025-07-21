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
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout

from ilastik.utility.gui import silent_qobject
from lazyflow.slot import OutputSlot

from .connectLabels import connect_regions


class LabelExplorer(QDialog):
    positionRequested = pyqtSignal(dict)

    def __init__(self, nonzero_blocks_slot: OutputSlot, label_slot: OutputSlot, parent=None):
        super().__init__(parent)
        self.nonzero_blocks_slot = nonzero_blocks_slot
        self.setupUi()
        self.label_slot = label_slot
        self.axistags = label_slot.meta.getAxisKeys()
        self.populateTable()
        self._labeled_blocks: dict[Tuple[int, ...], Block] = {}

        def _printy(slot, roi, **kwargs):
            print(f"{slot=} -- {roi.start=} {roi.stop}")

        label_slot.notifyDirty(self.populateTable)
        label_slot.notifyDirty(_printy)

        def _sync_viewer_position(currentRow, _currentColumn, _previousRow, _previousColumn):
            position = self.tableWidget.item(currentRow, 0).data(Qt.UserRole)
            self.positionRequested.emit(position)

        self.tableWidget.currentCellChanged.connect(_sync_viewer_position)

    def setupUi(self):
        layout = QVBoxLayout()
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(["position"])
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)

    def populateTable(self, *args, **kwargs):
        non_zero_slicings: List[Tuple[slice, ...]] = self.nonzero_blocks_slot.value

        regions_dict, regions = connect_regions(non_zero_slicings, self.label_slot, self.axistags)

        annotation_anchors = []
        for k, v in regions_dict.items():
            if k == v:
                annotation_anchors.append(regions[k].tagged_center)

        with silent_qobject(self.tableWidget):

            self.tableWidget.setRowCount(len(annotation_anchors))
            for row, roi in enumerate(annotation_anchors):
                roi_center = roi
                position_item = QTableWidgetItem(str(roi_center))
                position_item.setData(Qt.UserRole, roi_center)
                self.tableWidget.setItem(row, 0, position_item)
