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
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Literal, Optional, Tuple, Union

import vigra
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout
from vigra.analysis import extractRegionFeatures

from ilastik.utility.gui import silent_qobject
from lazyflow.slot import OutputSlot
from lazyflow.utility.io_util.write_ome_zarr import SPATIAL_AXES


class Neighbourhood(IntEnum):
    NONE = 0
    SINGLE = 1  # Axis aligned, 2D: 4, 3D: 6
    NDIM = 2  # Full block, 2D: 8, 3D: 26


class BlockBoundary(IntEnum):
    NONE = 0
    START = 1
    STOP = 2


BoundaryDescrRelative = dict[Literal["x", "y", "z"], BlockBoundary]
BoundaryDescr = dict[Literal["x", "y", "z"], Union[int, None]]


@dataclass
class Region:
    axistags: str
    slices: tuple[slice]
    label: int

    @property
    def tagged_slicing(self):
        return dict(zip(self.axistags, self.slices))

    @property
    def tagged_center(self):
        return {k: sl.stop - sl.start for k, sl in self.tagged_slicing.items()}

    def is_at_boundary(self, boundary: BoundaryDescr) -> bool:
        if all(b is None for b in boundary.values()):
            return False

        is_at_boundary = True
        for k, coord in boundary.items():
            if coord is not None:
                sl = self.tagged_slicing[k]
                if sl.start == coord or sl.stop == coord:
                    continue
                else:
                    return False

        return is_at_boundary


@dataclass
class Block:
    axistags: str
    slices: tuple[slice, ...]
    n_dim: Literal[2, 3]
    regions: List[Region]
    neigbourhood: Neighbourhood = Neighbourhood.NDIM

    @property
    def tagged_slices(self):
        return {tag: sl for tag, sl in zip(self.axistags, self.slices)}

    @property
    def spatial_axes(self):
        return [x for x in self.axistags if x in SPATIAL_AXES]

    def boundary_regions(self, boundary: BoundaryDescrRelative, label: Optional[int] = None):
        if self.neigbourhood == Neighbourhood.NONE:
            return

        def boundary_index_from_slice(sl: slice, boundary: BlockBoundary) -> Union[int, None]:
            if boundary == BlockBoundary.NONE:
                return None
            if boundary == BlockBoundary.START:
                return 0
            if boundary == BlockBoundary.STOP:
                return sl.stop - sl.start

        tagged_boundary = {}
        for at, bd in boundary.items():
            tagged_boundary[at] = boundary_index_from_slice(self.tagged_slices[at], bd)

        def labelmatch(region_label) -> bool:
            if label == None:
                return True
            else:
                return region_label == label

        for region in self.regions:
            if region.is_at_boundary(tagged_boundary) and labelmatch(region.label):
                yield region


class LabelExplorer(QDialog):
    positionRequested = pyqtSignal(dict)

    def __init__(self, nonzero_blocks_slot: OutputSlot, label_slot: OutputSlot, parent=None):
        super().__init__(parent)
        self.nonzero_blocks_slot = nonzero_blocks_slot
        self.setupUi()
        self.label_slot = label_slot
        self.axistags = label_slot.meta.getAxisKeys()
        self.populateTable()

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
        non_zero_slicings = self.nonzero_blocks_slot.value

        annotation_centers = []
        for roi in non_zero_slicings:
            annotation_centers.extend(self.extract_annotations(roi))

        with silent_qobject(self.tableWidget):

            self.tableWidget.setRowCount(len(annotation_centers))
            for row, roi in enumerate(annotation_centers):
                roi_center = roi
                position_item = QTableWidgetItem(str(roi_center))
                position_item.setData(Qt.UserRole, roi_center)
                self.tableWidget.setItem(row, 0, position_item)

    def extract_annotations(self, roi):
        tagged_roi = dict(zip(self.axistags, roi))
        labels_data = vigra.taggedView(self.label_slot[roi].wait(), "".join(self.axistags))
        if "z" in self.axistags:
            connected_components = vigra.analysis.labelVolumeWithBackground(
                labels_data.astype("uint32"),
            )
        else:
            connected_components = vigra.analysis.labelImageWithBackground(
                labels_data.astype("uint32"),
            )
        feats = extractRegionFeatures(
            labels_data.astype("float32"), connected_components, ignoreLabel=0, features=["RegionCenter"]
        )
        centers = feats["RegionCenter"].astype("uint32") + [
            tagged_roi[x].start for x in self.axistags if x in SPATIAL_AXES
        ]
        return [dict(zip(self.axistags, center)) for center in centers[1::]]
