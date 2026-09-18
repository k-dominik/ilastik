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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import vigra

from lazyflow.base import ItemId
from lazyflow.operator import Operator
from lazyflow.operators.ioOperators.opFileListToGridReader import FileListTableOutputSlot
from lazyflow.operators.ioOperators.types import FileList, FileListDataRow
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.utility.grid import _OUTPUT_AXIS_KEYS, ImageGrid

FileListInputSlot = InputSlot[FileList]
GridInputSlot = InputSlot[ImageGrid]
GridOutputSlot = OutputSlot[ImageGrid]
ObjectIdsInputslot = InputSlot[list[ItemId]]


@dataclass
class SimpleRoi:
    start: tuple[int, ...]
    stop: tuple[int, ...]


class OpGrid(Operator):
    FileList = FileListInputSlot()
    SubsetObjects = ObjectIdsInputslot(optional=True)

    Grid = GridOutputSlot(stype="object")

    MARGIN = 1

    def __init__(
        self,
        graph=None,
        parent=None,
        FileList=None,
        SubsetObjects=None,
        all_if_no_subselection: bool = True,
    ):
        super().__init__(graph=graph, parent=parent)
        self.FileList.setOrConnectIfAvailable(FileList)
        self.SubsetObjects.setOrConnectIfAvailable(SubsetObjects)
        self._all_if_no_subselection = all_if_no_subselection

    def setupOutputs(self):
        self._table = dict(self.FileList.value)
        sub_selection: list[ItemId] = []
        if self.SubsetObjects.ready():
            sub_selection = self.SubsetObjects.value
            print("Applyting sub selection", sub_selection)
            if len(sub_selection) > 0:
                self._table = {id: self._table[id] for id in sub_selection}

        if not sub_selection and not self._all_if_no_subselection:
            # empty sub_selection - empty grid
            self._table = {}

        # x, y, z, c
        shapes: list[tuple[int, ...]] = [item.shape for item in self._table.values()]

        if shapes:
            max_t = max(x[0] for x in shapes)
            max_x = max(x[1] for x in shapes)
            max_y = max(x[2] for x in shapes)
            max_z = max(x[3] for x in shapes)
            max_c = max(x[4] for x in shapes)
        else:
            max_t, max_x, max_y, max_z, max_c = 1, 0, 0, 1, 1

        self._grid = ImageGrid(
            list(self._table.keys()),
            grid_size_cell_px=max(max_x, max_y),
            input_axis_keys=_OUTPUT_AXIS_KEYS,
            max_t=max_t,
            max_z=max_z,
            n_c=max_c,
            margin=1,
        )

        self.Grid.meta.dtype = object
        self.Grid.meta.shape = (1,)

        print(f"Grid: {self._grid.output_shape} {self._grid.output_axis_keys}")

    def execute(self, slot, subindex, roi, result):
        if slot == self.Grid:
            return [self._grid]

        raise ValueError(f"Wrong slot {slot.name} for this operator")

    def propagateDirty(self, slot, subindex, roi):
        print("setting grid dirty")
        if slot == self.FileList:
            self.Grid.setDirty(())
        if slot == self.SubsetObjects:
            self.setupOutputs()
            self.Grid.setDirty(())

    def object_id_at(self, coordinate5d):
        tagged_coord = {ax: coord for ax, coord in zip("txyzc", coordinate5d)}
        tagged_start = {ax: tagged_coord[ax] for ax in self._grid.output_axis_keys}

        x = list(
            self._grid.grid_cells_per_roi(
                SimpleRoi(
                    start=tuple([tagged_start[x] for x in self._grid.output_axis_keys]),
                    stop=tuple([tagged_start[x] + 1 for x in self._grid.output_axis_keys]),
                )
            )
        )
        assert len(x) == 1
        return ItemId(x[0].image_index)


class OpGridView(Operator):
    FileList = FileListInputSlot()
    Grid = GridInputSlot(stype="object")
    Output = OutputSlot()

    def __init__(
        self,
        parent=None,
        graph=None,
        FileList: Optional[FileListInputSlot | FileListTableOutputSlot] = None,
        Grid: Optional[GridInputSlot | GridOutputSlot] = None,
    ):
        super().__init__(parent=parent, graph=graph)
        self.FileList.setOrConnectIfAvailable(FileList)
        self.Grid.setOrConnectIfAvailable(Grid)
        self._table: dict[ItemId, FileListDataRow] = {}
        self._grid: ImageGrid | None = None

    def setupOutputs(self):
        self._table = dict(self.FileList.value)
        assert not self._table or all(
            x.dtype == next(iter(self._table.values())).dtype for x in self._table.values()
        ), "All dtypes are required to be the same"
        # @output_axiskeys = "".join(_OUTPUT_AXIS_KEYS)
        # assert not self._table or all(
        #     x. == output_axiskeys for x in self._table
        # ), f"All output axis keys must be {_OUTPUT_AXIS_KEYS}"

        self._grid = self.Grid.value
        self.Output.meta.shape = self._grid.output_shape
        self.Output.meta.axistags = vigra.defaultAxistags("".join(self._grid.output_axis_keys))
        self.Output.meta.display_mode = "default"
        self.Output.meta.dtype = "uint8" if not self._table else next(iter(self._table.values())).dtype

    def execute(self, slot, subindex, roi, result):
        if slot == self.Output:
            assert self._grid
            block = np.zeros_like(result)
            min_val = None
            for grid_cell in self._grid.grid_cells_per_roi(roi):
                if grid_cell.image_index is None:
                    continue
                image = self._load_image(grid_cell.image_index)

                if image is None:
                    continue

                data = grid_cell.data(dtype=result.dtype, image_data=image)
                block[grid_cell.roi_local_clamped_slice] = data
                if min_val is None:
                    min_val = image.min()
                else:
                    min_val = image.min() if image.min() < min_val else min_val

            # TODO: find a better way to deal with the background
            if min_val is not None:
                block[block == 0] = min_val
            result[:] = block

    def _load_image(self, item_id: ItemId) -> vigra.VigraArray | None:
        try:
            return vigra.taggedView(self._table[item_id].accessor(), "".join(_OUTPUT_AXIS_KEYS))
        except IndexError:
            print(f"Warning: didn't find image for {item_id=}")
            return None

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty()
