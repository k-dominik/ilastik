import logging
import os
from dataclasses import fields
from functools import partial
from typing import Any, Callable, Final

from lazyflow.base import ItemId
from lazyflow.operators.opReorderAxes import OpReorderAxes
from lazyflow.utility.grid import ImageGrid

from .ioOperators import OpImageReader
from .opTiffReader import OpTiffReader
from .types import FileList, FileListDataRowNoId

logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt
import vigra

from lazyflow.graph import InputSlot, Operator, OutputSlot
from lazyflow.utility.grid import _OUTPUT_AXIS_KEYS

READER_LOOKUP: Final = {
    ".tiff": OpTiffReader,
    ".tif": OpTiffReader,
    ".png": OpImageReader,
    ".jpeg": OpImageReader,
    ".jpg": OpImageReader,
}


def as_shallow_dict(ds) -> dict[str, Any]:
    return {f.name: getattr(ds, f.name) for f in fields(ds)}


StrSlot = InputSlot[str]
FileListTableOutputSlot = OutputSlot[FileList]


class OpFileListToGridReader(Operator):
    FileList = StrSlot()

    FileListTable = FileListTableOutputSlot()
    Output = OutputSlot()

    # How many pixels around the images to add to the grid
    MARGIN = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._files_table = None
        self._internal_operators = None

    def setupOutputs(self):

        output_axis_keys = "".join(_OUTPUT_AXIS_KEYS)
        file_list_str = self.FileList.value

        file_list = [f for f in file_list_str.split(os.pathsep)]

        # for now we require all the same file types for simplicity, but
        # could be changed to READER_LOOKUP(f.extension)(FilePath=f)

        readers: list[Operator] = [OpImageReader(Filename=f, parent=self) for f in file_list]
        reorders: list[OpReorderAxes] = [
            OpReorderAxes(parent=self, Input=op.Image, AxisOrder=output_axis_keys) for op in readers
        ]
        shapes: list[tuple[int, ...]] = [op.Output.meta.shape for op in reorders]

        max_t = max(x[0] for x in shapes)
        max_x = max(x[1] for x in shapes)
        max_y = max(x[2] for x in shapes)
        max_z = max(x[3] for x in shapes)
        max_c = max(x[4] for x in shapes)

        def _get_data(slot):
            return slot[()].wait()

        data_accessors: list[Callable[[], vigra.VigraArray]] = [partial(_get_data, op.Output) for op in reorders]
        dtypes = [op.Output.meta.dtype for op in reorders]

        self._table: FileList = FileList()

        [
            self._table.add_no_id(
                FileListDataRowNoId(
                    filename=filename, shape=shape, accessor=accessor, dtype=dtype, axistags=output_axis_keys
                )
            )
            for filename, shape, accessor, dtype in zip(file_list, shapes, data_accessors, dtypes)
        ]

        self._grid = ImageGrid(
            list(self._table.keys()),
            grid_size_cell_px=max(max_x, max_y),
            input_axis_keys=_OUTPUT_AXIS_KEYS,
            max_t=max_t,
            max_z=max_z,
            n_c=max_c,
            margin=1,
        )
        dtype = reorders[0].Output.meta.dtype
        self.Output.meta.shape = self._grid.output_shape
        self.Output.meta.dtype = dtype
        self.Output.meta.axistags = vigra.defaultAxistags("".join(self._grid.output_axis_keys))
        self.Output.meta.display_mode = "default"

        self.FileListTable.meta.shape = (1,)
        self.FileListTable.meta.dtype = object

    def execute(self, slot, subindex, roi, result):
        if slot == self.Output:
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
        elif slot == self.FileListTable:
            return [self._table]

    def _load_image(self, item_id: ItemId) -> vigra.VigraArray | None:
        try:
            return vigra.taggedView(self._table[item_id].accessor(), "".join(_OUTPUT_AXIS_KEYS))
        except IndexError:
            print(f"Warning: didn't find image for {item_id=}")
            return None

    def propagateDirty(self, slot, subindex, roi):
        assert slot == self.FileList
        self.Output.setDirty(())
        self.FileListTable.setDirty(())
