import logging
from dataclasses import fields
from typing import Any, Final, Optional

from lazyflow.base import ItemId
from lazyflow.slot import Slot
from lazyflow.utility.grid import ImageGrid

from .ioOperators import OpImageReader
from .opTiffReader import OpTiffReader
from .types import FileList as FileListT

logger = logging.getLogger(__name__)

import numpy as np
import vigra

from lazyflow.graph import InputSlot, Operator, OutputSlot
from lazyflow.utility.grid import OUTPUT_AXIS_KEYS

READER_LOOKUP: Final = {
    ".tiff": OpTiffReader,
    ".tif": OpTiffReader,
    ".png": OpImageReader,
    ".jpeg": OpImageReader,
    ".jpg": OpImageReader,
}


def as_shallow_dict(ds) -> dict[str, Any]:
    return {f.name: getattr(ds, f.name) for f in fields(ds)}


def determine_grid_shape(file_list: FileListT) -> tuple[ImageGrid, np.dtype, tuple[int, ...], vigra.AxisTags]:
    shapes: list[tuple[int, ...]] = [f.shape for f in file_list.values()]

    max_t = max(x[0] for x in shapes)
    max_x = max(x[1] for x in shapes)
    max_y = max(x[2] for x in shapes)
    max_z = max(x[3] for x in shapes)
    max_c = max(x[4] for x in shapes)

    grid = ImageGrid(
        list(file_list.keys()),
        grid_size_cell_px=max(max_x, max_y),
        input_axis_keys=OUTPUT_AXIS_KEYS,
        max_t=max_t,
        max_z=max_z,
        n_c=max_c,
        margin=1,
    )
    dtype = next(iter(file_list.values())).dtype
    shape = grid.output_shape
    axistags = vigra.defaultAxistags("".join(grid.output_axis_keys))

    return (grid, dtype, shape, axistags)


class OpFileListToGridReader(Operator):
    FileList = InputSlot[FileListT]()

    Output = OutputSlot()

    # How many pixels around the images to add to the grid
    MARGIN = 1

    def __init__(self, FileList: Optional[Slot[FileListT]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FileList.setOrConnectIfAvailable(FileList)

    def setupOutputs(self):
        file_list = self.FileList.value
        grid, dtype, shape, axistags = determine_grid_shape(file_list)
        self._grid = grid

        self.Output.meta.shape = shape
        self.Output.meta.dtype = dtype.type
        self.Output.meta.axistags = axistags
        self.Output.meta.display_mode = "default"

    def execute(self, slot, subindex, roi, result):
        if slot == self.Output:
            block = np.zeros_like(result)
            tagged_start = dict(zip(self.Output.meta.axistags.keys(), roi.start))
            tagged_stop = dict(zip(self.Output.meta.axistags.keys(), roi.stop))
            c_slice = slice(tagged_start["c"], tagged_stop["c"])
            slicing = tuple([slice(None) if k != "c" else c_slice for k in tagged_start.keys()])
            for grid_cell in self._grid.grid_cells_per_roi(roi):
                if grid_cell.image_index is None:
                    continue
                image = self._load_image(grid_cell.image_index)

                if image is None:
                    continue

                data = grid_cell.data(dtype=result.dtype, image_data=image[slicing])
                block[grid_cell.roi_local_clamped_slice] = data

            result[:] = block

    def _load_image(self, item_id: ItemId) -> vigra.VigraArray | None:
        table = self.FileList.value
        try:
            return vigra.taggedView(table[item_id].accessor(), "".join(OUTPUT_AXIS_KEYS))
        except IndexError:
            print(f"Warning: didn't find image for {item_id=}")
            return None

    def propagateDirty(self, slot, subindex, roi):
        assert slot == self.FileList
        self.Output.setDirty(())
