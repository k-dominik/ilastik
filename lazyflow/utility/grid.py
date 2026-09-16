###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2026, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#          http://ilastik.org/license/
###############################################################################
from dataclasses import dataclass
from typing import Final, Iterator, Optional, Protocol, Sequence

from lazyflow.base import ItemId
import numpy as np
import numpy.typing as npt
import vigra


from lazyflow.base import Axiskey

Array4D = npt.NDArray[np.floating | np.integer]

_OUTPUT_AXIS_KEYS: Final = ("x", "y", "z", "c")


class RoiLike(Protocol):
    start: tuple[int, ...]
    stop: tuple[int, ...]


@dataclass
class ImageGridCell:
    roi_local_clamped_slice: tuple[slice, ...]
    """slice of the grid cell inside the roi - roi-local coordinates, excluding the margin"""

    image_index: ItemId | None
    """grid cell image index, value at ImageGird._grid_indices[grid_cell]"""

    _output_slice: tuple[slice, slice, slice, slice]
    """slice (xyzc) to extract the visible portion of the ImageGridCell in the roi - cell-local coordinates"""

    _grid_cell_full_shape: tuple[int, int, int, int]

    _margin: int

    def __post_init__(self):
        if any((sl.stop - sl.start) == 0 for sl in self._output_slice):
            raise ValueError("Cannot have 0 sizes")

    def data(
        self,
        dtype: npt.DTypeLike,
        image_data: Optional[vigra.VigraArray] = None,
        fill_value: Optional[int | float] = None,
        additional_margin: int = 0,
    ) -> Array4D:
        """Data that can be written into an array that"""
        grid_data = vigra.taggedView(np.zeros(self._grid_cell_full_shape, dtype=dtype), "".join(_OUTPUT_AXIS_KEYS))
        if fill_value is not None:
            m = self._margin + additional_margin
            grid_data[m:-m, m:-m, :, :] = fill_value

        sx, sy, sz, sc = self._grid_cell_full_shape

        if image_data is not None:
            image_data4d = image_data.withAxes(_OUTPUT_AXIS_KEYS)
            ix, iy, iz, ic = image_data4d.shape
            assert sc == ic, f"Expect Image {ic=} to have the same number of channels as grid {sc=}"
            offset_data_x = (sx - ix) // 2
            offset_data_y = (sy - iy) // 2
            offset_data_z = (sz - iz) // 2

            offset_image_x = (ix - sx) // 2
            offset_image_y = (iy - sy) // 2
            offset_image_z = (iz - sz) // 2

            grid_data[
                (
                    slice(max(0, offset_data_x), min(sx, ix + offset_data_x)),
                    slice(max(0, offset_data_y), min(sy, iy + offset_data_y)),
                    slice(max(0, offset_data_z), min(sz, iz + offset_data_z)),
                    slice(None),
                )
            ] = image_data4d[
                (
                    slice(max(0, offset_image_x), min(sx + offset_image_x, ix)),
                    slice(max(0, offset_image_y), min(sy + offset_image_y, iy)),
                    slice(max(0, offset_image_z), min(sz + offset_image_z, iz)),
                    slice(None),
                )
            ]

        return grid_data[self._output_slice]


@dataclass
class _GridCoords:
    x: int
    y: int


class ImageGrid:

    def __init__(
        self,
        grid_indices: Sequence[ItemId],
        grid_size_cell_px: int,
        input_axis_keys: Sequence[Axiskey],
        max_z: int,
        n_c: int,
        margin: int = 1,
    ):
        """Represents a square grid of images in x-y plane

        grid_indices = (i0, i1, i2, ...)

        +---+---+---+---+
        |i0 |i1 |i2 |i3 |
        +---+---+---+---+
        |i4 |   |   |   |
        +---+---+---+---+
        |   |i9 |   |   |
        +---+---+---+---+
        |   |   |i14|i15|
        +---+---+---+---+

        if there are less grid indices those ImageGridCells will have `None` as image_index


        Args:
            grid_indices: Sequence of indices that are the interface to find the
              image to display in a certain grid cell
            grid_size_cell_px: Maximum dimension in x-y to fit into the grid
            input_axis_keys: concatenated axis keys
            max_z: Maximum size along z axis
            n_c: Number of channels
            margin: Number of pixels around the grid_cell_size_px
        """
        self._grid_indices = tuple(grid_indices)
        self.n_objs = len(self._grid_indices)
        self._grid_width = int(np.ceil(np.sqrt(self.n_objs)))
        self._margin = margin
        self.grid_size_cell_px = int(grid_size_cell_px) + 2 * self._margin

        shape = (
            self.grid_size_cell_px * self._grid_width,
            self.grid_size_cell_px * self._grid_width,
            int(max_z),
            int(n_c),
        )
        self.input_axis_keys = tuple(input_axis_keys)
        self.output_shape = shape
        self._cell_shape = (self.grid_size_cell_px, self.grid_size_cell_px, int(max_z), int(n_c))
        self.output_axis_keys = _OUTPUT_AXIS_KEYS

    def grid_cells_per_roi(self, roi: RoiLike) -> Iterator[ImageGridCell]:
        x_start, y_start, z_start, c_start = roi.start
        x_stop, y_stop, z_stop, c_stop = roi.stop

        roi_start = _GridCoords(x_start, y_start)
        roi_stop = _GridCoords(x_stop, y_stop)

        grid_full_start = _GridCoords(
            int(np.floor(roi_start.x / self.grid_size_cell_px)), int(np.floor(roi_start.y / self.grid_size_cell_px))
        )
        grid_full_stop = _GridCoords(
            int(np.ceil(roi_stop.x / self.grid_size_cell_px)), int(np.ceil(roi_stop.y / self.grid_size_cell_px))
        )

        # move to 0 index for np.ndindex
        grids = [grid_full_stop.y - grid_full_start.y, grid_full_stop.x - grid_full_start.x]
        # and add start again after swapping the order to x - y again
        requested_grid_local: npt.NDArray[np.integer] = np.array(list(np.ndindex(*grids)))[:, [1, 0]] + np.array(
            [grid_full_start.x, grid_full_start.y]
        )

        requested_grid_start_coords_local = requested_grid_local * self.grid_size_cell_px
        # check start aligned
        for start_coords_px, start_coords_grid in zip(requested_grid_start_coords_local, requested_grid_local):
            start_px_x, start_px_y = start_coords_px
            start_grid_x, start_grid_y = start_coords_grid

            end_coords_clamped_roi_local = _GridCoords(
                min(start_px_x + self.grid_size_cell_px, roi_stop.x) - roi_start.x,
                min(start_px_y + self.grid_size_cell_px, roi_stop.y) - roi_start.y,
            )

            start_coords_clamped_roi_local = _GridCoords(
                max(start_px_x, roi_start.x) - roi_start.x, max(start_px_y, roi_start.y) - roi_start.y
            )

            start_offset_in_grid_cell = _GridCoords(
                start_coords_clamped_roi_local.x + roi_start.x - start_px_x,
                start_coords_clamped_roi_local.y + roi_start.y - start_px_y,
            )

            flat_index = start_grid_x + start_grid_y * self._grid_width
            image_index = self._grid_indices[flat_index] if flat_index < self.n_objs else None

            if start_coords_clamped_roi_local.x >= end_coords_clamped_roi_local.x:
                raise ValueError()

            if start_coords_clamped_roi_local.y >= end_coords_clamped_roi_local.y:
                raise ValueError()

            roi_local_clamped_slice = (
                slice(start_coords_clamped_roi_local.x, end_coords_clamped_roi_local.x),
                slice(start_coords_clamped_roi_local.y, end_coords_clamped_roi_local.y),
                slice(z_start, z_stop),
                slice(c_start, c_stop),
            )

            output_slice = (
                slice(
                    start_offset_in_grid_cell.x,
                    start_offset_in_grid_cell.x + end_coords_clamped_roi_local.x - start_coords_clamped_roi_local.x,
                ),
                slice(
                    start_offset_in_grid_cell.y,
                    start_offset_in_grid_cell.y + end_coords_clamped_roi_local.y - start_coords_clamped_roi_local.y,
                ),
                slice(z_start, z_stop),
                slice(c_start, c_stop),
            )

            yield ImageGridCell(
                roi_local_clamped_slice=roi_local_clamped_slice,
                image_index=ItemId(image_index) if image_index is not None else image_index,
                _output_slice=output_slice,
                _grid_cell_full_shape=self._cell_shape,
                _margin=self._margin,
            )

    def __repr__(self) -> str:
        return f"Grid with {self.n_objs} objects - {self.output_shape=} {self._cell_shape=}"
