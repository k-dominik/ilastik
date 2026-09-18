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
import numpy as np

from lazyflow.utility.grid import ImageGrid, ImageGridCell

import vigra


class Roi:
    def __init__(self, start, stop):
        self.start: tuple[int, ...] = start
        self.stop: tuple[int, ...] = stop


def test_grid_empty():
    g = ImageGrid(grid_indices=[], grid_size_cell_px=10, input_axis_keys=["x", "y"], max_t=1, max_z=1, n_c=1)
    assert g.n_objs == 0
    assert g.output_shape == (1, 0, 0, 1, 1)


def test_grid_2x2_full():
    g = ImageGrid(
        grid_indices=[10, 12, 15, 17], grid_size_cell_px=10, input_axis_keys=["x", "y"], max_t=1, max_z=1, n_c=1
    )
    assert g.n_objs == 4
    assert g.output_shape == (1, 24, 24, 1, 1)


def test_grid_2x2_partial():
    g = ImageGrid(grid_indices=[10, 12, 15], grid_size_cell_px=10, input_axis_keys=["x", "y"], max_t=1, max_z=1, n_c=1)
    assert g.n_objs == 3
    assert g.output_shape == (1, 24, 24, 1, 1)


def test_full_grid_2x2_grid_cell_aligned():
    g = ImageGrid(
        grid_indices=[10, 12, 15, 17], grid_size_cell_px=10, input_axis_keys=["x", "y"], max_t=1, max_z=1, n_c=1
    )

    roi = Roi((0, 0, 0, 0, 0), (1, 12, 12, 1, 1))

    grid_cells = list(g.grid_cells_per_roi(roi))
    assert len(grid_cells) == 1

    grid_cell = grid_cells[0]

    assert grid_cell.roi_local_clamped_slice == (slice(0, 1), slice(0, 12), slice(0, 12), slice(0, 1), slice(0, 1))
    assert grid_cell.image_index == 10
    assert grid_cell._output_slice == (slice(0, 1), slice(0, 12), slice(0, 12), slice(0, 1), slice(0, 1))


def test_full_grid_2x2_grid_cell_non_aligned():
    """Worst case: going into the neighboring cells just one pixel, with offset at start, too"""
    g = ImageGrid(
        grid_indices=[10, 12, 15, 17], grid_size_cell_px=10, input_axis_keys=["x", "y"], max_t=1, max_z=1, n_c=1
    )

    roi = Roi((0, 1, 1, 0, 0), (1, 13, 13, 1, 1))

    grid_cells = list(g.grid_cells_per_roi(roi))
    assert len(grid_cells) == 4

    grid_cell_0 = grid_cells[0]
    assert grid_cell_0.roi_local_clamped_slice == (slice(0, 1), slice(0, 11), slice(0, 11), slice(0, 1), slice(0, 1))
    assert grid_cell_0.image_index == 10
    assert grid_cell_0._output_slice == (slice(0, 1), slice(1, 12), slice(1, 12), slice(0, 1), slice(0, 1))

    grid_cell_1 = grid_cells[1]
    assert grid_cell_1.roi_local_clamped_slice == (slice(0, 1), slice(11, 12), slice(0, 11), slice(0, 1), slice(0, 1))
    assert grid_cell_1.image_index == 12
    assert grid_cell_1._output_slice == (slice(0, 1), slice(0, 1), slice(1, 12), slice(0, 1), slice(0, 1))

    grid_cell_2 = grid_cells[2]
    assert grid_cell_2.roi_local_clamped_slice == (slice(0, 1), slice(0, 11), slice(11, 12), slice(0, 1), slice(0, 1))
    assert grid_cell_2.image_index == 15
    assert grid_cell_2._output_slice == (slice(0, 1), slice(1, 12), slice(0, 1), slice(0, 1), slice(0, 1))

    grid_cell_3 = grid_cells[3]
    assert grid_cell_3.roi_local_clamped_slice == (slice(0, 1), slice(11, 12), slice(11, 12), slice(0, 1), slice(0, 1))
    assert grid_cell_3.image_index == 17
    assert grid_cell_3._output_slice == (slice(0, 1), slice(0, 1), slice(0, 1), slice(0, 1), slice(0, 1))


def test_full_grid_3x3_grid_cell_non_aligned():
    """
       0   12  24  36
     0 +---+---+---+
       |10 |12 |15 |
    12 +---+---+---+
       |17 |18 |19 |
    24 +---+---+---+
       |21 |   |   |
    36 +---+---+---+

    """
    g = ImageGrid(
        grid_indices=[10, 12, 15, 17, 18, 19, 21],
        grid_size_cell_px=10,
        input_axis_keys=["x", "y"],
        max_t=1,
        max_z=1,
        n_c=1,
    )

    roi = Roi((0, 15, 13, 0, 0), (1, 28, 29, 1, 1))

    grid_cells = list(g.grid_cells_per_roi(roi))
    assert len(grid_cells) == 4

    grid_cell_0 = grid_cells[0]
    assert grid_cell_0.roi_local_clamped_slice == (slice(0, 1), slice(0, 9), slice(0, 11), slice(0, 1), slice(0, 1))
    assert grid_cell_0.image_index == 18
    assert grid_cell_0._output_slice == (slice(0, 1), slice(3, 12), slice(1, 12), slice(0, 1), slice(0, 1))

    grid_cell_1 = grid_cells[1]
    assert grid_cell_1.roi_local_clamped_slice == (slice(0, 1), slice(9, 13), slice(0, 11), slice(0, 1), slice(0, 1))
    assert grid_cell_1.image_index == 19
    assert grid_cell_1._output_slice == (slice(0, 1), slice(0, 4), slice(1, 12), slice(0, 1), slice(0, 1))

    grid_cell_2 = grid_cells[2]
    assert grid_cell_2.roi_local_clamped_slice == (slice(0, 1), slice(0, 9), slice(11, 16), slice(0, 1), slice(0, 1))
    assert grid_cell_2.image_index == None
    assert grid_cell_2._output_slice == (slice(0, 1), slice(3, 12), slice(0, 5), slice(0, 1), slice(0, 1))

    grid_cell_3 = grid_cells[3]
    assert grid_cell_3.roi_local_clamped_slice == (slice(0, 1), slice(9, 13), slice(11, 16), slice(0, 1), slice(0, 1))
    assert grid_cell_3.image_index == None
    assert grid_cell_3._output_slice == (slice(0, 1), slice(0, 4), slice(0, 5), slice(0, 1), slice(0, 1))


def test_image_grid_cell_fill_value():
    c = ImageGridCell(
        roi_local_clamped_slice=(slice(0, 1), slice(10, 20), slice(20, 30), slice(0, 5), slice(0, 3)),
        image_index=None,
        _output_slice=(slice(0, 1), slice(0, 10), slice(0, 10), slice(0, 5), slice(0, 3)),
        _grid_cell_full_shape=(1, 10, 10, 5, 3),
        _margin=1,
    )

    data = c.data(np.uint8, fill_value=3)

    assert data.shape == (1, 10, 10, 5, 3)
    np.testing.assert_array_equal(np.array(data)[:, 1:-1, 1:-1, :, :], 3)


def test_imaeg_grid_cell_image_centering():
    """
                 4        8
          1               8
    x: 0 ---------------------------

               3            9
             2          7
    y: 0 ---------------------------

    """

    c = ImageGridCell(
        roi_local_clamped_slice=(slice(0, 1), slice(14, 18), slice(23, 29), slice(0, 5), slice(0, 3)),
        image_index=None,
        _output_slice=(slice(0, 1), slice(4, 8), slice(3, 9), slice(0, 5), slice(0, 3)),
        _grid_cell_full_shape=(1, 10, 10, 5, 3),
        _margin=1,
    )

    data = np.random.randint(0, 256, 7 * 5 * 3, dtype="uint8").reshape(7, 5, 3)
    image = vigra.taggedView(data, axistags="xyc")

    data = c.data(np.uint8, image_data=image)

    assert data.shape == (1, 4, 6, 5, 3)
    np.testing.assert_array_equal(data[:, :, 0:4, 2:3, :], image.withAxes("txyzc")[:, 3:7, 1:5, :, :])
