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
from typing import Protocol

from lazyflow.operators.ioOperators.types import ImageMeta
from lazyflow.roi import TinyVector, roiToSlice


import numpy
import vigra


class RoiP(Protocol):
    start: TinyVector[int]
    stop: TinyVector[int]


@dataclass
class SimpleRoi:
    start: TinyVector[int]
    stop: TinyVector[int]


def determine_metadata(filename: str) -> ImageMeta:
    info = vigra.impex.ImageInfo(filename)
    assert [tag.key for tag in info.getAxisTags()] == ["x", "y", "c"]

    shape_xyc = info.getShape()
    shape_yxc = (shape_xyc[1], shape_xyc[0], shape_xyc[2])

    dtype = info.getDtype()

    numImages = vigra.impex.numberImages(filename)

    if numImages == 1:
        # For 2D, we use order yxc.
        shape = shape_yxc
        v_tags = info.getAxisTags()
        axistags = vigra.AxisTags([v_tags[k] for k in "yxc"])
    else:
        shape_zyxc = (numImages,) + shape_yxc
        shape = shape_zyxc

        # Insert z tag
        z_tag = vigra.defaultAxistags("z")[0]
        tags_xyc = [tag for tag in info.getAxisTags()]
        tags_zyxc = [z_tag] + list(reversed(tags_xyc[:-1])) + tags_xyc[-1:]
        axistags = vigra.AxisTags(tags_zyxc)

    return ImageMeta(filename=filename, dtype=dtype, shape=shape, axistags=axistags)


def read_roi(meta: ImageMeta, roi: RoiP, result=None) -> vigra.VigraArray:
    if result is None:
        shape = roi.stop - roi.start
        result = numpy.ndarray(shape, dtype=meta.dtype)

    if len(roi.start) == 4:
        # Copy from each image slice into the corresponding slice of the result.
        roi_zyxc = numpy.array([roi.start, roi.stop])
        for z_global, z_result in zip(list(range(*roi_zyxc[:, 0])), list(range(result.shape[0]))):
            full_slice = vigra.impex.readImage(meta.filename, index=z_global)
            full_slice = full_slice.transpose(1, 0, 2)  # xyc -> yxc
            assert full_slice.shape == meta.shape[1:]
            result[z_result] = full_slice[roiToSlice(*roi_zyxc[:, 1:])]
    elif len(roi.start) == 3:
        full_slice = vigra.impex.readImage(meta.filename).transpose(1, 0, 2)  # xyc -> yxc
        assert full_slice.shape == meta.shape
        roi_yxc = numpy.array([roi.start, roi.stop])
        result[:] = full_slice[roiToSlice(*roi_yxc)]
    return result


def read(meta: ImageMeta) -> vigra.VigraArray:
    shape = meta.shape
    full_roi = SimpleRoi(start=TinyVector(0 for _ in shape), stop=TinyVector(shape))
    return vigra.taggedView(read_roi(meta=meta, roi=full_roi), axistags=meta.axistags)
