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
from typing import Callable

import vigra


def reordered(arr_accessor: Callable[[], vigra.VigraArray], axisorder: str) -> vigra.VigraArray:
    arr = arr_accessor()
    return arr.withAxes(axisorder)


def reordered_shape(shape: tuple[int, ...], tags: list[str] | str, axisorder: str) -> tuple[int, ...]:
    assert len(shape) == len(tags)
    tagged_shape = {k: v for k, v in zip(tags, shape)}
    assert all(tagged_shape[k] == 1 for k in set(tags).difference(axisorder))
    return tuple(tagged_shape.get(ax, 1) for ax in axisorder)
