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
from collections import UserDict
from typing import Annotated, Callable, Sequence
from functools import cached_property, partial, lru_cache

import numpy.typing as npt
import numpy
import vigra
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, TypeAdapter, ValidateAs
from pydantic.dataclasses import dataclass

from lazyflow.base import ItemId
from lazyflow.utility.helpers import TrueInc
from lazyflow.operators.functional.reorderAxes import reordered, reordered_shape
from .functional import imageReader as f_imageReader


def _at_to_json(at: vigra.AxisTags) -> str:
    return at.toJSON()


def _ensure_at(at: vigra.AxisTags | str) -> str:
    if isinstance(at, vigra.AxisTags):
        return at

    return vigra.AxisTags.fromJSON(at)


def _ensure_dtype(dtype: npt.DTypeLike) -> numpy.dtype:
    return numpy.dtype(dtype)


def _dtype_name(dtype: npt.DTypeLike) -> str:
    return _ensure_dtype(dtype).name


VigraAxisTags = Annotated[vigra.AxisTags, BeforeValidator(_ensure_at), PlainSerializer(_at_to_json)]


class RootModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RowBase(RootModel):
    id: ItemId


class ImageMeta(RootModel):
    filename: str
    dtype: Annotated[numpy.dtype, BeforeValidator(_ensure_dtype), PlainSerializer(_dtype_name)]
    shape: tuple[int, ...]
    axistags: VigraAxisTags


class ImageMetaR(ImageMeta):
    original_axistags: VigraAxisTags
    original_shape: tuple[int, ...]

    @cached_property
    def accessor(self) -> Callable[[], vigra.VigraArray]:
        meta = ImageMeta(
            filename=self.filename, dtype=self.dtype, shape=self.original_shape, axistags=self.original_axistags
        )
        return partial(reordered, arr_accessor=partial(f_imageReader.read, meta), axisorder=self.axistags)

    @classmethod
    def from_image_meta_with_axes(cls, meta: ImageMeta, axisorder: Sequence[str] | str) -> "ImageMetaR":
        new_shape = reordered_shape(meta.shape, meta.axistags.keys(), axisorder)
        return cls(
            filename=meta.filename,
            dtype=meta.dtype,
            shape=new_shape,
            original_shape=meta.shape,
            axistags=vigra.defaultAxistags("".join(axisorder)),
            original_axistags=meta.axistags,
        )


FileListDataRowNoId = ImageMetaR


class FileListDataRow(FileListDataRowNoId, RowBase):
    pass


class FileList(UserDict[ItemId, FileListDataRow]):
    def __init__(self, dict=None, /, init_count: int = 0, **kwargs):
        super().__init__(dict, **kwargs)

        self._counter = TrueInc(init_count)

    def __setitem__(self, key: ItemId, item: FileListDataRow) -> None:
        assert item.id == key, "Cannot assign mismatching key and item."
        super().__setitem__(key, item)

    def add(self, item: FileListDataRow) -> None:
        self._counter.ensure_ceil(item.id)
        self[item.id] = item

    def add_no_id(self, item: FileListDataRowNoId) -> None:
        new_id = self._counter.inc()
        self[new_id] = FileListDataRow(
            id=new_id,
            filename=item.filename,
            shape=item.shape,
            original_shape=item.original_shape,
            dtype=item.dtype,
            axistags=item.axistags,
            original_axistags=item.original_axistags,
        )

    def state(self):
        return self._counter.state


class _FileListS(BaseModel):
    state: int
    rows: list[FileListDataRow]


def to_pydantic(f: FileList) -> _FileListS:
    return _FileListS.model_validate(dict(state=f.state(), rows=list(f.values())))


def from_pydantic(f: _FileListS) -> FileList:
    return FileList(((row.id, row) for row in f.rows), f.state)


FileListSerializer = Annotated[
    FileList,
    ValidateAs(_FileListS, from_pydantic),
    PlainSerializer(to_pydantic),
]
