from collections import UserDict
from typing import Callable

import numpy.typing as npt
import vigra
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from lazyflow.base import ItemId
from lazyflow.utility.helpers import TrueInc


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RowBase:
    id: ItemId


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class FileListDataRowNoId:
    filename: str
    shape: tuple[int, ...]
    accessor: Callable[[], vigra.VigraArray]
    dtype: npt.DTypeLike
    axistags: str


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
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
            accessor=item.accessor,
            dtype=item.dtype,
            axistags=item.axistags,
        )
