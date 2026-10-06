from collections import UserDict
from typing import Annotated, Callable

import numpy.typing as npt
import vigra
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, ValidateAs
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
    dtype: npt.DTypeLike
    axistags: str

    accessor: Callable[[], vigra.VigraArray] = Field(exclude=True)


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

    def state(self):
        return self._counter.state


class _FileListS(BaseModel):
    state: int
    rows: list[FileListDataRow]


def to_pydantic(f: FileList) -> _FileListS:
    return _FileListS.model_validate(dict(state=f.state, keys=list(f.keys()), rows=list(f.values())))


def from_pydantic(f: _FileListS) -> FileList:
    return FileList(((row.id, row) for row in f.rows), f.state)


FileListSerializer = Annotated[
    FileList,
    ValidateAs(_FileListS, from_pydantic),
    PlainSerializer(to_pydantic),
]
