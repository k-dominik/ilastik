import json

import vigra

from lazyflow.base import ItemId
from lazyflow.operators.ioOperators.types import (
    _FileListS,
    FileList,
    FileListDataRow,
    FileListDataRowNoId,
    FileListSerializer,
)
import numpy

import pytest

from pydantic import TypeAdapter


@pytest.fixture
def file_list() -> FileList:
    row = FileListDataRowNoId(
        filename="test.f",
        axistags=vigra.defaultAxistags("xyc"),
        original_axistags=vigra.defaultAxistags("xy"),
        shape=(1, 2, 3),
        original_shape=(1, 2),
        dtype=numpy.dtype(numpy.uint8),
    )
    f = FileList()
    f.add_no_id(row)
    return f


class TestFileList:
    def test_add_row(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        f = FileList()
        f.add_no_id(row)
        assert f.state() == 1

    def test_add_row_w_init(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        f = FileList(init_count=42)
        assert f.state() == 42
        f.add_no_id(row)
        assert f.state() == 43

    def test_add_row_repeatedly_gives_new_id(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        f = FileList()
        assert f.state() == 0
        f.add_no_id(row)
        assert list(f.keys()) == [ItemId(1)]
        assert f.state() == 1
        del f[next(iter(f.keys()))]
        assert f.state() == 1
        f.add_no_id(row)
        assert f.state() == 2
        assert list(f.keys()) == [ItemId(2)]

    def test_serialize_python(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        f = FileList()
        f.add_no_id(row)

        tv = TypeAdapter(FileListSerializer)
        file_list_dict = tv.dump_python(f)
        assert file_list_dict == {
            "state": 1,
            "rows": [
                {
                    "id": 1,
                    "filename": "test.f",
                    "dtype": "uint8",
                    "shape": (1, 2, 3),
                    "axistags": vigra.defaultAxistags("xyc").toJSON(),
                    "original_shape": (1, 2),
                    "original_axistags": vigra.defaultAxistags("xy").toJSON(),
                },
            ],
        }

    def test_deserialize_python(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        expected = FileList()
        expected.add_no_id(row)

        tv = TypeAdapter(FileListSerializer)
        file_list = tv.validate_python(
            {
                "state": 1,
                "rows": [
                    {
                        "id": 1,
                        "filename": "test.f",
                        "dtype": "uint8",
                        "shape": (1, 2, 3),
                        "axistags": vigra.defaultAxistags("xyc").toJSON(),
                        "original_shape": (1, 2),
                        "original_axistags": vigra.defaultAxistags("xy").toJSON(),
                    },
                ],
            }
        )
        assert file_list == expected

    def test_serialize_json(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        f = FileList()
        f.add_no_id(row)

        tv = TypeAdapter(FileListSerializer)
        file_list_dict = tv.dump_json(f)
        assert json.loads(file_list_dict) == {
            "state": 1,
            "rows": [
                {
                    "id": 1,
                    "filename": "test.f",
                    "dtype": "uint8",
                    "shape": [1, 2, 3],
                    "axistags": vigra.defaultAxistags("xyc").toJSON(),
                    "original_shape": [1, 2],
                    "original_axistags": vigra.defaultAxistags("xy").toJSON(),
                },
            ],
        }

    def test_deserialize_json(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        expected = FileList()
        expected.add_no_id(row)

        tv = TypeAdapter(FileListSerializer)
        file_list = tv.validate_json(
            json.dumps(
                {
                    "state": 1,
                    "rows": [
                        {
                            "id": 1,
                            "filename": "test.f",
                            "dtype": "uint8",
                            "shape": [1, 2, 3],
                            "axistags": vigra.defaultAxistags("xyc").toJSON(),
                            "original_shape": [1, 2],
                            "original_axistags": vigra.defaultAxistags("xy").toJSON(),
                        },
                    ],
                }
            )
        )
        assert file_list == expected


class TestRow:
    def test_serialize_python(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )

        row_s = row.model_dump()

        assert row_s == {
            "dtype": "uint8",
            "filename": "test.f",
            "axistags": vigra.defaultAxistags("xyc").toJSON(),
            "original_axistags": vigra.defaultAxistags("xy").toJSON(),
            "original_shape": (1, 2),
            "shape": (1, 2, 3),
        }

    def test_serialize_json(self):
        row = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )

        row_s = row.model_dump_json()

        assert json.loads(row_s) == {
            "dtype": "uint8",
            "filename": "test.f",
            "axistags": vigra.defaultAxistags("xyc").toJSON(),
            "original_axistags": vigra.defaultAxistags("xy").toJSON(),
            "original_shape": [1, 2],
            "shape": [1, 2, 3],
        }

    def test_deserialize_python(self):
        expected = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        row = FileListDataRowNoId.model_validate(
            {
                "dtype": "uint8",
                "filename": "test.f",
                "axistags": vigra.defaultAxistags("xyc").toJSON(),
                "original_axistags": vigra.defaultAxistags("xy").toJSON(),
                "original_shape": (1, 2),
                "shape": (1, 2, 3),
            }
        )
        assert row == expected

    def test_deserialize_json(self):
        expected = FileListDataRowNoId(
            filename="test.f",
            axistags=vigra.defaultAxistags("xyc"),
            original_axistags=vigra.defaultAxistags("xy"),
            shape=(1, 2, 3),
            original_shape=(1, 2),
            dtype=numpy.dtype(numpy.uint8),
        )
        row = FileListDataRowNoId.model_validate_json(
            json.dumps(
                {
                    "dtype": "uint8",
                    "filename": "test.f",
                    "axistags": vigra.defaultAxistags("xyc").toJSON(),
                    "original_axistags": vigra.defaultAxistags("xy").toJSON(),
                    "original_shape": [1, 2],
                    "shape": [1, 2, 3],
                }
            )
        )
        assert row == expected
