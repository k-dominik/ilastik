import enum
import json
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Any

import annotated_types
import h5py
from pydantic import BaseModel, BeforeValidator, Field, StringConstraints, computed_field
from PyQt5.QtCore import Qt, QSize, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPainter, QColorConstants, QColor
from PyQt5.QtWidgets import (
    QItemDelegate,
    QApplication,
    QFileDialog,
    QListView,
    QPushButton,
    QUndoCommand,
    QUndoStack,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QStyle,
    QTableView,
    QStyledItemDelegate,
)

INPUT_DATA_PATH = "Input Data"

NDShape = Annotated[Tuple[int, ...], annotated_types.Len(2, 6)]


def deserialize_str_list(str_list):
    return [x.decode() for x in str_list]


def decode_ds(bin_ds):
    return bin_ds[()].decode()


def at_from_ds(ds):
    return [VigraAxisTags(**info) for info in json.loads(ds[()].decode())["axes"]]


def single_from_ds(ds):
    return ds[()]


def data_from_ds(ds):
    return ds[()]


class VigraAxisTags(BaseModel):
    key: str
    typeFlags: int
    resolution: int
    description: str


LaneName = Annotated[str, StringConstraints(pattern=r"lane\d{4}")]


class DatasetInfo(BaseModel):
    allow_labels: Annotated[bool, BeforeValidator(single_from_ds)] = Field(alias="allowLabels")
    axistags: Annotated[List[VigraAxisTags], BeforeValidator(at_from_ds)]
    dataset_id: Annotated[str, BeforeValidator(decode_ds)] = Field(alias="datasetId")
    display_mode: Annotated[
        Literal["default", "grayscale", "rgba", "random-colortable", "alpha-modulated", "binary-mask"],
        BeforeValidator(decode_ds),
    ]
    file_path: Annotated[Path, BeforeValidator(decode_ds)] = Field(alias="filePath")
    klass: Annotated[str, BeforeValidator(decode_ds)] = Field(alias="__class__")
    location: Annotated[str, BeforeValidator(decode_ds)]
    nickname: Annotated[str, BeforeValidator(decode_ds)]
    shape: Annotated[NDShape, BeforeValidator(lambda x: tuple(x.tolist())), BeforeValidator(data_from_ds)]
    normalize_display: Annotated[Optional[bool], BeforeValidator(single_from_ds)] = Field(alias="normalizeDisplay")
    scale_locked: Annotated[Optional[bool], BeforeValidator(single_from_ds)] = None
    working_scale: Annotated[Optional[str], BeforeValidator(decode_ds)] = None

    _replace_path: Optional[Path]

    def model_post_init(self, __context):
        print(__context)
        self.__ilp_file = __context["ilp_file"]

    @property
    def file_exists(self):
        if "relative" in self.klass.lower():
            f = self.__ilp_file.parent / self.file_path
        else:
            f = self.file_path
        print(f, f.exists())
        return f.exists()

    def replace_filepath(self, fp):
        self._replace_path = fp


class InputData(BaseModel):
    role_names: Annotated[List[str], BeforeValidator(deserialize_str_list)] = Field(alias="Role Names")
    storage_version: Annotated[str, BeforeValidator(decode_ds)] = Field(alias="StorageVersion")
    infos: Dict[
        LaneName, Dict[str, Annotated[Optional[DatasetInfo], BeforeValidator(lambda x: None if len(x) == 0 else x)]]
    ]


NROWS = 5
ROWNUDGE = 5
LEFTNUDGE = 5


class Columns(enum.IntEnum):
    LANE = 0
    FILEPATH = 1
    DESCRIPTION = 2


class TableModel(QAbstractTableModel):
    def __init__(self, data: InputData):
        super().__init__()
        self._data = data
        self._original_data = data.copy(deep=True)
        self._row_keys = []
        for lane_key, dataset_infos in data.infos.items():
            for role_name, info in dataset_infos.items():
                if not info:
                    continue

                self._row_keys.append((lane_key, role_name))

        self._headers = ["Lane", "Filename", "Link Type", "Found"]

    def rowCount(self, parent=None):
        return len(self._row_keys)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role):  # type: ignore[reportIncompatibleMethodOverride]
        if role in (Qt.DisplayRole, Qt.EditRole):
            idx_row = index.row()
            idx_col = index.column()
            data_keys = self._row_keys[idx_row]

            if idx_col == 0:
                return f"{data_keys[0]}-{data_keys[1]}"

            row_data = self._data.infos[data_keys[0]][data_keys[1]]
            assert row_data

            if idx_col == 1:
                return str(row_data.file_path)

            if idx_col == 2:
                return row_data.location

            if idx_col == 3:
                return row_data.file_exists

            assert False

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if index.isValid() and role == Qt.EditRole:
            row = self._data[index.row()]
            column = index.column()
            if column == 0:
                row.original["filename"] = row.filename
                row.filename = value
            elif column == 1:
                row.original["shape"] = row.shape
                row.shape = value
            elif column == 2:
                row.original["description"] = row.description
                row.description = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(self, index):
        if index.column() == 1:
            return super().flags(index) | Qt.ItemIsEditable

        return super().flags(index)

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]


class CustomItemDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index: QModelIndex):
        idx_row = index.row()
        idx_col = index.column()
        data_keys = index.model()._row_keys[idx_row]

        row_data = index.model()._data.infos[data_keys[0]][data_keys[1]]
        original_row_data = index.model()._original_data.infos[data_keys[0]][data_keys[1]]

        idx_col = index.column()

        original_value = None
        current_value = None

        if idx_col == 0:
            # these two will not change (we do not allow reordering)
            current_value = f"{data_keys[0]}-{data_keys[1]}"
            original_value = f"{data_keys[0]}-{data_keys[1]}"
        elif idx_col == 1:
            current_value = row_data.file_path
            original_value = original_row_data.file_path
        elif idx_col == 2:
            current_value = row_data.location
            original_value = original_row_data.location
        elif idx_col == 3:
            current_value = row_data.file_exists
            original_value = original_row_data.file_exists

        if current_value != original_value:
            painter.save()
            painter.setPen(QColor("red"))
            painter.drawText(option.rect, Qt.AlignTop, original_value)
            painter.setPen(QColor("green"))
            painter.drawText(option.rect, Qt.AlignBottom, current_value)
            painter.restore()
        else:
            QStyledItemDelegate.paint(self, painter, option, index)


# class ChangeInfoCommand(QUndoCommand):
#     def __init__(self, parent=None, *, item: DataSetInfoItem, new_info: DatasetInfo):
#         super().__init__(parent)
#         self._item = item
#         self._old_info = item.info
#         self._new_info = new_info

#     def redo(self):
#         self._item.info = self._new_info
#         self._item.emitDataChanged()

#     def undo(self):
#         undo_data = self._old_info
#         self._item.info = undo_data
#         self._item.emitDataChanged()


class PathApp(QWidget):
    def __init__(self, ilp_file):
        super().__init__()
        self._undo_stack = QUndoStack()
        self._ilp_file = Path(ilp_file)
        assert self._ilp_file.exists()
        with h5py.File(self._ilp_file, "r") as f:
            data = InputData.model_validate(f[INPUT_DATA_PATH], context={"ilp_file": self._ilp_file})

        self._data = data

        self.model = TableModel(data)

        self.initUI()

    def initUI(self):
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        table = QTableView()

        # Set model for the table view
        table.setModel(self.model)

        delegate = CustomItemDelegate()
        table.setItemDelegate(delegate)

        self.table = table
        # self.table.doubleClicked.connect(self.changePath)
        self._layout.addWidget(self.table)

        self.change_button = QPushButton("Change Path")
        # self.change_button.clicked.connect(self.changePath)
        self._layout.addWidget(self.change_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self._layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        self._layout.addWidget(self.redo_button)

    # def changePath(self):
    #     index = self.table.currentIndex()
    #     if index.isValid():
    #         info: DatasetInfoItem = self.model.itemFromIndex(index)
    #         new_info = info.info.copy(deep=True)
    #         new_path = QFileDialog.getOpenFileName(self, f"Replace `{new_info.file_path!r}`")[0]
    #         if new_path:
    #             new_info.file_path = Path(new_path)
    #             command = ChangeInfoCommand(item=info, new_info=new_info)
    #             self._undo_stack.push(command)

    def undo(self):
        idx = self._undo_stack.index()
        self._undo_stack.setIndex(idx - 1)

    def redo(self):
        cap = self._undo_stack.count()
        idx = self._undo_stack.index()
        self._undo_stack.setIndex(min(cap, idx + 1))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PathApp(ilp_file="/Users/kutra/test.ilp")
    main_window.show()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        sys.exit(0)
