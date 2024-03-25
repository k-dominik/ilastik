import json
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import annotated_types
import h5py
from pydantic import BaseModel, BeforeValidator, Field, StringConstraints, computed_field
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPainter, QColorConstants
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


class DataSetInfoItem(QStandardItem):
    def __init__(self, text: str, info: DatasetInfo):
        super().__init__(text)
        self.info = info
        self.title = text

    def data(self, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.info
        elif role == Qt.UserRole:
            return self.title
        return super().data(role)


NROWS = 5
ROWNUDGE = 5
LEFTNUDGE = 5


class DatasetInfoDelegate(QItemDelegate):
    def paint(self, painter: QPainter, option, index):
        rect = option.rect
        data: DatasetInfo = index.data()
        title: str = index.data(Qt.UserRole)

        tagged_shape = dict(zip([ax.key for ax in data.axistags], data.shape))

        if not data.file_exists:
            painter.fillRect(option.rect, QColorConstants.Svg.lightpink)

        if bool(option.state & QStyle.State_MouseOver):
            painter.fillRect(option.rect, option.palette.highlight())

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(LEFTNUDGE + rect.x(), rect.y() + rect.height() // NROWS + ROWNUDGE, title)

        font.setBold(False)
        painter.setFont(font)
        painter.drawText(LEFTNUDGE + rect.x(), rect.y() + 2.5 * rect.height() // NROWS, str(data.file_path))
        painter.drawText(LEFTNUDGE + rect.x(), rect.y() + 3.5 * rect.height() // NROWS, str(data.klass))
        painter.drawText(LEFTNUDGE + rect.x(), rect.y() + 5 * rect.height() // NROWS - ROWNUDGE, str(tagged_shape))

        painter.drawRect(rect)

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), NROWS * 20 + 10)


class ChangeInfoCommand(QUndoCommand):
    def __init__(self, parent=None, *, item: DataSetInfoItem, new_info: DatasetInfo):
        super().__init__(parent)
        self._item = item
        self._old_info = item.info
        self._new_info = new_info

    def redo(self):
        self._item.info = self._new_info
        self._item.emitDataChanged()

    def undo(self):
        undo_data = self._old_info
        self._item.info = undo_data
        self._item.emitDataChanged()


class PathApp(QWidget):
    def __init__(self, ilp_file):
        super().__init__()
        self._undo_stack = QUndoStack()
        self._ilp_file = Path(ilp_file)
        assert self._ilp_file.exists()
        with h5py.File(self._ilp_file, "r") as f:
            data = InputData.model_validate(f[INPUT_DATA_PATH], context={"ilp_file": self._ilp_file})

        self._data = data

        self.model = QStandardItemModel()

        for lane_id, infos in self._data.infos.items():
            for role, info in infos.items():
                if info is None:
                    continue
                self.model.appendRow(DataSetInfoItem(f"{lane_id}-{role}", info))
        self.initUI()

    def initUI(self):
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setItemDelegate(DatasetInfoDelegate(self))
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setSpacing(5)
        self.view.setMouseTracking(True)
        self.view.doubleClicked.connect(self.changePath)
        self._layout.addWidget(self.view)

        self.change_button = QPushButton("Change Path")
        self.change_button.clicked.connect(self.changePath)
        self._layout.addWidget(self.change_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self._layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        self._layout.addWidget(self.redo_button)

    def changePath(self):
        index = self.view.currentIndex()
        if index.isValid():
            info: DatasetInfoItem = self.model.itemFromIndex(index)
            new_info = info.info.copy(deep=True)
            new_path = QFileDialog.getOpenFileName(self, f"Replace `{new_info.file_path!r}`")[0]
            if new_path:
                new_info.file_path = Path(new_path)
                command = ChangeInfoCommand(item=info, new_info=new_info)
                self._undo_stack.push(command)

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
