import enum
import json
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import annotated_types
import h5py
from pydantic import BaseModel, BeforeValidator, Field, StringConstraints
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QColorConstants, QContextMenuEvent, QDragMoveEvent, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QListView,
    QListWidget,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QUndoCommand,
    QUndoStack,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QDialogButtonBox,
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

RELATIVE_CLASS = "RelativeFilesystemDatasetInfo"
ABSOLUTE_CLASS = "FilesystemDatasetInfo"


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
        fp = self.full_path
        return fp.exists()

    @property
    def link_type(self) -> str:
        if "relative" in self.klass.lower():
            return "relative"
        else:
            return "absolute"

    @property
    def full_path(self) -> Path:
        if "relative" in self.klass.lower():
            return self.__ilp_file.parent / self.file_path

        return self.file_path

    @property
    def can_relative(self) -> bool:
        fp = self.full_path
        ilp_loc = self.__ilp_file.parent
        return ilp_loc in fp.parents

    def replace_filepath(self, fp: Path, try_relative: bool = False):
        if try_relative or self.klass == RELATIVE_CLASS:

            ilp_loc = self.__ilp_file.parent
            try:
                new_path = fp.relative_to(ilp_loc)
                new_klass = RELATIVE_CLASS

            except ValueError:
                new_path = fp
                new_klass = ABSOLUTE_CLASS

            self.file_path = new_path
            self.klass = new_klass

        else:
            self.file_path = fp


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


class InputDataModel(QAbstractTableModel):
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
                return row_data.link_type

            if idx_col == 3:
                return "✅" if row_data.file_exists else "❌"

            assert False

    # def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
    #     if index.isValid() and role == Qt.EditRole:
    #         ...
    #         self.dataChanged.emit(index, index, [role])
    #         return True
    #     return False

    def flags(self, index):
        # if index.column() == 1:
        #     return super().flags(index) | Qt.ItemIsEditable

        return super().flags(index)

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]

    def itemFromIndex(self, index):
        idx_row = index.row()
        data_keys = self._row_keys[idx_row]
        return self._data.infos[data_keys[0]][data_keys[1]]


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
            current_value = "✅" if row_data.file_exists else "❌"
            original_value = "✅" if original_row_data.file_exists else "❌"

        if current_value != original_value:
            painter.save()
            painter.setPen(QColor("red"))
            painter.drawText(option.rect, Qt.AlignTop, f"\t{original_value}")
            painter.setPen(QColor("green"))
            painter.drawText(option.rect, Qt.AlignBottom, f"\t{current_value}")
            painter.restore()
        else:
            QStyledItemDelegate.paint(self, painter, option, index)


class ChangeInfoCommand(QUndoCommand):
    def __init__(self, parent=None, *, index: QModelIndex, orig: InputData, new: InputData):
        super().__init__(parent)
        self._index = QPersistentModelIndex(index)
        self._old_input_info = orig
        self._new_input_info = new

    def redo(self):
        assert self._index.isValid()

        self._index.model()._data = self._new_input_info

        change_start = self._index.model().index(0, 0)
        change_stop = self._index.model().index(self._index.model().rowCount(), self._index.model().columnCount())

        self._index.model().dataChanged.emit(change_start, change_stop)

    def undo(self):
        assert self._index.isValid()

        self._index.model()._data = self._old_input_info

        change_start = self._index.model().index(0, 0)
        change_stop = self._index.model().index(self._index.model().rowCount(), self._index.model().columnCount())

        self._index.model().dataChanged.emit(change_start, change_stop)


class InputDataView(QTableView):

    replacePathEvent = pyqtSignal(int, Path)  # row, path

    def contextMenuEvent(self, a0: QContextMenuEvent) -> None:
        return super().contextMenuEvent(a0)

    def dragEnterEvent(self, a0):
        # Only accept drag-and-drop events that consist of urls to local files.
        if not a0.mimeData().hasUrls():
            return
        urls = a0.mimeData().urls()
        if len(urls) != 1:
            return
        if all(map(QUrl.isLocalFile, urls)):
            a0.acceptProposedAction()

    def dragMoveEvent(self, a0: QDragMoveEvent):
        pass

    def dropEvent(self, a0):
        file_paths = [Path(QUrl.toLocalFile(url)) for url in a0.mimeData().urls()]

        idx_row = self.rowAt(a0.pos().y())
        # Last row is the button.
        if idx_row == -1:
            return
        self.replacePathEvent.emit(idx_row, file_paths[0])


class BulkChangeDialog(QDialog):
    def __init__(self, data: InputData, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._data = data

        self.setupUi()

    def setupUi(self):
        self._layout = QVBoxLayout()

        list_layout = QHBoxLayout()

        self.list_widget = QListWidget()

        list_layout.addWidget(self.list_widget)

        btn_layout = QVBoxLayout()

        addButton = QPushButton("+")
        addButton.clicked.connect(self.addFile)
        btn_layout.addWidget(addButton)

        removeButton = QPushButton("-")
        removeButton.clicked.connect(self.removeFile)
        btn_layout.addWidget(removeButton)
        btn_layout.addStretch()

        list_layout.addLayout(btn_layout)

        self._layout.addLayout(list_layout)

        default_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        default_buttons.accepted.connect(self.accept)
        default_buttons.rejected.connect(self.reject)
        self._layout.addWidget(default_buttons)

        self.setLayout(self._layout)

        self.list_widget.setDragDropMode(QListWidget.InternalMove)

    def addFile(self):
        filePath = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if filePath:
            self.list_widget.addItem(filePath)

    def removeFile(self):
        for selectedItem in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(selectedItem))

    def selected_paths(self) -> List[Path]:
        return [Path(self.list_widget.item(idx).text()) for idx in range(self.list_widget.count())]


class PathApp(QWidget):
    def __init__(self, ilp_file):
        super().__init__()
        self._undo_stack = QUndoStack()
        self._ilp_file = Path(ilp_file)
        assert self._ilp_file.exists()
        with h5py.File(self._ilp_file, "r") as f:
            data = InputData.model_validate(f[INPUT_DATA_PATH], context={"ilp_file": self._ilp_file})

        self.model = InputDataModel(data)

        self.initUI()

    def initUI(self):
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        table = InputDataView()
        table.setAcceptDrops(True)
        # Set model for the table view
        table.setModel(self.model)
        table.replacePathEvent.connect(self._change_path)

        delegate = CustomItemDelegate()
        table.setItemDelegate(delegate)

        self.table = table
        self.table.doubleClicked.connect(self.changePath)
        self._layout.addWidget(self.table)

        self.change_button = QPushButton("Bulk Change")
        self.change_button.clicked.connect(self.bulk_change)
        self._layout.addWidget(self.change_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self._layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        self._layout.addWidget(self.redo_button)

    def changePath(self):
        index = self.table.currentIndex()
        idx_row = index.row()

        if index.isValid():
            new_path = QFileDialog.getOpenFileName(self, f"Select image file")[0]
            if new_path:
                self._change_path(idx_row, new_path)

    def _change_path(self, idx_row, new_path):
        print("change path", idx_row, new_path)
        index = self.table.currentIndex()
        if index.isValid():
            data_keys = index.model()._row_keys[idx_row]
            new_data = index.model()._data.copy(deep=True)
            orig_data = new_data.copy(deep=True)
            new_data.infos[data_keys[0]][data_keys[1]].replace_filepath(Path(new_path))
            command = ChangeInfoCommand(index=index, orig=orig_data, new=new_data)
            self._undo_stack.push(command)

    def bulk_change(self):
        dlg = BulkChangeDialog(data=self.model._data)
        dlg.exec()

        selected_paths = dlg.selected_paths()
        if not selected_paths:
            return

    def undo(self):
        idx = self._undo_stack.undo()

    def redo(self):
        self._undo_stack.redo()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PathApp(ilp_file="/Users/kutra/test.ilp")
    main_window.show()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        sys.exit(0)
