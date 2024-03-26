import json
import signal
import sys
import warnings
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import annotated_types
import fire
import h5py
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QColorConstants, QContextMenuEvent, QDragMoveEvent, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QUndoCommand,
    QUndoStack,
    QVBoxLayout,
    QWidget,
)

from lazyflow.utility import PathComponents

INPUT_DATA_PATH = "Input Data"
GREEN_CHECK = "✅"
RED_X = "❌"

NDShape = Annotated[Tuple[int, ...], annotated_types.Len(2, 6)]


signal.signal(signal.SIGINT, signal.SIG_DFL)


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
    model_config = ConfigDict(validate_assignment=True)

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
    _real_filename: Tuple[Path, Optional[Path]]

    def model_post_init(self, __context):
        self.__ilp_file = __context["ilp_file"]

    @property
    def file_exists(self):
        fp = self.ext_path
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
    def is_relative(self) -> bool:
        if self.klass == RELATIVE_CLASS:
            return True

        return False

    @property
    def can_relative(self) -> bool:
        fp = self.full_path
        ilp_loc = self.__ilp_file.parent
        return ilp_loc in fp.parents

    def replace_filepath(self, fp: Path, try_relative=False):
        pc = PathComponents(self.full_path.as_posix())
        pc_new = PathComponents(fp.as_posix())

        internal = pc.internalPath if pc_new.has_internal() else ""
        if internal:
            internal = internal.lstrip("/").lstrip("\\")

        if try_relative or self.klass == RELATIVE_CLASS:
            ilp_loc = self.__ilp_file.parent
            try:
                new_path = fp.relative_to(ilp_loc)
                new_klass = RELATIVE_CLASS

            except ValueError:
                new_path = fp
                new_klass = ABSOLUTE_CLASS

            self.file_path = new_path / internal if internal else new_path
            self.klass = new_klass

        else:
            self.file_path = fp / internal if internal else fp

    def update_ilp_location(self, location: Path):
        self.__ilp_file = location

    @property
    def ext_path(self):
        pc = PathComponents(self.full_path.as_posix())
        return Path(pc.externalPath)

    @property
    def int_path(self):
        pc = PathComponents(self.full_path.as_posix())
        return Path(pc.internalPath)


class InputData(BaseModel):
    role_names: Annotated[List[str], BeforeValidator(deserialize_str_list)] = Field(alias="Role Names")
    storage_version: Annotated[str, BeforeValidator(decode_ds)] = Field(alias="StorageVersion")
    infos: Dict[
        LaneName, Dict[str, Annotated[Optional[DatasetInfo], BeforeValidator(lambda x: None if len(x) == 0 else x)]]
    ]

    def model_post_init(self, __context):
        self.__ilp_file: Path = __context["ilp_file"]

    @property
    def ilp(self):
        return self.__ilp_file

    def update_ilp_location(self, location: Path):
        self.__ilp_file = location

        for lane_key, dataset_infos in self.infos.items():
            for role_name, info in dataset_infos.items():
                if info:
                    info.update_ilp_location(location)

    def get_summary(self) -> List[str]:
        summary: List[str] = []
        link_types: List[str] = []
        existing: List[bool] = []

        for lane_key, dataset_infos in self.infos.items():
            for role_name, info in dataset_infos.items():
                if info:
                    link_types.append(info.klass)
                    existing.append(info.file_exists)

        if all(existing):
            summary.append(f"{GREEN_CHECK} All {len(existing)} datasets can be reached on your machine.")
        else:
            summary.append(
                f"{RED_X} Problem: not all {len(existing)} datasets can be reached on your machine - missing {len([ex for ex in existing if not ex])}"
            )

        if all([lnk == RELATIVE_CLASS for lnk in link_types]):
            summary.append(
                f"All {len(link_types)} are relative links. The project can be moved on the same, as well as to other machines if data is moved along, relative to the project file."
            )
        else:
            summary.append(f"Not all links relative")

        return summary


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
                return row_data.file_path.as_posix()

            if idx_col == 2:
                return row_data.link_type

            if idx_col == 3:
                return GREEN_CHECK if row_data.file_exists else RED_X

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

        align = Qt.AlignLeft

        if idx_col == 0:
            # these two will not change (we do not allow reordering)
            current_value = f"{data_keys[0]}-{data_keys[1]}"
            original_value = f"{data_keys[0]}-{data_keys[1]}"
        elif idx_col == 1:
            current_value = row_data.file_path.as_posix()
            original_value = original_row_data.file_path.as_posix()
            align = Qt.AlignRight
        elif idx_col == 2:
            current_value = row_data.link_type
            original_value = original_row_data.link_type
        elif idx_col == 3:
            current_value = GREEN_CHECK if row_data.file_exists else RED_X
            original_value = GREEN_CHECK if original_row_data.file_exists else RED_X

        if current_value != original_value:
            painter.save()
            painter.setPen(QColor("red"))
            painter.drawText(option.rect, int(Qt.AlignTop | align), f"\t{original_value}")
            painter.setPen(QColor("green"))
            painter.drawText(option.rect, int(Qt.AlignBottom | align), f"\t{current_value}")
            painter.restore()
        else:
            painter.drawText(option.rect, Qt.AlignVCenter, f"\t{original_value}")


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
        label = QLabel("Search paths:")
        self._layout.addWidget(label)
        self.list_widget = QListWidget()

        list_layout.addWidget(self.list_widget)

        btn_layout = QVBoxLayout()

        addButton = QPushButton("+")
        addButton.clicked.connect(self.addFile)
        btn_layout.addWidget(addButton)

        removeButton = QPushButton("-")
        removeButton.clicked.connect(self.removeFile)
        btn_layout.addWidget(removeButton)

        addIlpButton = QPushButton("ilp")

        def _add_ilp_parent():
            parent = self._data.ilp.parent.absolute().as_posix()
            if not self.list_widget.findItems(parent, Qt.MatchFixedString):
                self.list_widget.addItem(parent)

        addIlpButton.clicked.connect(_add_ilp_parent)
        btn_layout.addWidget(addIlpButton)
        btn_layout.addStretch()

        list_layout.addLayout(btn_layout)

        self._layout.addLayout(list_layout)

        self.try_relative = QCheckBox("Prefer relative links")
        self.try_relative.setToolTip(
            f"Whenever possible links to files are resolved relative to the .ilp file location ({self._data.ilp.as_posix()})."
        )
        self._layout.addWidget(self.try_relative)

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


class SummaryWidget(QWidget):
    def __init__(self, data: InputDataModel, undo_stack: QUndoStack, parent: Optional["QWidget"] = None) -> None:
        super().__init__(parent)
        self._model: InputDataModel = data
        self._undo_stack = undo_stack
        self.setupUi()
        self.update_summary()

        self._model.dataChanged.connect(self.update_summary)

    def setupUi(self):
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        path_layout = QHBoxLayout()
        self.path_label = QLabel(f".ilp location: {self._model._data.ilp}")

        path_layout.addWidget(self.path_label)
        self._layout.addLayout(path_layout)

        self._text_edit = QLabel()
        self._text_edit.setWordWrap(True)
        self._layout.addWidget(self._text_edit)

    def update_summary(self, *args):
        self._text_edit.clear()

        data = self._model._data
        summary = "<br/> ".join(data.get_summary())
        self.path_label.setText(f"<b>.ilp location:</b> {data.ilp}")
        self._text_edit.setText(f"<b>Summary</b><br/>{summary}")


class PathApp(QWidget):
    def __init__(self, ilp_file, parent: Optional["QWidget"] = None):
        super().__init__(parent)
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

        summary = SummaryWidget(self.model, self._undo_stack)
        self._layout.addWidget(summary)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.change_button = QPushButton("Find files")
        self.change_button.clicked.connect(self.bulk_change)
        btn_layout.addWidget(self.change_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        btn_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        btn_layout.addWidget(self.redo_button)

        self.write_changes_button = QPushButton("Write changes")
        self.write_changes_button.clicked.connect(self.update_ilp)
        btn_layout.addWidget(self.write_changes_button)

        self._layout.addLayout(btn_layout)
        self.setMinimumSize(800, 600)
        self.table.resizeColumnsToContents()

    def changePath(self):
        index = self.table.currentIndex()
        idx_row = index.row()

        if index.isValid():
            new_path = QFileDialog.getOpenFileName(self, f"Select image file")[0]
            if new_path:
                self._change_path(idx_row, new_path)

    def _change_path(self, idx_row, new_path):
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

        prefer_relative = dlg.try_relative.isChecked()

        new_data = self.model._data.copy(deep=True)
        orig_data = new_data.copy(deep=True)

        for lane_key, dataset_infos in orig_data.infos.items():
            for role_name, info in dataset_infos.items():
                if info:
                    new_data.infos[lane_key][role_name] = self._find_file_in_search_paths(
                        info, selected_paths, prefer_relative=prefer_relative
                    )

        if orig_data != new_data:
            command = ChangeInfoCommand(index=self.table.currentIndex(), orig=orig_data, new=new_data)
            self._undo_stack.push(command)

    def _find_file_in_search_paths(
        self, ds: DatasetInfo, search_paths: List[Path], prefer_relative=True
    ) -> DatasetInfo:
        external = ds.ext_path.name

        ds_new = ds.copy(deep=True)
        for sp in search_paths:
            new_path = sp / external
            if new_path.exists():
                ds_new.replace_filepath(new_path, prefer_relative)
                return ds_new
        return ds_new

    def undo(self):
        self._undo_stack.undo()

    def redo(self):
        self._undo_stack.redo()

    def update_ilp(self):
        if self.model._data == self.model._original_data:
            QMessageBox.information(
                self, "No Changes", f"No changes were made, not updating\n{self.model._data.ilp.as_posix()}"
            )
            return


def startup_app(ilp_file: Path):
    app = QApplication(sys.argv)
    main_window = PathApp(ilp_file=ilp_file)
    main_window.show()
    return app.exec_()


if __name__ == "__main__":
    fire.Fire(startup_app)
