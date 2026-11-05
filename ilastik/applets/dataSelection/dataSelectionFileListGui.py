###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2026, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#          http://ilastik.org/license.html
###############################################################################
import logging

from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from volumina.interpreter import ClickInterpreter

from lazyflow.operators.ioOperators.functional.imageReader import SimpleRoi
from lazyflow.operators.ioOperators.opFileListToGridReader import determine_grid_shape

from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from ilastik.utility.gui import ThreadRouter, threadRouted
from ilastik.widgets.ImageFileDialog import ImageFileDialog

from .fileListTableModel import FileListTableModel
from .fileListTableView import FileListTableView

logger = logging.getLogger(__name__)


class _DoubleClickInterpreter(ClickInterpreter):
    """Small wrapper around ClickInterpreter that forwards double clicks"""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonDblClick and event.button() == self.button:
            super().eventFilter(watched, event)

        return self.baseInterpret.eventFilter(watched, event)


class _CollectionViewer(LayerViewerGui):
    def setupLayers(self):
        slot = self.topLevelOperatorView.Image
        if not slot.ready():
            return []
        layer = self.createStandardLayerFromSlot(slot)
        layer.name = "Image collection"
        return [layer]


class DataSelectionFileListGui(QWidget):
    def __init__(self, parentApplet, dataSelectionOperator, instructionText):
        super().__init__()
        self.parentApplet = parentApplet
        self.topLevelOperator = dataSelectionOperator
        self._viewer = None
        self._stopped = False
        self.threadRouter = ThreadRouter(self)

        self._drawer = QWidget()
        drawer_layout = QVBoxLayout(self._drawer)
        instructions = QLabel(instructionText)
        instructions.setWordWrap(True)
        drawer_layout.addWidget(instructions)
        drawer_layout.addStretch()

        self.tableView = FileListTableView(self)
        self.tableModel = FileListTableModel(self, self.topLevelOperator)
        self.tableView.setModel(self.tableModel)
        self.tableView.addFilesRequested.connect(self.addFiles)
        self.tableView.filesDropped.connect(self.addFileNames)
        self.tableView.removeFilesRequested.connect(self.removeFiles)
        self.tableView.doubleClicked.connect(self._show_cell)

        add_button = QPushButton("Add files...")
        add_button.clicked.connect(self.addFiles)
        self.removeButton = QPushButton("Remove selected files")
        self.removeButton.clicked.connect(self.tableView.removeSelectedFiles)
        self.removeButton.setEnabled(False)
        self.tableView.selectionModel().selectionChanged.connect(self._update_remove_button)
        self.tableModel.modelReset.connect(self._update_remove_button)

        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(self.removeButton)
        buttons.addStretch()
        self._countLabel = QLabel()
        buttons.addWidget(self._countLabel)

        self._viewerStack = QStackedWidget(self)
        self._viewerStack.addWidget(QWidget())
        self._viewerControlWidgetStack = QStackedWidget(self)
        splitter = QSplitter(Qt.Vertical, self)
        splitter.addWidget(self.tableView)
        splitter.addWidget(self._viewerStack)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(splitter)
        self._update_remove_button()
        if len(self.topLevelOperator.DatasetGroup):
            self.imageLaneAdded(0)

    def centralWidget(self):
        return self

    def appletDrawer(self):
        return self._drawer

    def menus(self):
        return []

    def viewerControlWidget(self):
        return self._viewerControlWidgetStack

    def secondaryControlsWidget(self):
        return None

    def allowLaneSelectionChange(self):
        return False

    def setImageIndex(self, imageIndex):
        # File selection is independent of the shell's sole image lane.
        pass

    @threadRouted
    def imageLaneAdded(self, laneIndex):
        if self._stopped or self._viewer is not None:
            return
        if laneIndex != 0:
            raise ValueError("File-list data selection supports only lane zero.")
        self._viewer = _CollectionViewer(self.parentApplet, self.topLevelOperator.getLane(0), crosshair=False)
        editor = self._viewer.editor
        editor.setNavigationInterpreter(
            _DoubleClickInterpreter(editor, None, self._select_cell_row, parent=self._viewer, right=False, double=True)
        )
        self._viewer.volumeEditorWidget.quadview.ensureMaximized(2)
        self._viewerStack.addWidget(self._viewer)
        self._viewerStack.setCurrentWidget(self._viewer)
        self._viewerControlWidgetStack.addWidget(self._viewer.viewerControlWidget())

    def _select_cell_row(self, layer, pos5d, window_position):
        table = self.parentApplet.file_list
        if not table:
            return
        grid, *_ = determine_grid_shape(table)
        if not all(0 <= position < size for position, size in zip(pos5d, grid.output_shape)):
            return
        cell = next(grid.grid_cells_per_roi(SimpleRoi(start=pos5d, stop=tuple(p + 1 for p in pos5d))))
        if cell.image_index is None:
            return
        for row in range(self.tableModel.rowCount()):
            if self.tableModel.get_row(row).id == cell.image_index:
                self.tableView.selectRow(row)
                self.tableView.scrollTo(self.tableModel.index(row, 0), QAbstractItemView.PositionAtCenter)
                break

    def _show_cell(self, index):
        if self._viewer is None or not index.isValid():
            return
        table = self.parentApplet.file_list
        item_id = self.tableModel.get_row(index.row()).id
        if item_id not in table:
            return
        grid, *_ = determine_grid_shape(table)
        cell_size = grid.grid_size_cell_px
        grid_width = grid.output_shape[1] // cell_size
        cell_y, cell_x = divmod(list(table).index(item_id), grid_width)
        position = [size // 2 for size in grid.output_shape]
        position[1] = cell_x * cell_size + cell_size // 2
        position[2] = cell_y * cell_size + cell_size // 2
        self._viewer.setViewerPos(position)

    @threadRouted
    def imageLaneRemoved(self, laneIndex, finalLength):
        self._clean_up_viewer()

    def _clean_up_viewer(self):
        if self._viewer is not None:
            self._viewerStack.removeWidget(self._viewer)
            self._viewerControlWidgetStack.removeWidget(self._viewer.viewerControlWidget())
            self._viewer.stopAndCleanUp()
            self._viewer = None

    def stopAndCleanUp(self):
        self._stopped = True
        self.tableModel.stopAndCleanUp()
        self._clean_up_viewer()

    def _update_remove_button(self, *args):
        self.removeButton.setEnabled(bool(self.tableView.selected_item_ids()))
        self._countLabel.setText(f"{self.tableModel.rowCount()} files")

    def addFiles(self):
        self.addFileNames(ImageFileDialog(self).getSelectedPaths())

    def _change_collection(self, change):
        self.parentApplet.busy = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.parentApplet.appletStateUpdateRequested()
            change()
        except Exception as error:
            logger.exception("Could not update image collection")
            QMessageBox.critical(self, "File selection error", str(error))
        finally:
            QApplication.restoreOverrideCursor()
            self.parentApplet.busy = False
            self.parentApplet.appletStateUpdateRequested()

    def addFileNames(self, paths):
        if paths:
            self._change_collection(lambda: self.parentApplet.add_files(paths))

    def removeFiles(self, item_ids):
        if item_ids:
            self._change_collection(lambda: self.parentApplet.remove_files(item_ids))
