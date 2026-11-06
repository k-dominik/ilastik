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

from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from lazyflow.operators.ioOperators.types import FileList
from lazyflow.request import Request

from .opObjectClassificationCollectionExport import merge_collection_predictions

logger = logging.getLogger(__name__)


class CollectionExportTableModel(QAbstractTableModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.table = merge_collection_predictions(FileList(), {}, [])

    def set_table(self, table):
        self.beginResetModel()
        self.table = table
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.table.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.table.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        value = self.table.rows[index.row()][index.column()]
        if value is None:
            return ""
        return str(value) if role == Qt.ToolTipRole else value

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return self.table.headers[section] if orientation == Qt.Horizontal else section + 1


class ObjectClassificationCollectionExportGui(QWidget):
    refreshRequested = Signal()
    operationFinished = Signal(object, object, int, bool)

    def __init__(self, parentApplet, topLevelOperator):
        super().__init__()
        self.parentApplet = parentApplet
        self.topLevelOperator = topLevelOperator
        self._stopped = False
        self._dirty = True
        self._generation = 0
        self._request = None

        self.tableModel = CollectionExportTableModel(self)
        self.tableView = QTableView(self)
        self.tableView.setModel(self.tableModel)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tableView)

        self._drawer = QWidget()
        drawer_layout = QVBoxLayout(self._drawer)
        instructions = QLabel("Review the image metadata and predictions, then export the table as CSV.")
        instructions.setWordWrap(True)
        drawer_layout.addWidget(instructions)
        self.refreshButton = QPushButton("Refresh predictions")
        self.refreshButton.clicked.connect(self.refresh)
        drawer_layout.addWidget(self.refreshButton)
        self.exportButton = QPushButton("Export CSV...")
        self.exportButton.clicked.connect(self.export)
        drawer_layout.addWidget(self.exportButton)
        self.statusLabel = QLabel("Empty.")
        self.statusLabel.setWordWrap(True)
        drawer_layout.addWidget(self.statusLabel)
        drawer_layout.addStretch()

        self.refreshRequested.connect(self._invalidate, Qt.QueuedConnection)
        self.operationFinished.connect(self._finished, Qt.QueuedConnection)
        self._cleanup = [
            topLevelOperator.Table.notifyReady(self._request_refresh),
            topLevelOperator.Table.notifyUnready(self._request_refresh),
            topLevelOperator.Table.notifyDirty(self._request_refresh),
        ]
        self._update_buttons()

    def centralWidget(self):
        return self

    def appletDrawer(self):
        return self._drawer

    def menus(self):
        return []

    def viewerControlWidget(self):
        return None

    def secondaryControlsWidget(self):
        return None

    def allowLaneSelectionChange(self):
        return False

    def setImageIndex(self, index):
        pass

    def imageLaneAdded(self, laneIndex):
        pass

    def imageLaneRemoved(self, laneIndex, finalLength):
        self._request_refresh()

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def _request_refresh(self, *args):
        if not self._stopped:
            self.refreshRequested.emit()

    def _invalidate(self):
        if self._stopped:
            return
        self._generation += 1
        self._dirty = True
        if not self.topLevelOperator.Table.ready():
            self.tableModel.set_table(merge_collection_predictions(FileList(), {}, []))
            self.statusLabel.setText("Empty.")
        self._update_buttons()
        if self.isVisible():
            self.refresh()

    def _update_buttons(self):
        idle = self._request is None
        self.refreshButton.setEnabled(idle and self.topLevelOperator.Table.ready())
        self.exportButton.setEnabled(idle and not self._dirty and self.tableModel.table.can_export)

    def refresh(self):
        if not self._stopped and self._request is None and self.topLevelOperator.Table.ready():
            self._start_operation(export=False)

    def export(self):
        if self._request is not None:
            return
        dialog = QFileDialog(
            self, "Export collection predictions", str(self.topLevelOperator.export_path), "CSV files (*.csv)"
        )
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("csv")
        if dialog.exec_() == QDialog.Accepted:
            self.topLevelOperator.OutputFilename.setValue(dialog.selectedFiles()[0])
            self._start_operation(export=True)

    def _start_operation(self, export):
        generation = self._generation
        op = self.topLevelOperator
        work = op.run_export if export else lambda: op.Table.value
        self._request = request = Request(work)

        def finish(table=None, error=None):
            if not self._stopped:
                self.operationFinished.emit(table, error, generation, export)

        request.notify_finished(finish)
        request.notify_failed(lambda error, info: finish(error=error))
        request.notify_cancelled(lambda: finish(error=RuntimeError("Prediction request cancelled.")))
        self.parentApplet.busy = True
        self.parentApplet.appletStateUpdateRequested()
        self.parentApplet.progressSignal(0)
        self.parentApplet.progressSignal(1)
        self.statusLabel.setText("Exporting predictions..." if export else "Calculating predictions...")
        self._update_buttons()
        request.submit()

    def _finished(self, table, error, generation, exported):
        if self._stopped:
            return
        self._request = None
        self.parentApplet.busy = False
        self.parentApplet.progressSignal(100)
        self.parentApplet.appletStateUpdateRequested()
        if error is not None:
            logger.error("Collection export failed: %s", error)
            self.statusLabel.setText(str(error))
            if exported:
                QMessageBox.critical(self, "Export failed", str(error))
        elif generation == self._generation:
            self.tableModel.set_table(table)
            self._dirty = False
            if exported:
                self.statusLabel.setText(f"Exported {len(table.rows)} files to {self.topLevelOperator.export_path}")
            elif table.missing_predictions:
                self.statusLabel.setText(
                    f"{len(table.rows)} files; {len(table.missing_predictions)} have no predictions. "
                    "Calculate embeddings and train a classifier to enable export."
                )
            else:
                self.statusLabel.setText(f"{len(table.rows)} files ready to export.")
        self._update_buttons()
        if error is None and self._dirty and self.isVisible():
            self.refresh()

    def stopAndCleanUp(self):
        self._stopped = True
        for unsubscribe in self._cleanup:
            unsubscribe()
        self._cleanup.clear()
        if self._request is not None:
            self._request.cancel()
        self.parentApplet.busy = False
