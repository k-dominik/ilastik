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
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableView

from .fileListTableModel import FileListColumn


class FileListTableView(QTableView):
    addFilesRequested = Signal()
    filesDropped = Signal(list)
    removeFilesRequested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        # Stretching overrides interactive resizing for the last column.
        self.horizontalHeader().setStretchLastSection(False)

    def selected_item_ids(self):
        if self.selectionModel() is None:
            return []
        return [index.data(Qt.UserRole) for index in self.selectionModel().selectedRows(FileListColumn.Id)]

    def removeSelectedFiles(self):
        ids = self.selected_item_ids()
        if ids:
            self.removeFilesRequested.emit(ids)

    def _context_menu(self, point):
        menu = QMenu(self)
        menu.addAction("Add files...", self.addFilesRequested.emit)
        remove = menu.addAction("Remove selected files", self.removeSelectedFiles)
        remove.setEnabled(bool(self.selected_item_ids()))
        menu.exec_(self.viewport().mapToGlobal(point))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.removeSelectedFiles()
            event.accept()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and all(url.isLocalFile() for url in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and all(url.isLocalFile() for url in urls):
            self.filesDropped.emit([url.toLocalFile() for url in urls])
            event.acceptProposedAction()
        else:
            event.ignore()
