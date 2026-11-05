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

from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

from lazyflow.operators.ioOperators.types import FileListDataRow


class FileListColumn:
    Id = 0
    Name = 1
    Location = 2
    TaggedShape = 3
    Dtype = 4
    NumColumns = 5


class FileListTableModel(QAbstractTableModel):
    refreshRequested = Signal()

    def __init__(self, parent, topLevelOperator):
        super().__init__(parent)
        self._op = topLevelOperator
        self._rows: list[FileListDataRow] = []
        self._slot = None
        self._stopped = False
        self._refresh_pending = False
        self._slot_cleanup = []
        # Slot notifications may arrive from a worker. Read and reset the model
        # later in the GUI thread, after the operator has finished reconfiguring.
        self.refreshRequested.connect(self._refresh, Qt.QueuedConnection)
        self._cleanup = [
            self._op.FileListTable.notifyInserted(self._request_refresh),
            self._op.FileListTable.notifyRemoved(self._request_refresh),
        ]
        self._refresh()

    def _request_refresh(self, *args):
        if not self._stopped and not self._refresh_pending:
            self._refresh_pending = True
            self.refreshRequested.emit()

    def _refresh(self):
        self._refresh_pending = False
        if self._stopped:
            return
        slot = self._op.FileListTable[0] if len(self._op.FileListTable) else None
        if slot is not self._slot:
            for unsubscribe in self._slot_cleanup:
                unsubscribe()
            self._slot_cleanup = []
            self._slot = slot
            if slot is not None:
                self._slot_cleanup = [
                    slot.notifyReady(self._request_refresh),
                    slot.notifyUnready(self._request_refresh),
                    slot.notifyDirty(self._request_refresh),
                    slot.notifyDisconnect(self._request_refresh),
                ]
        # Fetch the FileList once per update. Painting cells never requests images
        # or traverses the graph, even for large collections.
        rows = list(slot.value.values()) if slot is not None and slot.ready() else []
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def stopAndCleanUp(self):
        self._stopped = True
        for unsubscribe in self._slot_cleanup + self._cleanup:
            unsubscribe()
        self._slot_cleanup.clear()
        self._cleanup.clear()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else FileListColumn.NumColumns

    def get_row(self, row: int) -> FileListDataRow:
        return self._rows[row]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        if role == Qt.UserRole:
            return row.id
        if role == Qt.ToolTipRole:
            return row.filename
        if role != Qt.DisplayRole:
            return None
        column = index.column()
        if column == FileListColumn.Id:
            return row.id
        if column == FileListColumn.Name:
            return Path(row.filename).name
        if column == FileListColumn.Location:
            return row.filename
        if column == FileListColumn.TaggedShape:
            return ", ".join(f"{axis}: {size}" for axis, size in zip(row.axistags.keys(), row.shape))
        if column == FileListColumn.Dtype:
            return row.dtype.name
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Vertical:
            return section + 1
        headers = ("ID", "Name", "Location", "Shape", "Data Type")
        return headers[section] if 0 <= section < len(headers) else None
