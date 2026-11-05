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
from lazyflow.operators.ioOperators.types import FileList

from .dataSelectionApplet import DataSelectionApplet
from .fileListDataset import read_file_list
from .opDataSelection import GridDatasetInfo
from .opDataSelectionFileList import OpDataSelectionFileList


class DataSelectionFileListApplet(DataSelectionApplet):
    DEFAULT_INSTRUCTIONS = (
        "Add image files to the collection using the buttons on the top right or drag and drop files/folders."
    )
    operatorClass = OpDataSelectionFileList

    def __init__(
        self,
        workflow,
        title,
        projectFileGroupName,
        supportIlastik05Import=False,
        batchDataGui=False,
        forceAxisOrder=None,
        instructionText=DEFAULT_INSTRUCTIONS,
        show_axis_details=False,
    ):
        if supportIlastik05Import:
            raise ValueError("No way to support 0.5 import here.")
        super().__init__(
            workflow,
            title,
            projectFileGroupName,
            batchDataGui=batchDataGui,
            forceAxisOrder=forceAxisOrder,
            instructionText=instructionText,
            max_lanes=1,
            show_axis_details=show_axis_details,
        )
        self._file_list_counter = 0

    def getMultiLaneGui(self):
        if self._gui is None:
            from .dataSelectionFileListGui import DataSelectionFileListGui

            self._gui = DataSelectionFileListGui(self, self.topLevelOperator, self._instructionText)
        return self._gui

    @property
    def file_list(self) -> FileList:
        if self.num_lanes:
            slot = self.get_lane(0).FileListTable
            if slot.ready():
                table = slot.value
                self._file_list_counter = max(self._file_list_counter, table.state())
                return table
        return FileList(init_count=self._file_list_counter)

    def set_file_list(self, table: FileList):
        if len(self.role_names) != 1:
            raise ValueError("File-list data selection requires exactly one dataset role.")
        if table:
            info = GridDatasetInfo.from_file_list(table)
            if self.num_lanes:
                slot = self.get_lane(0).DatasetGroup[0]
                previous = slot.value if slot.ready() else None
                try:
                    slot.setValue(info)
                except Exception:
                    if previous is None:
                        slot.disconnect()
                    else:
                        slot.setValue(previous)
                    raise
            else:
                self.pushLane({self.role_names[0]: info})
        elif self.num_lanes:
            # Retain the collection lane so downstream connections stay intact.
            self.get_lane(0).DatasetGroup[0].disconnect()
        self._file_list_counter = max(self._file_list_counter, table.state())
        self.appletStateUpdateRequested()

    def add_files(self, paths):
        previous = self.file_list
        table = read_file_list(paths, existing=previous)
        if len(table) != len(previous):
            self.set_file_list(table)

    def remove_files(self, item_ids):
        previous = self.file_list
        removed = set(item_ids)
        table = FileList(init_count=previous.state())
        for item_id, row in previous.items():
            if item_id not in removed:
                table.add(row)
        self.set_file_list(table)

    def create_lane_configs(
        self,
        role_inputs,
        input_axes=(),
        preconvert_stacks=False,
        ignore_training_axistags=False,
        stack_along="grid",
    ):
        if len(self.role_names) != 1:
            raise ValueError("File-list data selection requires exactly one dataset role.")
        if preconvert_stacks or any(input_axes):
            raise ValueError(
                "File-list datasets use image metadata and do not support stack conversion or custom axes."
            )
        table = read_file_list(role_inputs.get(self.role_names[0], []))
        return [{self.role_names[0]: GridDatasetInfo.from_file_list(table)}] if table else []
