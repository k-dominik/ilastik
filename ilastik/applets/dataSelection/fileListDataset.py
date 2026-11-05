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
from typing import Iterable

from lazyflow.operators.ioOperators.functional import imageReader
from lazyflow.operators.ioOperators.types import FileList, FileListDataRowNoId
from lazyflow.utility.grid import OUTPUT_AXIS_KEYS

SUPPORTED_SUFFIXES = (".png", ".jpeg", ".jpg")


def read_file_list(paths: Iterable[str | Path], existing: FileList | None = None) -> FileList:
    """Return a new table, retaining existing IDs and reading only new image headers.

    Validate all extensions before reading anything. The caller's table is never
    mutated, including when reading metadata fails partway through the selection.
    """
    filenames = [str(Path(path).absolute()) for path in paths]
    unsupported = [name for name in filenames if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES]
    if unsupported:
        raise ValueError(f"Unsupported files: {unsupported[:5]}. Supported types: {', '.join(SUPPORTED_SUFFIXES)}.")

    table = FileList(existing, init_count=existing.state()) if existing is not None else FileList()
    known_paths = {row.filename for row in table.values()}
    for filename in filenames:
        if filename in known_paths:
            continue
        meta = imageReader.determine_metadata(filename)
        table.add_no_id(FileListDataRowNoId.from_image_meta_with_axes(meta, axisorder=OUTPUT_AXIS_KEYS))
        known_paths.add(filename)
    return table
