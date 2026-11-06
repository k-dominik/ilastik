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
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lazyflow.graph import InputSlot, Operator, OutputSlot
from lazyflow.operators.ioOperators.types import FileList

if TYPE_CHECKING:
    from .opObjectClassificationCollection import PredictionTable


@dataclass(frozen=True)
class CollectionExportTable:
    headers: tuple[str, ...]
    rows: tuple[tuple, ...]
    missing_predictions: tuple[int, ...]

    @property
    def can_export(self):
        return bool(self.rows) and not self.missing_predictions


def merge_collection_predictions(files: FileList, predictions: "PredictionTable", label_names) -> CollectionExportTable:
    num_classes = max(
        len(label_names),
        max((len(predictions[item_id].probabilities) for item_id in files if item_id in predictions), default=0),
    )
    class_names = [
        label_names[i] if i < len(label_names) and label_names[i] else f"Class {i + 1}" for i in range(num_classes)
    ]
    headers = (
        "ID",
        "Name",
        "Location",
        "Shape",
        "Data Type",
        "Prediction",
        "Predicted Class",
        "Uncertainty",
        *(f"Probability {i + 1} ({name})" for i, name in enumerate(class_names)),
    )
    rows = []
    missing = []
    for item_id, file in files.items():
        metadata = (
            item_id,
            Path(file.filename).name,
            file.filename,
            ", ".join(f"{axis}: {size}" for axis, size in zip(file.original_axistags.keys(), file.original_shape)),
            file.dtype.name,
        )
        prediction = predictions.get(item_id)
        if prediction is None:
            missing.append(item_id)
            rows.append(metadata + (None,) * (3 + num_classes))
            continue
        label = prediction.prediction
        class_name = class_names[label - 1] if 0 < label <= len(class_names) else f"Class {label}"
        probabilities = tuple(prediction.probabilities)
        rows.append(
            metadata
            + (label, class_name, prediction.uncertainty_estimate)
            + probabilities
            + (None,) * (num_classes - len(probabilities))
        )
    return CollectionExportTable(headers, tuple(rows), tuple(missing))


class OpObjectClassificationCollectionExport(Operator):
    FileList = InputSlot[FileList]()
    Predictions = InputSlot["PredictionTable"](optional=True)
    LabelNames = InputSlot(stype="object", value=[])
    WorkingDirectory = InputSlot(stype="filestring", optional=True)
    OutputFilename = InputSlot(stype="string", value="collection_predictions.csv")

    Table = OutputSlot[CollectionExportTable](stype="object")

    def setupOutputs(self):
        self.Table.meta.shape = (1,)
        self.Table.meta.dtype = object

    def execute(self, slot, subindex, roi, result):
        predictions = self.Predictions.value if self.Predictions.ready() else {}
        return [merge_collection_predictions(self.FileList.value, predictions, self.LabelNames.value)]

    def propagateDirty(self, slot, subindex, roi):
        if slot in (self.FileList, self.Predictions, self.LabelNames):
            self.Table.setDirty(())

    @property
    def export_path(self) -> Path:
        path = Path(self.OutputFilename.value).expanduser()
        if not path.is_absolute() and self.WorkingDirectory.ready():
            path = Path(self.WorkingDirectory.value) / path
        return path.absolute()

    def run_export(self) -> CollectionExportTable:
        table = self.Table.value
        if not table.rows:
            raise ValueError("There are no files to export.")
        if table.missing_predictions:
            raise ValueError(
                f"{len(table.missing_predictions)} files have no predictions. "
                "Calculate embeddings and train a classifier before exporting."
            )
        path = self.export_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(table.headers)
            writer.writerows(table.rows)
        return table
