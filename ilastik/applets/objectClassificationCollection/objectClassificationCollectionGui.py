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
import enum
import logging
import traceback
import weakref
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

import numpy
import pyqtgraph
import volumina.colortables as colortables
from qtpy.QtCore import QRectF, Qt, QThread, Signal
from qtpy.QtGui import QAction, QBrush, QColor, QIcon, QMouseEvent, QShowEvent
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGraphicsRectItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QUndoCommand,
    QVBoxLayout,
    QWidget,
)
from volumina.api import ColortableLayer, LazyflowSinkSource
from volumina.interpreter import ClickInterpreter
from volumina.pixelpipeline.datasources.factories import createDataSource

from ilastik.applets.labeling.labelingGui import LabelingGui, LabelingSlots
from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from ilastik.applets.objectClassification.opObjectClassification import InvalidObjectIndex
from ilastik.shell.gui.iconMgr import ilastikIcons
from ilastik.utility.bind import bind
from ilastik.utility.gui import ThreadRouter, threadRouted
from lazyflow.base import ItemId
from lazyflow.cancel_token import CancellationTokenSource
from lazyflow.slot import InputSlot, Slot, valueContext

from .opObjectClassificationCollection import EmbeddingSource, PredictionRow
from .types import AdaptionParameters, ProjectorData

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ilastik.widgets.labelListModel import Label

    from .opObjectClassificationCollection import OpLabelView, OpOCC


def _listReplace(old, new):
    if len(old) > len(new):
        return new + old[len(new) :]
    else:
        return new


@dataclass
class SelectionInformation:
    shift_down: bool
    selection: list[int]


EFFORT_DICT = {
    0: AdaptionParameters(n_unlabeled=64, labeled_fraction=0.25),
    1: AdaptionParameters(n_unlabeled=256, labeled_fraction=0.25),
    2: AdaptionParameters(n_unlabeled=768, labeled_fraction=0.25),
}


class SelectionViewBox(pyqtgraph.ViewBox):
    selectionFinished = Signal(SelectionInformation)

    def __init__(self, scatter):
        super().__init__(enableMenu=False)
        self.scatter = scatter

        self.origin = None

        self.rectItem = QGraphicsRectItem()
        self.rectItem.setPen(pyqtgraph.mkPen(color="#27F5CF", width=2))
        self.rectItem.setBrush(pyqtgraph.mkBrush(20, 210, 200, 30))
        self.rectItem.hide()
        self.addItem(self.rectItem)

    def mouseDragEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return

        event.accept()

        if event.isStart():
            self.origin = self.mapSceneToView(event.buttonDownScenePos())
            self.rectItem.show()

        current = self.mapSceneToView(event.scenePos())
        rect = QRectF(self.origin, current).normalized()
        self.rectItem.setRect(rect)

        if event.isFinish():
            self.rectItem.hide()
            xs, ys = self.scatter.getData()
            mask = (xs >= rect.left()) & (xs <= rect.right()) & (ys >= rect.top()) & (ys <= rect.bottom())
            self.selectionFinished.emit(
                SelectionInformation(
                    shift_down=bool(event.modifiers() & Qt.ShiftModifier), selection=numpy.where(mask)[0].tolist()
                )
            )


class ScatterWidget(QWidget):
    brush_color = "#AAAAAA"

    def __init__(self, tlo: "OpOCC", parent=None):
        super().__init__(parent)

        self._full_table = tlo.Embedding.value
        # TODO: make sure to update this if Embeddings change
        self._indices = list(self._full_table.keys())
        self._values = list(x.embedding_vector for x in self._full_table.values())
        self._tlo = tlo
        self.scatter: pyqtgraph.ScatterPlotItem | None = None
        self.label_scatter: pyqtgraph.ScatterPlotItem | None = None
        self.plot = None
        self._last_flat_indices = None
        self._embeddings = {}
        self._umap: numpy.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        combo_layout_coloring = QHBoxLayout(self)
        combo_layout_coloring.addWidget(QLabel("Color by"))
        self.wants_label_colors = QCheckBox("labels")
        self.wants_prediction_colors = QCheckBox("predictions")
        button_group_coloring = QButtonGroup(self)
        button_group_coloring.setExclusive(False)
        button_group_coloring.addButton(self.wants_label_colors)
        button_group_coloring.addButton(self.wants_prediction_colors)
        radio_layout_coloring = QHBoxLayout(self)
        radio_layout_coloring.addWidget(self.wants_label_colors)
        radio_layout_coloring.addWidget(self.wants_prediction_colors)
        combo_layout_coloring.addStretch()
        combo_layout_coloring.addLayout(radio_layout_coloring)
        self.wants_label_colors.setChecked(True)
        button_group_coloring.buttonToggled.connect(self._recolor)
        layout.addLayout(combo_layout_coloring)

        calculate_embeddings_layout = QHBoxLayout(self)
        emb_button = QPushButton("Show")

        emb_button.clicked.connect(self._request_umap_data)
        calculate_embeddings_layout.addWidget(emb_button)

        layout.addLayout(calculate_embeddings_layout)

        self._graphics = pyqtgraph.GraphicsLayoutWidget()
        layout.addWidget(self._graphics)
        self.setLayout(layout)

        self._tlo.AnnotationsTable.notifyDirty(self._recolor)
        self._tlo.CachedPredictions.notifyDirty(self._recolor)
        self._tlo.PmapColors.notifyDirty(self._recolor)
        self._tlo.LabelColors.notifyDirty(self._recolor)
        self._tlo.SelectEmbeddingSource.notifyDirty(self._request_umap_data)

    def showEvent(self, a0: QShowEvent) -> None:
        ret = super().showEvent(a0)
        self._request_umap_data()
        return ret

    def _request_umap_data(self, *args, **kwargs):

        tlo = self._tlo

        class _CalcThread(QThread):
            umapDone = Signal(object)
            error = Signal(str)

            def run(self):
                try:
                    # HACK: for now go deep
                    if (
                        tlo.SelectEmbeddingSource.value == EmbeddingSource.Adapted
                        and not tlo.projector_cache.hasCacheValue()
                    ):
                        umap_result = None
                    else:
                        umap_result = numpy.array([(r.x, r.y) for r in tlo.UmapForDisplay.value.values()])
                    self.umapDone.emit(umap_result)
                except Exception:
                    self.error.emit(traceback.format_exc())
                finally:
                    self.finished.emit()

        t = _CalcThread(parent=self)
        t.umapDone.connect(self._update_embeddings)
        t.error.connect(lambda x: logger.error(x))
        t.start()

    def _update_embeddings(self, umap_result: numpy.ndarray | None):
        if self.plot:
            self._graphics.removeItem(self.plot)
            self.plot = None
        if umap_result is None:
            return

        x, y = umap_result[:, 0], umap_result[:, 1]

        self.scatter = pyqtgraph.ScatterPlotItem(
            x=x,
            y=y,
            pen=None,
            brush=self.brush_color,
            size=8,
        )

        self.viewBox = None
        self.viewBox = SelectionViewBox(self.scatter)
        self.plot = pyqtgraph.PlotItem(viewBox=self.viewBox)
        self.plot.addItem(self.scatter)
        self._graphics.addItem(self.plot)
        self._umap = umap_result
        self._recolor()
        self.onSelection()
        self.viewBox.selectionFinished.connect(self.onSelection)

    def _base_brushes(self) -> list[QBrush]:
        """self.brush_color or prediction color if checked"""
        if self.scatter is None or not self._indices:
            return []

        if self.wants_prediction_colors.isChecked() and self._tlo.CachedPredictions.ready():
            brushes = self._prediction_brushes()
            if not brushes:
                brushes = [pyqtgraph.mkBrush(self.brush_color) for id in self._indices]
        else:
            brushes = [pyqtgraph.mkBrush(self.brush_color) for id in self._indices]

        return brushes

    def _prediction_brushes(self) -> list[QBrush]:
        if self.scatter is None:
            return []

        predictions: dict[ItemId, PredictionRow] = self._tlo.CachedPredictions.value

        if not predictions:
            return []

        pmap_colors = [QColor(*v).lighter() for v in self._tlo.PmapColors.value]

        colors = [pmap_colors[predictions[id].prediction - 1] for id in self._indices]
        brushes = [pyqtgraph.mkBrush(color) for color in colors]
        return brushes

    def _add_annotations_brushes(self) -> None:
        if self.label_scatter:
            try:
                self.plot.removeItem(self.label_scatter)
            except:
                pass
            del self.label_scatter
            self.label_scatter = None
        if not self.wants_label_colors.isChecked():
            return

        labels_dict = self._tlo.AnnotationsTable.value

        indices = [(x.id, x.label) for x in labels_dict.values() if x.label != 0]
        flat_indices = [self._indices.index(x[0]) for x in indices]

        assert self._umap is not None
        x, y = self._umap[:, 0], self._umap[:, 1]

        label_colors = self._tlo.LabelColors.value
        brushes = [label_colors[idx_label[1] - 1] for idx_label in indices]

        scatter = pyqtgraph.ScatterPlotItem(
            x=x[flat_indices],
            y=y[flat_indices],
            pen=None,
            brush=brushes,
            size=8,
        )
        self.plot.addItem(scatter)
        self.label_scatter = scatter

    def _recolor(self, *args, **kwargs):
        if self.scatter is None:
            return
        self.scatter.setBrush(self._base_brushes())
        self._add_annotations_brushes()

    def _to_object_ids(self, flat_indices):
        object_ids = [self._indices[ii] for ii in flat_indices]
        return object_ids

    def onSelection(self, selection_information: SelectionInformation | None = None):
        if selection_information is None:
            if self._last_flat_indices is None:
                self._last_flat_indices = []
        else:
            if selection_information.shift_down:
                self._last_flat_indices = self._last_flat_indices + selection_information.selection
            else:
                self._last_flat_indices = selection_information.selection

        n = len(self.scatter.points())

        pens = [pyqtgraph.mkPen(None)] * n

        for i in self._last_flat_indices:
            pens[i] = pyqtgraph.mkPen("w", width=2)

        self.scatter.setPen(pens)
        indices = self._to_object_ids(self._last_flat_indices)
        self._tlo.SubsetObjects.setValue(indices)


class SimpleLabelView(LayerViewerGui["OpLabelView"]):

    def __init__(
        self,
        parentApplet,
        topLevelOperatorView: "OpLabelView",
        labelColorsSlot: Slot[list[QColor]],
        additionalMonitoredSlots=[],
        centralWidgetOnly=False,
        crosshair=True,
        is_3d_widget_visible=False,
    ):
        super().__init__(
            parentApplet,
            topLevelOperatorView,
            additionalMonitoredSlots,
            centralWidgetOnly,
            crosshair,
            is_3d_widget_visible,
        )
        self._labelColorsSlot = labelColorsSlot
        self._color_table = [
            0,
        ] + [QColor(*x).rgba() for x in self._labelColorsSlot.value]

        def _update_colors(*args):
            self._color_table = [
                0,
            ] + [QColor(*x).rgba() for x in self._labelColorsSlot.value]
            if label_layer := self.getLayerByName("Labels"):
                label_layer.colorTable = self._color_table

        self._labelColorsSlot.notifyDirty(_update_colors)

    def setupLayers(self):
        layers = []
        raw = self.topLevelOperatorView.GridImage
        if any(x == 0 for x in raw.meta.shape):
            return []

        labelOutput = self.topLevelOperatorView.GridLabels
        if not labelOutput.ready():
            return []
        else:
            labellayer = ColortableLayer(createDataSource(labelOutput), colorTable=self._color_table, direct=False)

            labellayer.name = "Labels"
            labellayer.ref_object = None
            labellayer.zeroIsTransparent = True
            labellayer.colortableIsRandom = True
            labellayer.opacity = 1.0
            layers.append(labellayer)

        if raw.ready():
            raw_layer = self.createStandardLayerFromSlot(raw, name="raw data")
            layers.append(raw_layer)

        return layers


def has_labels(slot):
    if not slot.ready():
        return False

    labels_table = slot.value
    if any(x.label != 0 for x in labels_table.values()):
        return True


class EmbeddingWidget(QWidget):

    display_text = "Embedding View"

    def __init__(self, tlo: "OpOCC", parent: Optional[QWidget]):
        super().__init__(parent)

        self._scatter_widget = ScatterWidget(tlo, self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        splitter = QSplitter()
        splitter.setOrientation(Qt.Vertical)
        splitter.addWidget(self._scatter_widget)
        placeholder = QWidget()
        placeholder.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        # splitter.addWidget(placeholder)
        vol = SimpleLabelView(None, tlo.label_view_op, centralWidgetOnly=True, labelColorsSlot=tlo.LabelColors)
        splitter.addWidget(vol)
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.setContentsMargins(0, 0, 0, 0)


class LabelObjectCommand(QUndoCommand):
    """Redo/Undo for object labelings

    Note: the undo is not 100% true: In object classification the labeling dictionary
    is somewhat sparse - it will only contain entries up to the labeled object with
    the highest object id. The undo here will not revert the potential increase
    in dictionary size.
    Clearing a label in object classification will also not touch the length of
    this label array, so that's in a way consistent.
    """

    def __init__(
        self,
        parent=None,
        *,
        slot: InputSlot,
        old_value: int,
        new_value: int,
        object_id,
    ):
        super().__init__(parent)
        self.__old_value = old_value
        self.__new_value = new_value
        self.__object_id = object_id
        self.__slot = slot

    def _update_label(self, value):
        self.__slot[self.__object_id] = value
        self.__slot.setDirty(self.__object_id)

    def redo(self):
        self._update_label(self.__new_value)

    def undo(self):
        self._update_label(self.__old_value)


class ProgressButton(QPushButton):
    start_idle: Signal = Signal()
    start_in_progress: Signal = Signal()
    start_waiting = Signal()

    class State(enum.IntEnum):
        IDLE = enum.auto()
        PROGRESS = enum.auto()
        WAITING = enum.auto()

    def __init__(self, idle_text: str, in_progress_text: str, waiting_text: str, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self._state = self.State.IDLE

        self._text: dict["ProgressButton.State", str] = {
            self.State.IDLE: idle_text,
            self.State.PROGRESS: in_progress_text,
            self.State.WAITING: waiting_text,
        }

        self._signal_transition: dict["ProgressButton.State", Signal] = {
            self.State.IDLE: self.start_idle,
            self.State.PROGRESS: self.start_in_progress,
            self.State.WAITING: self.start_waiting,
        }
        self.clicked.connect(self.progress_state)
        self._update_text()

    def _emit(self, state):
        self._signal_transition[state].emit()

    def progress_state(self):
        if self._state == self.State.IDLE:
            self.set_state(self.State.PROGRESS)
        elif self._state == self.State.PROGRESS:
            self.set_state(self.State.WAITING)
        elif self._state == self.State.WAITING:
            # ignore clicks - should be disabled anyway
            pass

    def set_state(self, state: "ProgressButton.State"):
        self._state = state
        self._update_text()
        self._emit(state)
        if state == self.State.WAITING:
            self.setEnabled(False)
        else:
            self.setEnabled(True)

    def _update_text(self):
        self.setText(self._text[self._state])


class HorizontalDivider(QFrame):
    def __init__(
        self, parent: Optional[QWidget] = None, flags: Union[Qt.WindowFlags, Qt.WindowType] = Qt.WindowFlags()
    ) -> None:
        super().__init__(parent, flags)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class ObjectClassificationCollectionGui(LabelingGui["OpOCC"]):
    """A subclass of LabelingGui for labeling objects.

    Handles labeling objects, viewing the predicted results, and
    displaying warnings from the top level operator. Also provides a
    dialog for choosing subsets of the precalculated features provided
    by the object extraction applet.

    """

    def centralWidget(self):
        return self

    def secondaryControlsWidget(self) -> None:
        return self._secondary_controls

    def stopAndCleanUp(self):
        # Unsubscribe to all signals
        for fn in self.__cleanup_fns:
            fn()

        # Base class
        super().stopAndCleanUp()

    PREDICTION_LAYER_NAME = "Prediction"

    def __init__(self, parentApplet, op: "OpOCC"):
        self.parentApplet = parentApplet
        self.isInitialized = False
        self.__cleanup_fns = []
        # Tell our base class which slots to monitor
        labelSlots = LabelingSlots(
            labelInput=op.LabelInputs,
            labelOutput=op.GridLabels,
            labelEraserValue=op.Eraser,
            labelDelete=op.DeleteLabel,
            labelNames=op.LabelNames,
        )
        self._interactiveMode = False
        self._labelMode = True

        self.op = op
        self.applet = parentApplet

        super().__init__(
            parentApplet,
            labelingSlots=labelSlots,
            topLevelOperatorView=op,
            drawerUiPath=None,
            rawInputSlot=op.GridImage,
            crosshair=False,
        )
        self.interactiveMode = False
        self.threadRouter = ThreadRouter(self)
        self._retained_weakrefs = []
        self.forceAtLeastTwoLabels(True)

    def initAppletDrawerUi(self):
        super().initAppletDrawerUi()
        mainOperator = self.topLevelOperatorView
        # unused
        self.labelingDrawerUi.brushSizeComboBox.setEnabled(False)
        self.labelingDrawerUi.brushSizeComboBox.setVisible(False)
        self.labelingDrawerUi.brushSizeCaption.setVisible(False)

        self._colorTable_forpmaps = list(colortables.default16_new)
        self._undoStack = self.editor._undoStack
        self._live_update_button = QToolButton()
        self._live_update_button.setText("Live Update")
        self._live_update_button.setCheckable(True)
        self._live_update_button.setEnabled(True)
        self._live_update_button.setIcon(QIcon(ilastikIcons.Play))
        self._live_update_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._live_update_button.toggled.connect(self.handleInteractiveModeClicked)
        self._labelControlUi.horizontalLayout.addWidget(self._live_update_button)
        self._labelControlUi.verticalLayout.addWidget(HorizontalDivider())

        wants_original_embedding = QCheckBox("original")
        wants_adapted_embedding = QCheckBox("adapted")
        button_group_umap_source = QButtonGroup(self)
        button_group_umap_source.setExclusive(True)
        button_group_umap_source.addButton(wants_original_embedding, id=EmbeddingSource.Original)
        button_group_umap_source.addButton(wants_adapted_embedding, id=EmbeddingSource.Adapted)
        radio_layout_embedding = QHBoxLayout(self)
        radio_layout_embedding.addWidget(wants_original_embedding)
        radio_layout_embedding.addWidget(wants_adapted_embedding)

        if not self.topLevelOperatorView.SelectEmbeddingSource.ready():
            self.topLevelOperatorView.SelectEmbeddingSource.setValue(EmbeddingSource.Original)
        self._button_group_umap_source = button_group_umap_source
        button_group_umap_source.buttonToggled.connect(self._update_embedding_source)
        button_group_umap_source.button(self.topLevelOperatorView.SelectEmbeddingSource.value).setChecked(True)
        self._wants_adapted_embedding_checkbox = wants_adapted_embedding
        self._wants_adapted_embedding_checkbox.setEnabled(False)
        self._wants_original_embedding_checkbox = wants_original_embedding

        def _update_chk(*args, **kwargs):
            if self.topLevelOperatorView.projector_cache.hasCacheValue():
                self._wants_adapted_embedding_checkbox.setEnabled(True)
            else:
                self._wants_adapted_embedding_checkbox.setEnabled(False)
                self._wants_original_embedding_checkbox.setChecked(True)

        self.topLevelOperatorView.FreezeProjector.notifyDirty(_update_chk)
        # self._labelControlUi.verticalLayout.addLayout(combo_layout_embedding)

        self._secondary_controls = EmbeddingWidget(mainOperator, self)

        btn = ProgressButton("Adapt Features", "Cancel", "Cancelling")
        self._cancellation_token_source = None

        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel("Embedding"), 0, 0, 1, 1)
        grid_layout.addLayout(radio_layout_embedding, 0, 1, 1, 1)
        grid_layout.addWidget(QLabel("Effort"), 1, 0, 1, 1)

        slider_layout = QVBoxLayout()
        self._effort_slider = QSlider(orientation=Qt.Horizontal)
        self._effort_slider.setRange(0, 2)
        self._effort_slider.setTickInterval(1)
        slider_layout.addWidget(self._effort_slider)

        effort_labels_layout = QHBoxLayout()
        effort_labels_layout.addWidget(QLabel("low"))
        high_label = QLabel("high")
        high_label.setAlignment(Qt.AlignRight)
        effort_labels_layout.addWidget(high_label)
        slider_layout.addLayout(effort_labels_layout)
        grid_layout.addLayout(slider_layout, 1, 1, 1, 1)
        grid_layout.addWidget(btn, 2, 0, 1, 2)

        self.topLevelOperatorView.AdaptionParameters.setValue(EFFORT_DICT[0])

        def cancel(*args):
            assert self._cancellation_token_source
            self._cancellation_token_source.cancel()

        btn.start_waiting.connect(cancel)

        def finalize(*args):
            self._cancellation_token_source.cancel()
            self._cancellation_token_source = None
            btn.set_state(btn.State.IDLE)

        def _cal_embedding(*args, **kwargs):
            if self._cancellation_token_source is not None:
                # still in progress I guess
                print("Should not happen - still in progress")
                return

            effort = self._effort_slider.value()

            self.topLevelOperatorView.AdaptionParameters.setValue(EFFORT_DICT[effort])

            class _CalcThread(QThread):

                def run(self):
                    print(f"_fine_tuning {args=} {kwargs=}")
                    with valueContext(mainOperator.FreezeProjector, False):
                        _ = mainOperator.AdaptedProjectorData.value

            self._cancellation_token_source = CancellationTokenSource()
            token = self._cancellation_token_source.token

            if mainOperator.AdaptedProjectorData.ready():
                mainOperator.cancellation_token(token)
                t = _CalcThread(parent=self)
                t.start()
                t.finished.connect(finalize)
            else:
                print("waiting for labels")
                finalize()

        btn.start_in_progress.connect(_cal_embedding)

        drawer = self.appletDrawer()
        drawer.verticalLayout.addLayout(grid_layout)
        drawer.verticalLayout.addStretch()

    @property
    def labelMode(self):
        return self._labelMode

    @labelMode.setter
    def labelMode(self, val):
        self._labelMode = val

    ### Function dealing with label name and color consistency
    def _getNext(self, slot, parentFun, transform=None):
        numLabels = self.labelListData.rowCount()
        value = slot.value
        if numLabels < len(value):
            result = value[numLabels]
            if transform is not None:
                result = transform(result)
            return result
        else:
            return parentFun()

    def _update_embedding_source(self):
        selected_embedding = EmbeddingSource(self._button_group_umap_source.checkedId())
        self.topLevelOperatorView.SelectEmbeddingSource.setValue(selected_embedding)

    def _onLabelChanged(self, parentFun, mapf, slot):
        parentFun()
        new = list(map(mapf, self.labelListData))
        old = slot.value
        slot.setValue(_listReplace(old, new))

    def createLabelLayer(self, direct=False):
        """Return a colortable layer that displays the label slot
        data, along with its associated label source.

        direct: whether this layer is drawn synchronously by volumina

        """
        labelInput = self._labelingSlots.labelInput
        labelOutput = self._labelingSlots.labelOutput
        if not labelOutput.ready():
            return (None, None)
        else:

            labelsrc = LazyflowSinkSource(labelOutput, labelInput)
            labellayer = ColortableLayer(labelsrc, colorTable=self._colorTable16, direct=direct)
            labellayer.name = "Labels"
            labellayer.ref_object = None
            labellayer.zeroIsTransparent = True
            labellayer.colortableIsRandom = True
            labellayer.opacity = 1.0

            clickInt = ClickInterpreter(self.editor, labellayer, self.onClick, right=False, double=False)
            self.editor.brushingInterpreter = clickInt
            return labellayer, labelsrc

    def setupLayers(self):
        # Base class provides the label layer and the raw layer
        layers = super().setupLayers()

        predictionSlot = self.op.GridPredictions
        # from PyQt5.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook;pyqtRemoveInputHook();breakpoint()
        if predictionSlot.ready():
            predictsrc = createDataSource(predictionSlot)
            self._colorTable_forpmaps[0] = 0
            predictLayer = ColortableLayer(predictsrc, colorTable=self._colorTable_forpmaps)

            predictLayer.name = self.PREDICTION_LAYER_NAME
            predictLayer.ref_object = None
            predictLayer.opacity = 0.25
            predictLayer.setToolTip("Classification results, assigning a label to each object")

            # This weakref stuff is a little more fancy than strictly necessary.
            # The idea is to use the weakref's callback to determine when this layer instance is destroyed by the garbage collector,
            #  and then we disconnect the signal that updates that layer.
            weak_predictLayer = weakref.ref(predictLayer)
            colortable_changed_callback = bind(self._setPredictionColorTable, weak_predictLayer)
            self._labelControlUi.labelListModel.dataChanged.connect(colortable_changed_callback)
            weak_predictLayer2 = weakref.ref(
                predictLayer, partial(self._disconnect_dataChange_callback, colortable_changed_callback)
            )
            # We have to make sure the weakref isn't destroyed because it is responsible for calling the callback.
            # Therefore, we retain it by adding it to a list.
            self._retained_weakrefs.append(weak_predictLayer2)

            # Ensure we're up-to-date (in case this is the first time the prediction layer is being added.
            for row in range(self._labelControlUi.labelListModel.rowCount()):
                self._setPredictionColorTableForRow(predictLayer, row)

            # put right after Labels, so that it is visible after hitting "live
            # predict".
            layers.insert(1, predictLayer)

        return layers

    def _disconnect_dataChange_callback(self, colortable_changed_callback, *args):
        """
        When instances of the prediction layer are garbage collected, we no longer want the list model to call them back.
        This function disconnects the signal that was connected in setupLayers, above.
        """
        self._labelControlUi.labelListModel.dataChanged.disconnect(colortable_changed_callback)

    def _setPredictionColorTable(self, weak_predictLayer, index1, index2):
        predictLayer = weak_predictLayer()
        if predictLayer is None:
            return
        row = index1.row()
        self._setPredictionColorTableForRow(predictLayer, row)

    def _setPredictionColorTableForRow(self, predictLayer, row):

        if row >= 0 and row < self._labelControlUi.labelListModel.rowCount():
            element = self._labelControlUi.labelListModel[row]
            try:
                oldcolor = self._colorTable_forpmaps[row + 1]
            except IndexError:
                self._colorTable_forpmaps.append(element.pmapColor().rgba())
                predictLayer.colorTable = self._colorTable_forpmaps
                return

            if oldcolor != element.pmapColor().rgba():
                self._colorTable_forpmaps[row + 1] = element.pmapColor().rgba()
                predictLayer.colorTable = self._colorTable_forpmaps

    def _updateObjLabel(self, pos5d, label):
        try:
            old_label, object_id = self.topLevelOperatorView.prepareObjectLabels(pos5d)
            print(f"{old_label=}, {object_id=}")
        except InvalidObjectIndex:
            return

        if old_label == label:
            label = 0

        self._undoStack.push(
            LabelObjectCommand(
                slot=self.topLevelOperatorView.LabelInputs,
                old_value=old_label,
                new_value=label,
                object_id=object_id,
            )
        )

    def onClick(self, layer, pos5d, pos):
        """Extracts the object index that was clicked on and updates
        that object's label.

        """
        label = self.editor.brushingModel.drawnNumber
        if label == self.editor.brushingModel.erasingNumber:
            label = 0

        topLevelOp = self.topLevelOperatorView

        self._updateObjLabel(pos5d, label)

    def _object_id_at(self, pos5d):
        return self.topLevelOperatorView.object_at_coordinate(pos5d)

    def handleEditorRightClick(self, position5d, globalWindowCoordinate):
        obj_id = self._object_id_at(position5d)
        if obj_id == -1:
            return

        menu = QMenu(self)

        clearlabel = f"Clear label for object {obj_id}"
        clear_action = menu.addAction(clearlabel)
        numLabels = self.labelListData.rowCount()
        label_actions = []
        for l in range(numLabels):
            color_icon = self.labelListData.createIconForLabel(l)
            act_text = f'Label object {obj_id} as "{self.labelListData[l].name}"'
            act = QAction(color_icon, act_text, menu)
            act.setIconVisibleInMenu(True)
            label_actions.append(act)
            menu.addAction(act)

        action = menu.exec_(globalWindowCoordinate)

        if action == clear_action:
            topLevelOp = self.topLevelOperatorView
            self._updateObjLabel(position5d, 0)
        elif action in label_actions:
            label = label_actions.index(action)
            topLevelOp = self.topLevelOperatorView
            self._updateObjLabel(position5d, label + 1)

    @property
    def interactiveMode(self):
        return self._interactiveMode

    @interactiveMode.setter
    def interactiveMode(self, val):
        self._interactiveMode = val
        self._live_update_button.setChecked(val)
        if val:
            self.showPredictions = True
            self._live_update_button.setIcon(QIcon(ilastikIcons.Pause))
        else:
            self._live_update_button.setIcon(QIcon(ilastikIcons.Play))

        self.labelMode = not val
        self.op.FreezePredictions.setValue(not val)
        self.parentApplet.appletStateUpdateRequested()

    def handleInteractiveModeClicked(self):
        self.interactiveMode = self._live_update_button.isChecked()

    def _getNext(
        self,
        slot: Slot[list[Any]],
        parentFun: Callable[[], str | QColor | None],
        transform: Optional[Callable[[str | QColor], QColor | str]] = None,
    ):
        numLabels = cast(int, self.labelListData.rowCount())

        value = slot.value
        if numLabels < len(value):
            result = value[numLabels]
            if transform is not None:
                result = transform(result)
            return result
        else:
            return parentFun()

    def _onLabelRemoved(self, parent, start: int, end: int):
        super()._onLabelRemoved(parent, start, end)
        op = self.topLevelOperatorView
        op.removeLabel(start)

    def _onLabelChanged(
        self,
        parentFun: Callable[[], None],
        mapf: Callable[["Label"], str | QColor],
        slot: Slot[list[str] | list[QColor]],
    ):
        parentFun()
        new = list(map(mapf, self.labelListData))
        old = slot.value
        slot.setValue(_listReplace(old, new))

    def getNextLabelName(self):
        return self._getNext(self.topLevelOperatorView.LabelNames, super().getNextLabelName)

    def getNextLabelColor(self):
        return self._getNext(
            self.topLevelOperatorView.LabelColors,
            super().getNextLabelColor,
            lambda x: QColor(*x),
        )

    def getNextPmapColor(self):
        return self._getNext(
            self.topLevelOperatorView.PmapColors,
            super().getNextPmapColor,
            lambda x: QColor(*x),
        )

    def onLabelNameChanged(self):
        self._onLabelChanged(
            super().onLabelNameChanged,
            lambda l: l.name,
            self.topLevelOperatorView.LabelNames,
        )

    def onLabelColorChanged(self):
        self._onLabelChanged(
            super().onLabelColorChanged,
            lambda l: (l.brushColor().red(), l.brushColor().green(), l.brushColor().blue()),
            self.topLevelOperatorView.LabelColors,
        )

    def onPmapColorChanged(self):
        self._onLabelChanged(
            super().onPmapColorChanged,
            lambda l: (l.pmapColor().red(), l.pmapColor().green(), l.pmapColor().blue()),
            self.topLevelOperatorView.PmapColors,
        )
