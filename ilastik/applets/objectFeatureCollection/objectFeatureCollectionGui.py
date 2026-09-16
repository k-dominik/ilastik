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
# 		   http://ilastik.org/license.html
###############################################################################
from qtpy.QtCore import QThread
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from ilastik.applets.objectFeatureCollection.opObjectFearturesCollection import OpObjectFeaturesCollection


class ObjectFeatureCollectionGui(LayerViewerGui[OpObjectFeaturesCollection]):
    def __init__(
        self,
        parentApplet,
        topLevelOperatorView: OpObjectFeaturesCollection,
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

    def setupLayers(self):
        mainOperator = self.topLevelOperatorView
        layers = []

        if mainOperator.Output.ready():
            layerraw = super().createStandardLayerFromSlot(mainOperator.Output, name="Raw Data")
            return [layerraw]
        return layers

    def initAppletDrawerUi(self):
        super().initAppletDrawerUi()

        assert self._drawer
        drawer_container: QWidget = self._drawer
        assert isinstance(drawer_container, QWidget)

        mainOperator = self.topLevelOperatorView

        def _cal_embedding(*args, **kwargs):
            class _CalcThread(QThread):

                def run(self):
                    print(f"_cal_embedding {args=} {kwargs=}")
                    _ = mainOperator.Embedding[()].wait()

            t = _CalcThread(parent=self)
            t.start()

        layout = QVBoxLayout()
        label = QLabel("--selected embedding--")

        def _update_selected_embedding(*args, **kwargs):
            print(f"_update_selected_embedding {args=} {kwargs=}")
            embeddings = mainOperator.SelectedPlugin.value
            label.setText(f"embedding: {embeddings}")

        _update_selected_embedding()

        btn = QPushButton("Calculate embeddings")
        btn.clicked.connect(_cal_embedding)
        layout.addWidget(label)
        layout.addWidget(btn)
        layout.addStretch()

        drawer_container.setLayout(layout)
