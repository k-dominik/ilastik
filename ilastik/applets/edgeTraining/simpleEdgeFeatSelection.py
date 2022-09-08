import enum
from functools import partial
from typing import Dict, List, Set

from ilastikrag.gui import FeatureSelectionDialog
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ilastik.widgets.CollapsibleWidget import CollapsibleWidget


class SimpleEdgeFeatureSelection(QDialog):
    @enum.unique
    class FeatureGroup(enum.Enum):
        shape = "shape"
        boundary_edge = "boundary channel along edges"
        boundary_sp = "boundary channel on superpixels"
        raw_edge = "raw data along edges"
        raw_sp = "raw data on superpixels"

    openAdvanced = pyqtSignal()

    def __init__(
        self, raw_channels, boundary_channels, probability_channels, selection, supported_features, parent=None
    ):
        super().__init__(parent)
        layout = QVBoxLayout()

        self.setWindowTitle("Edge Feature Selection")
        self.raw_channels = raw_channels
        self.boundary_channels = boundary_channels
        self.probability_channels = probability_channels
        self.channel_names = raw_channels + boundary_channels + probability_channels

        self.feat_vals = self.default_features(raw_channels, boundary_channels)

        # for the "big" feature selection dialog
        self.supported_features = supported_features
        # self.default_features = []

        if selection:
            checks = self._checkmarks(self.feat_vals, selection)
            for check, val in checks.items():
                self.feat_vals[check]["state"] = val

        def make_checkbox(name):
            checkbox = QCheckBox(name.value)
            details = self.feat_vals[name]
            checkbox.setCheckState(Qt.Checked if details["state"] else Qt.Unchecked)
            checkbox.stateChanged.connect(partial(self._update_state, name))
            checkbox.setToolTip(details["description"])
            return checkbox

        self.checkboxes = {}

        shapeGroupBox = QGroupBox("Shape")
        shapeLayout = QVBoxLayout()
        self.checkboxes[self.FeatureGroup.shape] = make_checkbox(self.FeatureGroup.shape)
        shapeLabel = QLabel(
            "Shape features take into account the shape of the superpixels. "
            "These include the length/area, as well as an estimate of radii of an "
            "ellipse/ellipsoid fitted to each edge."
        )
        shapeLabel.setWordWrap(True)

        shapeLayout.addWidget(self.checkboxes[self.FeatureGroup.shape])
        shapeLayout.addWidget(CollapsibleWidget(shapeLabel))
        shapeGroupBox.setLayout(shapeLayout)
        self.shapeGroupBox = shapeGroupBox

        layout.addWidget(shapeGroupBox)

        intensityGroupBox = QGroupBox("Intensity statistics")
        intensityLayout = QVBoxLayout()
        self.checkboxes[self.FeatureGroup.boundary_edge] = make_checkbox(self.FeatureGroup.boundary_edge)
        self.checkboxes[self.FeatureGroup.boundary_sp] = make_checkbox(self.FeatureGroup.boundary_sp)
        intensityLayout.addWidget(self.checkboxes[self.FeatureGroup.boundary_edge])
        intensityLayout.addWidget(self.checkboxes[self.FeatureGroup.boundary_sp])

        self.checkboxes[self.FeatureGroup.raw_edge] = make_checkbox(self.FeatureGroup.raw_edge)
        self.checkboxes[self.FeatureGroup.raw_sp] = make_checkbox(self.FeatureGroup.raw_sp)
        intensityLayout.addWidget(self.checkboxes[self.FeatureGroup.raw_edge])
        intensityLayout.addWidget(self.checkboxes[self.FeatureGroup.raw_sp])
        intensityLabel = QLabel(
            "Intensity statistics are computed along either edges or superpixel area/volume. "
            "Quantities computed include 10th and 90th quantile, and mean intensity. "
            "For raw data these quantities are computed per channel."
        )
        intensityLabel.setWordWrap(True)
        intensityLayout.addWidget(CollapsibleWidget(intensityLabel))
        intensityGroupBox.setLayout(intensityLayout)
        self.intensityGroupBox = intensityGroupBox
        layout.addWidget(intensityGroupBox)

        buttonbox = QDialogButtonBox(Qt.Horizontal)
        buttonbox.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

        advButton = QPushButton("advanced")
        advButton.clicked.connect(self.openAdvancedDlg)
        buttonbox.addButton(advButton, QDialogButtonBox.ActionRole)

        resetButton = QPushButton("reset")
        resetButton.clicked.connect(self.reset_to_default)
        buttonbox.addButton(resetButton, QDialogButtonBox.ResetRole)

        layout.addWidget(buttonbox)

        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        if not self.check_selection_compatible_with_dlg_features(self.feat_vals, selection):
            self.setCheckboxesEnabled(False)

    def setCheckboxesEnabled(self, state: bool):
        """
        User has selected something in the advanced dialog, disable all
        checkboxes.
        """
        self.intensityGroupBox.setEnabled(state)
        self.shapeGroupBox.setEnabled(state)

    def _update_state(self, name, state):
        self.feat_vals[name]["state"] = bool(state)

    def _get_states(self):
        return self.feat_vals

    def selections(self) -> Dict[str, List[str]]:
        """
        feature selection compatible with operator/ilastikrag.gui.feature_selection_dialog

        returns:
          dict of currently selected features (values) per channel name (keys)
        """
        dlg_features = self.feat_vals

        selected_features: Dict[str, Set[str]] = {chan: set() for chan in self.channel_names}
        for group_features in dlg_features.values():
            if group_features["state"]:
                for channel, features in group_features["features"].items():
                    selected_features[channel] |= set(features)

        return {k: list(v) for k, v in selected_features.items()}

    def check_selection_compatible_with_dlg_features(self, default_features, selection) -> bool:
        for group, group_features in default_features.items():
            # make sure that groups are not _partly_ selected
            for chan, features in group_features["features"].items():
                overlap_sum = sum(x in features for x in selection)
                if overlap_sum not in [0, len(features)]:
                    return False
        return True

    def reset_to_default(self):
        self.feat_vals = self.default_features(self.raw_channels, self.boundary_channels)

        for k, v in self.feat_vals.items():
            self.checkboxes[k].setCheckState(QCheckBox.Checked if v["state"] else QCheckBox.Unchecked)

    def _checkmarks(self, default_features, selection) -> Dict[str, bool]:
        if not self.check_selection_compatible_with_dlg_features(default_features, selection):
            # don't bother
            {}

        checks = {}
        for group, group_features in default_features.items():
            # make sure that groups are not _partly_ selected
            for chan, features in group_features["features"].items():
                overlap_sum = sum(x in features for x in selection[chan])
                if overlap_sum == len(features):
                    checks[group] = True
                else:
                    checks[group] = False

        return checks

    def openAdvancedDlg(self):
        dlg = FeatureSelectionDialog(self.channel_names, self.supported_features, self.feature_selection(), parent=self)
        dlg_result = dlg.exec_()
        if dlg_result != dlg.Accepted:
            return

        selections = dlg.selections()

    @classmethod
    def default_features(cls, raw_channels, edge_channels, data_is_3D=False):
        default_sp_features = [
            "standard_sp_mean",
            "standard_sp_quantiles_10",
            "standard_sp_quantiles_90",
        ]
        default_boundary_features = [
            "standard_edge_mean",
            "standard_edge_quantiles_10",
            "standard_edge_quantiles_90",
        ]
        default_shape_feautures = [
            "edgeregion_edge_area",
            "edgeregion_edge_regionradii_0",
            "edgeregion_edge_regionradii_1",
        ]

        if data_is_3D:
            default_shape_feautures += ["edgeregion_edge_regionradii_2", "edgeregion_edge_volume"]

        selected_features = {}

        for channel in raw_channels:
            selected_features[channel] = default_sp_features

        for channel in edge_channels:
            selected_features[channel] = default_boundary_features

        default_dialog_features = {
            cls.FeatureGroup.shape: {
                "features": {edge_channels[0]: default_shape_feautures},
                "state": True,
                "description": "shape stuff, duh.",
            },
            cls.FeatureGroup.boundary_edge: {
                "description": "Intensity statistics (mean, Q10, Q90) computed on the boundary channel along the edges.",
                "features": {channel: default_boundary_features for channel in edge_channels},
                "state": True,
            },
            cls.FeatureGroup.raw_edge: {
                "description": "Intensity statistics (mean, Q10, Q90) computed on the raw data along the edges.",
                "features": {channel: default_boundary_features for channel in raw_channels},
                "state": False,
            },
            cls.FeatureGroup.boundary_sp: {
                "description": "Intensity statistics (mean, Q10, Q90) computed on the boundary channel on superpixels.",
                "features": {channel: default_sp_features for channel in edge_channels},
                "state": False,
            },
            cls.FeatureGroup.raw_sp: {
                "description": "Intensity statistics (mean, Q10, Q90) computed on the raw data on superpixels.",
                "features": {channel: default_sp_features for channel in raw_channels},
                "state": True,
            },
        }

        return default_dialog_features


if __name__ == "__main__":
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.environ["QT_MAC_WANTS_LAYER"] = "1"
    os.environ["VOLUMINA_SHOW_3D_WIDGET"] = "0"

    from PyQt5.QtWidgets import QApplication

    app = QApplication([])

    supported_features = [
        "edgeregion_edge_regionradii_0",
        "edgeregion_edge_regionradii_1",
        "standard_sp_mean",
        "standard_sp_quantiles_10",
        "standard_sp_quantiles_100",
        "standard_sp_quantiles_20",
        "standard_sp_quantiles_30",
        "standard_sp_quantiles_40",
        "standard_sp_quantiles_50",
        "standard_sp_quantiles_60",
        "standard_sp_quantiles_70",
        "standard_sp_quantiles_80",
        "standard_sp_quantiles_90",
        "standard_edge_mean",
        "standard_edge_quantiles_10",
        "standard_edge_quantiles_100",
        "standard_edge_quantiles_20",
        "standard_edge_quantiles_30",
        "standard_edge_quantiles_40",
        "standard_edge_quantiles_50",
        "standard_edge_quantiles_60",
        "standard_edge_quantiles_70",
        "standard_edge_quantiles_80",
        "standard_edge_quantiles_90",
    ]

    dlg = SimpleEdgeFeatureSelection(
        [
            "raw 0",
            # "raw 1",
            # "raw 2",
            # "raw 3",
            # "raw 4",
            # "raw 5",
            # "raw 6",
            # "raw 10",
            # "raw 11",
            # "raw 12",
            # "raw 13",
            # "raw 14",
            "raw 51",
            "raw 16",
        ],
        ["boundary"],
        ["probs 1", "probs 2"],
        {},
        supported_features=supported_features,
    )
    dlg.exec_()

    states = dlg._get_states()
    print({k: v["state"] for k, v in states.items()})
    features = dlg.feature_selection()
    for k, v in features.items():
        print(k, v)
