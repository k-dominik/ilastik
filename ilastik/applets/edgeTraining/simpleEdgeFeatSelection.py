import enum
from functools import partial
from typing import Dict, List, Set

from ilastikrag.gui import FeatureSelectionDialog
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from ilastik.widgets.collapsibleWidget import CollapsibleWidget
from ilastik.utility.gui.widgets import silent_qobject

# need proper names:
# * feature_dict: {channel: [feats]}
# * internal_state_dict {all the internal stuff}


class SimpleEdgeFeatureSelection(QDialog):
    @enum.unique
    class FeatureGroup(enum.Enum):
        shape = "shape"
        boundary_edge = "boundary channel along edges"
        boundary_sp = "boundary channel on superpixels"
        raw_edge = "raw data along edges"
        raw_sp = "raw data on superpixels"

    def __init__(
        self,
        raw_channels,
        boundary_channels,
        probability_channels,
        selection,
        supported_features,
        data_is_3d=False,
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout()

        self.setWindowTitle("Edge Feature Selection")
        self.raw_channels = raw_channels
        self.boundary_channels = boundary_channels
        self.probability_channels = probability_channels
        self.data_is_3d = data_is_3d
        self.channel_names = raw_channels + boundary_channels + probability_channels

        self._internal_state = self._default_features_state_dict(raw_channels, boundary_channels, self.data_is_3d)

        # for the "big" feature selection dialog
        self.supported_features = supported_features

        def make_checkbox(name):
            checkbox = QCheckBox(name.value)
            details = self._internal_state[name]
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

        advButton = QPushButton("Advanced")
        advButton.clicked.connect(self.openAdvancedDlg)
        self._advButton = advButton
        buttonbox.addButton(advButton, QDialogButtonBox.ActionRole)

        resetButton = QPushButton("Reset")
        resetButton.setToolTip("Reset selection to default.")
        resetButton.clicked.connect(self.reset_to_default)
        buttonbox.addButton(resetButton, QDialogButtonBox.ResetRole)

        layout.addWidget(buttonbox)

        self.setLayout(layout)
        layout.setSizeConstraint(QLayout.SetFixedSize)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        self.setFeatures(selection)

    def setFeatures(self, features):
        # set some internal feature variable
        self._current_selection = features

        if not self.check_selection_compatible_with_dlg_features(self._internal_state, self._current_selection):
            self.setEnabledState(False)
            for checkbox in self.checkboxes.values():
                with silent_qobject(checkbox) as w:
                    w.setChecked(Qt.Unchecked)
            return

        self.setEnabledState(True)

        # synchronize checkboxes
        if self._current_selection:
            checks = self._checkmarks(self._internal_state, self._current_selection)
            for check, val in checks.items():
                self._internal_state[check]["state"] = val
                with silent_qobject(self.checkboxes[check]) as w:
                    w.setCheckState(Qt.Checked if val else Qt.Unchecked)

    def setEnabledState(self, state: bool):
        """
        User has selected something in the advanced dialog, disable all
        checkboxes.
        """
        self.intensityGroupBox.setEnabled(state)
        self.shapeGroupBox.setEnabled(state)

        if state:
            self._advButton.setText("Advanced")
            self._advButton.setToolTip("Open advanced Feature Selection Dialog for more fine-grained control.")
        else:
            self._advButton.setText("Advanced*")
            self._advButton.setToolTip(
                "Non-standard feature selection - reset or open advanced Feature Selection Dialog for more fine-grained control."
            )

    def _update_state(self, group, state):
        """update after checkbox change - always valid feature set"""
        self._internal_state[group]["state"] = bool(state)
        self._current_selection = SimpleEdgeFeatureSelection._to_feature_dict(self._internal_state)

    def selections(self) -> Dict[str, List[str]]:
        """
        feature selection compatible with operator/ilastikrag.gui.feature_selection_dialog

        returns:
          dict of currently selected features (values) per channel name (keys)
        """
        return self._current_selection

    def check_selection_compatible_with_dlg_features(self, default_features, selection) -> bool:
        for group, group_features in default_features.items():
            # make sure that groups are not _partly_ selected
            for chan, features in group_features["features"].items():
                overlap_sum = sum(x in features for x in selection.get(chan, []))
                if overlap_sum not in [0, len(features)]:
                    return False
            # now check that there are no features not in the default_feauture_set
            feats_flat = SimpleEdgeFeatureSelection._to_feature_dict(default_features)

            for chan, feats in selection.items():
                if feats and (chan not in feats_flat):
                    return False
                if any(x not in feats_flat[chan] for x in feats):
                    return False
        return True

    def reset_to_default(self):
        self.setFeatures(
            SimpleEdgeFeatureSelection._to_feature_dict(
                self._default_features_state_dict(self.raw_channels, self.boundary_channels)
            )
        )

    def _checkmarks(self, default_features, selection) -> Dict[str, bool]:
        if not self.check_selection_compatible_with_dlg_features(default_features, selection):
            # don't bother
            return {}

        checks = {}
        for group, group_features in default_features.items():
            # make sure that groups are not _partly_ selected
            for chan, features in group_features["features"].items():
                overlap_sum = sum(x in features for x in selection.get(chan, []))
                if overlap_sum == len(features):
                    checks[group] = True
                else:
                    checks[group] = False

        return checks

    def openAdvancedDlg(self):
        default_features = self._default_features_state_dict(self.raw_channels, self.boundary_channels)
        default_features = SimpleEdgeFeatureSelection._to_feature_dict(default_features)

        dlg = FeatureSelectionDialog(
            self.channel_names, self.supported_features, self.selections(), default_features, parent=self
        )
        dlg_result = dlg.exec_()
        if dlg_result != dlg.Accepted:
            return

        selections = dlg.selections()
        self.setFeatures(selections)

    @classmethod
    def _default_features_state_dict(cls, raw_channels, edge_channels, data_is_3d=False):
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

        if data_is_3d:
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

    @staticmethod
    def _to_feature_dict(state_dict) -> Dict[str, List[str]]:
        selected_features: Dict[str, Set[str]] = {}

        for group_features in state_dict.values():
            if group_features["state"]:
                for channel, features in group_features["features"].items():
                    if channel not in selected_features:
                        selected_features[channel] = set()
                    selected_features[channel] |= set(features)

        return {k: list(v) for k, v in selected_features.items()}

    @classmethod
    def default_features(cls, raw_channels, boundary_channels, data_is_3d):
        return cls._to_feature_dict(cls._default_features_state_dict(raw_channels, boundary_channels, data_is_3d))


if __name__ == "__main__":
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.environ["QT_MAC_WANTS_LAYER"] = "1"
    os.environ["VOLUMINA_SHOW_3D_WIDGET"] = "0"

    from PyQt5.QtWidgets import QApplication

    app = QApplication([])

    supported_features = [
        "edgeregion_edge_area",
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
        {
            "boundary": ["standard_sp_mean", "standard_sp_quantiles_10", "standard_sp_quantiles_90"],
            "raw 0": [
                "standard_edge_mean",
                "standard_edge_quantiles_10",
                "standard_edge_quantiles_90",
            ],
            "raw 51": [
                "standard_edge_mean",
                "standard_edge_quantiles_10",
                "standard_edge_quantiles_90",
            ],
            "raw 16": [
                "standard_edge_mean",
                "standard_edge_quantiles_10",
                "standard_edge_quantiles_90",
            ],
        },
        supported_features=supported_features,
    )
    dlg.exec_()

    features = dlg.selections()
    for k, v in features.items():
        print(k, v)
