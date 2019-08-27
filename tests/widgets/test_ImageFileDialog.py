import pytest
import os
from pathlib import Path
from unittest import mock

from PyQt5.QtCore import QTimer

from ilastik.widgets.ImageFileDialog import ImageFileDialog


@pytest.fixture
def blank_preferences():
    preference_manager = mock.Mock()
    preference_manager.get = mock.Mock(return_value=None)
    return preference_manager


@pytest.fixture
def image(tmp_path) -> Path:
    image_path = Path(tmp_path) / "some_picture.png"
    image_path.touch()
    return image_path

# @pytest.mark.skipif(os.name == 'nt', reason="Unimportant tests that either hang of segfaults on windows")
def test_default_image_directory_is_home_with_blank_preferences_file(blank_preferences):
    dialog = ImageFileDialog(None, preferences_manager=blank_preferences)
    assert dialog.directory().absolutePath() == Path("~").expanduser().absolute().as_posix()


# @pytest.mark.skipif(os.name == 'nt', reason="Unimportant tests that either hang of segfaults on windows")
def test_picking_file_updates_default_image_directory_to_previously_used( blank_preferences, image: Path):
    preferences = blank_preferences
    dialog = ImageFileDialog(None, preferences_manager=preferences)
    assert dialog.directory().absolutePath() == Path("~").expanduser().absolute().as_posix()
    dialog.selectFile(image.as_posix())

    QTimer.singleShot(10, dialog.accept)
    assert dialog.getSelectedPaths() == [image]

    preferences.set.assert_called_once_with(dialog.preferences_group, dialog.preferences_setting, image.as_posix())


# @pytest.mark.skipif(os.name == 'nt', reason="Unimportant tests that either hang of segfaults on windows")
def test_picking_n5_json_file_returns_directory_path(tmp_n5_file: Path, blank_preferences):
    dialog = ImageFileDialog(None, preferences_manager=blank_preferences)
    dialog.setDirectory(str(tmp_n5_file))
    dialog.selectFile("attributes.json")

    QTimer.singleShot(10, dialog.accept)
    assert dialog.getSelectedPaths() == [tmp_n5_file]
