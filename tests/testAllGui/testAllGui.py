###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2017, the ilastik developers
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
"""Basic tests that can be applied to _all_ workflows in gui mode

This is meant as a sanity check to make sure that workflows can be at least
started after changes are committed.

Also this can be used as a basis for further gui-mode testing.
"""
import imp
import os
import shutil
import sys
import tempfile

import ilastik
from ilastik.workflow import getAvailableWorkflows
from tests.helpers import ShellGuiTestCaseBase

import logging
logger = logging.getLogger(__name__)


class TestHeadlessWorkflowStartup(object):
    """Start a ilastik gui shell and create a project for each workflow"""
    @classmethod
    def setupClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)


class TestGui(ShellGuiTestCaseBase):
    """"""
    @classmethod
    def workflowClass(cls):
        return PixelClassificationWorkflow

    @classmethod
    def setupClass(cls):
        # Base class first
        super().setupClass()
        cls.temp_dir = tempfile.mkdtemp()
        cls.project_file = os.path.join(cls.temp_dir, 'test_project.ilp')

        # Start the timer
        cls.timer = Timer()
        cls.timer.unpause()

    @classmethod
    def teardownClass(cls):
        cls.timer.pause()
        logger.debug( "Total Time: {} seconds".format( cls.timer.seconds() ) )

        # Call our base class so the app quits!
        super().teardownClass()

        # Clean up: Delete any test files we generated
        shutil.rmtree(cls.temp_dir)

    def test_1_NewProject(self):
        """Create a blank project and save it."""
        def impl():
            projFilePath = self.PROJECT_FILE

            shell = self.shell

            # New project
            shell.createAndLoadNewProject(projFilePath, self.workflowClass())

            # Save and close
            shell.projectManager.saveProject()
            shell.ensureNoCurrentProject(assertClean=True)

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)
