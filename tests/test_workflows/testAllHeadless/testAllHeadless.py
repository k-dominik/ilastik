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
"""Basic tests that can be applied to _all_ workflows in headless mode

This is meant as a sanity check to make sure that workflows can be at least
started after changes are committed.

Also this can be used as a basis for further headless-mode testing.
"""
import imp
import os
import shutil
import sys
import tempfile

import ilastik
from ilastik.workflow import getAvailableWorkflows

import logging
logger = logging.getLogger(__name__)


class TestHeadlessWorkflowStartup(object):
    """Start a headless shell and create a project for each workflow"""
    @classmethod
    def setupClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.workflow_list = getAvailableWorkflows()

        logger.debug('looking for ilastik.py...')
        ilastik_entry_file_path = os.path.join(
            os.path.split(os.path.realpath(ilastik.__file__))[0],
            "../ilastik.py")
        if not os.path.exists( ilastik_entry_file_path ):
            raise RuntimeError(
                f"Couldn't find ilastik.py startup script: {ilastik_entry_file_path}")

        cls.ilastik_startup = imp.load_source(
            'ilastik_startup',
            ilastik_entry_file_path
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_workflows(self):
        for wf in self.workflow_list:
            yield self.start_workflow, wf

    def start_workflow(self, workflow_class_tuple):
        """
        Args:
            workflow_class_tuple (tuple): tuple returned from getAvailableWorkflows
              with (workflow_class, workflow_name, workflow_class.workflowDisplayName)
        """
        workflow_class, workflow_name, display_name = workflow_class_tuple
        logger.debug(f'starting {workflow_name}')
        project_file = os.path.join(
            self.temp_dir,
            f'test_project_{"_".join(workflow_name.split())}.ilp'
        )
        args = [
            '--headless',
            f'--new_project={project_file}',
            f'--workflow={workflow_name}',
        ]
        # Clear the existing commandline args so it looks like we're starting fresh.
        sys.argv = ['ilastik.py']
        sys.argv.extend(args)

        # Start up the ilastik.py entry script as if we had launched it from the command line
        self.ilastik_startup.main()

        # now check if the project file has been created:
        assert os.path.exists(project_file), f"Project File {project_file} creation not successful"

