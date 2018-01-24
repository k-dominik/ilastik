"""Basic views to that interact with the shell on a workflow level

"""
import logging
import os
from apistar import get_current_app, Response, http
from .ilastikAPItypes import (
    LocalDataset
)

from lazyflow.utility.pathHelpers import PathComponents

logger = logging.getLogger(__name__)


def get_current_workflow_name():
    logger.debug('getting current workflow name')
    try:
        workflow_name = get_current_app()._ilastik_api.workflow_name
    except AttributeError as e:
        return Response(
            status=404,
        )
    return {'workflow_name': workflow_name}


def add_input_to_current_workflow(dataset: LocalDataset):
    data_name = dataset['data_name']
    data_path = get_current_app()._ilastik_config.data_path
    image_file = os.path.join(data_path, data_name)
    pc = PathComponents(image_file)
    if not os.path.exists(pc.externalPath):
        ret = {'message': 'Could not find image file. Is it on the server?'}
        status_code = 404
        return Response(content=ret, status=status_code)

    get_current_app()._ilastik_api.add_dataset(image_file)

    data = {
        'data_loaded': data_name
    }
    return data


def get_structured_info():
    dataset_names, json_states = get_current_app()._ilastik_api.get_structured_info()
    resp = {
        'states': json_states,
        'image_names': dataset_names
    }
    return resp

