import os

from apistar import get_current_app, Response
from apistar.backends.sqlalchemy_backend import Session
from apistar.types import Settings
from ..tools import log_function_call, time_function_call
from ..ilastikAPItypes import LocalProject, NewLocalProject
from ..models import models
import logging
import typing
logger = logging.getLogger(__name__)


@time_function_call(logger)
def get_project_list(session: Session, settings: Settings) -> typing.List[dict]:
    records = session.query(models.DatasetRecord).all()
    return records


@time_function_call(logger)
def new_project(project: NewLocalProject, session: Session, settings: Settings):
    app = get_current_app()
    project_name = project['project_name']
    project_type = project['project_type']
    projects_path = settings['ILASTIK_CONFIG']['PROJECTS_PATH']
    project_file_name = os.path.join(
        projects_path,
        "{project_name}.ilp".format(project_name=project_name))
    app._ilastik_api.create_project(project_file_name, project_type)

    record = models.ProjectRecord(
        path=project_file_name,
        is_located_relative=True,
        thumbnail=os.path.join(
            projects_path,
            os.path.join('thumbnails', f"{project_name}.png")
        )
    )
    session.add(record)

    session.flush()
    return {
        'record': record,
        'project_name': project_name,
        'project_type': project_type,
        'message': 'Project created.'
    }


@time_function_call(logger)
@log_function_call(logger)
def load_project(project: LocalProject, session: Session, settings: Settings):
    # TODO: adapt to database usage
    raise NotImplementedError("Needs to be reimplemented")
    project_name = project['project_name']
    project_list = get_project_list()
    project_names = [os.path.splitext(os.path.split(x))[0] for x in project_list]
    if project_name not in project_names:
        return Response(
            content={
                'message': 'Project could not be found.',
            },
            status=404
        )

    get_current_app()._ilastik_api.load_project_file(

    )

    return {
        'project_loaded': project,
        'message': 'Project loaded successfully.'
    }
