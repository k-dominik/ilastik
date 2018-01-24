"""Basic views for data interaction

"""
from apistar.backends.sqlalchemy_backend import Session
from apistar.types import Settings
from ..models import models
from ..tools import log_function_call, time_function_call
import logging
import typing

logger = logging.getLogger(__name__)


@time_function_call(logger)
def get_project_list(session: Session, settings: Settings) -> typing.List[models.ProjectRecord]:
    records = session.query(models.ProjectRecord).all()
    return records


@time_function_call(logger)
def get_data_list(session: Session, settings: Settings) -> typing.List[models.DatasetRecord]:
    records = session.query(models.DatasetRecord).all()
    return records


@time_function_call(logger)
def get_network_list(session: Session, settings: Settings) -> typing.List[models.NetworkRecord]:
    records = session.query(models.NetworkRecord).all()
    return records
