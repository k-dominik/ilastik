from apistar.backends.sqlalchemy_backend import (
    Session, SQLAlchemyBackend, drop_tables, create_tables
)
import typing
from apistar import Command, Settings
from .models.models import DatasetRecord, NetworkRecord, ProjectRecord
import os
from lazyflow.operators.ioOperators import OpInputDataReader
from sqlalchemy.ext.declarative.api import DeclarativeMeta


def rescan_folders(session: Session, backend: SQLAlchemyBackend):
    """Rescan the local database and add everything to a fresh database
    """
    # HACK: For some reason, DI for `Settings` does not work here, so values hard-coded for now :(
    # TODO: find out how to resolve this and use `Settings` in param DI
    import warnings
    warnings.warn("settings hard-coded! Use dependency injection.")
    settings = {
        'ILASTIK_CONFIG': {
            'DATA_PATH': os.path.expanduser('~/ilastik_server/data'),
            'PROJECTS_PATH': os.path.expanduser('~/ilastik_server/projects'),
            'NETWORKS_PATH': os.path.expanduser('~/ilastik_server/networks'),
        }
    }
    # delete everything and create new tables
    drop_tables(backend)
    create_tables(backend)
    ilastik_settings = settings['ILASTIK_CONFIG']
    data_path = ilastik_settings['DATA_PATH']
    projects_path = ilastik_settings['PROJECTS_PATH']
    networks_path = ilastik_settings['NETWORKS_PATH']

    scan_folders(session, data_path, projects_path, networks_path)


def scan_folders(session: Session, data_path: str, projects_path: str, networks_path: str) -> None:
    report = []

    runs = [
        (session, DatasetRecord, data_path, OpInputDataReader.SupportedExtensions),
        (session, NetworkRecord, networks_path, ['nn']),
        (session, ProjectRecord, projects_path, ['ilp'])
    ]

    for run in runs:
        report.append(scan_folder(*run))

    print(report)


def scan_folder(
        session: Session,
        record_type: DeclarativeMeta,
        path: str,
        allowed_extensions: typing.List[str]) -> dict:
    """ """
    files = [
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ]

    records = []
    for file in files:
        base_name, extension = os.path.splitext(file)
        if extension.strip('.') in allowed_extensions:
            record = record_type(
                alias=base_name,
                path=os.path.join(path, file),
                is_located_relative=True,
                thumbnail=os.path.join(
                    path,
                    os.path.join('thumbnails', f"{base_name}.png")
                )
            )
            session.add(record)
            records.append(record)

    session.flush()

    return {
        'scanned_path': path,
        'discovered_records': len(records),
    }


commands = [
    Command('rescan-folders', rescan_folders)
]
