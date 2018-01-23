from functools import partial
import json
import vigra
from apistar import typesystem
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, VARCHAR

class ModelBase(object):
    pass

Base = declarative_base(cls=ModelBase)


class JSONEncodedAxistags(TypeDecorator):
    """Represents an `vigra.axistags` as a json-encoded string."""

    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = vigra.AxisTags.toJSON(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = vigra.AxisTags.fromJSON(value)
        return value


class JSONEncodedTuple(TypeDecorator):
    """Represents a python tuple as a json encoded string"""

    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


LastModifiedColumn = partial(Column, DateTime(timezone=True), default=func.now(), onupdate=func.now())


class DatasetRecord(Base):
    __tablename__ = "Dataset"
    id = Column(Integer, primary_key=True)
    alias = Column(String)
    axistags = Column(JSONEncodedAxistags)
    # this is True, if it is in the local server storage
    is_located_relative = Column(Boolean)
    path = Column(String)
    shape = Column(JSONEncodedTuple)
    thumbnail = Column(String)
    last_updated = LastModifiedColumn()


class NetworkRecord(Base):
    __tablename__ = "Network"
    id = Column(Integer, primary_key=True)
    data_description = Column(String)
    # this is True, if it is in the local server storage
    is_located_relative = Column(Boolean)
    thumbnail = Column(String)
    last_updated = LastModifiedColumn()


class ProjectRecord(Base):
    __tablename__ = "Project"
    id = Column(Integer, primary_key=True)
    # this is True, if it is in the local server storage
    is_located_relative = Column(Boolean)
    # Absolute path, unless `is_located_relative`. Then relative to settings['ILASTIK_CONFIG']['']
    path = Column(String)
    thumbnail = Column(String)
    last_updated = LastModifiedColumn()
