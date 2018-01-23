import json
from apistar.renderers import Renderer
from apistar import http
from datetime import datetime
from .models import models
import numpy


import logging

logger = logging.getLogger(__name__)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # check if object has to_json() method
        logger.debug(f'Serializing object {obj}')
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, models.ModelBase):
            return json.dumps(
                {
                    k: v
                    for k, v in obj.__dict__.items()
                    if not k.startswith('_')
                },
                cls=self.__class__
            )
        elif hasattr(obj, 'to_json'):
            return obj.to_json()

        return json.JSONEncoder.default(self, obj)


class IlastikJSONRenderer(Renderer):
    def render(self, data: http.ResponseData) -> bytes:
        return json.dumps(data, cls=CustomEncoder).encode('utf-8')
