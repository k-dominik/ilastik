from apistar import Route, annotate, render_template
from apistar import http
from apistar import interfaces
from .server import get_data_list, get_network_list
import logging


logger = logging.getLogger(__name__)


def landing(
        injector: interfaces.Injector,
        data_id: http.QueryParam=None,
        network_id: http.QueryParam=None) -> http.Response:
    logger.debug(f"in landing {data_id}, {network_id}")
    if (data_id is None) or (network_id is None):
        # return the rendered frame
        project_list = injector.run(get_data_list)
        network_list = injector.run(get_network_list)
        return render_template(
            'landing.html', dataset_list=project_list, network_list=network_list)
    else:
        logger.debug('do something')

