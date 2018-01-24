from apistar import Route
from ..views import server


routes = [
    Route('/project-list', 'GET', server.get_project_list),
    Route('/data-list', 'GET', server.get_data_list),
    Route('/network-list', 'GET', server.get_network_list)
]
