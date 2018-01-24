from apistar import Route
from ..views import project


routes = [
    Route('/new-project', 'POST', project.new_project),
    Route('/load-project', 'POST', project.load_project)
]
