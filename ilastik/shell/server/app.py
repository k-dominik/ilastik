from apistar.frameworks.wsgi import WSGIApp as App
from apistar import Route, Include
from apistar.handlers import docs_urls, static_urls
from apistar.renderers import HTMLRenderer
import os
from apistar.backends import sqlalchemy_backend
from .routes import basic, data, project, workflow, sites, server
from .ilastikAPI import IlastikAPI
from .renderer import IlastikJSONRenderer
from .models.models import Base
from .commands import commands as own_commands


import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


routes = [
    Include('/api/docs', docs_urls),
    Include('/static', static_urls),
    Include('/api/project', project.routes),
    Include('/api/data', data.routes),
    Include('/api/workflow', workflow.routes),
    Include('/api/server', server.routes),
    Include('/site', sites.routes),
]

# extend here in order to add them to site root
routes.extend(basic.routes)

settings = {
    'RENDERERS': [IlastikJSONRenderer(), HTMLRenderer()],
    'SCHEMA': {
        'TITLE': "ilastik-API",
        'DESCRIPTION': "ilastiks http intrerface for third party applications."
    },
    'DATABASE': {
        'URL': f"sqlite:///{os.path.expanduser('~/ilastik_server/db.sqlite')}",
        'METADATA': Base.metadata,
    },
    'ILASTIK_CONFIG': {
        'DATA_PATH': os.path.expanduser('~/ilastik_server/data'),
        'PROJECTS_PATH': os.path.expanduser('~/ilastik_server/projects'),
        'NETWORKS_PATH': os.path.expanduser('~/ilastik_server/networks'),
    },
    'STATICS': {
        'ROOT_DIR': 'ilastik/shell/server/static',
        'PACKAGE_DIRS': ['apistar']  # Include the built-in apistar static files.
    },
    'TEMPLATES': {
        'ROOT_DIR': 'ilastik/shell/server/templates',     # Include the 'templates/' directory.
        'PACKAGE_DIRS': ['apistar']  # Include the built-in apistar templates.
    }
}


app = App(
    routes=routes,
    settings=settings,
    commands=sqlalchemy_backend.commands + own_commands,
    components=sqlalchemy_backend.components,

)


ilastik_api = IlastikAPI()
app._ilastik_api = ilastik_api


if __name__ == '__main__':
    app.main()
