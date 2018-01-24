from apistar import Route
from ..views import sites


# using explicit names here as apistar doesn't allow multiple methods on routes
routes = [
    Route('/landing', 'GET', sites.landing),
]
