from apistar import Route

from .workflowViews import (
    get_current_workflow_name,
    add_input_to_current_workflow,
    get_structured_info,
)

routes = [
    Route('/current_workflow_name', 'GET', get_current_workflow_name),
    Route('/structured-info', 'GET', get_structured_info),
    Route('/add-input-data', 'POST', add_input_to_current_workflow)
]
