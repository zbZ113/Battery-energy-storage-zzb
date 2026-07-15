"""Runnable foundation API with only context-free implemented tools.

This entry point is intentionally smaller than the competition application. It
exists for transport smoke tests and does not fabricate enterprise resolvers,
models, calibration cohorts, policies, uploads, or workflow state.
"""

from quanxin_life.api import (
    create_available_tool_invocation_service,
    create_fastapi_app,
)

app = create_fastapi_app(create_available_tool_invocation_service())
