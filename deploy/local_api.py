"""ASGI entrypoint for the loopback-only local competition API."""

from __future__ import annotations

import os

from deploy.competition_runtime import create_competition_runtime
from deploy.local_runtime_settings import LocalCompetitionRuntimeSettings

settings = LocalCompetitionRuntimeSettings.from_environment(os.environ)
runtime = create_competition_runtime(
    settings,
    auth_environment="development",
)
app = runtime.http_app
