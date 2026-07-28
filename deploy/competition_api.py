"""ASGI entrypoint for the Advanced competition API."""

from __future__ import annotations

import os

from deploy.competition_runtime import create_competition_runtime
from deploy.runtime_settings import CompetitionRuntimeSettings

settings = CompetitionRuntimeSettings.from_environment(os.environ)
runtime = create_competition_runtime(settings)
app = runtime.http_app
