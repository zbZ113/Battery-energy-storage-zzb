"""Celery entrypoint for Agent and Advanced calibration workers."""

from __future__ import annotations

import os

from deploy.competition_runtime import (
    create_competition_runtime,
    register_feishu_sibling_recovery,
)
from deploy.runtime_settings import CompetitionRuntimeSettings

settings = CompetitionRuntimeSettings.from_environment(os.environ)
runtime = create_competition_runtime(settings)
register_feishu_sibling_recovery(runtime)
app = runtime.celery_app
