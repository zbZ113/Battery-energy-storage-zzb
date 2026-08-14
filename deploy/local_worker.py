"""Celery entrypoint for the loopback-only local competition runtime."""

from __future__ import annotations

import os

from deploy.competition_runtime import (
    create_competition_runtime,
    register_feishu_sibling_recovery,
)
from deploy.local_runtime_settings import LocalCompetitionRuntimeSettings

settings = LocalCompetitionRuntimeSettings.from_environment(os.environ)
runtime = create_competition_runtime(
    settings,
    auth_environment="development",
)
register_feishu_sibling_recovery(runtime)
app = runtime.celery_app
