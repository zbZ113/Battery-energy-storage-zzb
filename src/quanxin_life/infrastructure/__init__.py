"""Optional infrastructure adapters with explicit runtime dependency injection."""

from quanxin_life.infrastructure.celery_queue import (
    AGENT_RUN_QUEUE,
    AGENT_RUN_TASK,
    CeleryAgentRunQueue,
    CeleryQueueConfig,
    create_celery_app,
)
from quanxin_life.infrastructure.object_store import (
    MinioObjectStoreConfig,
    MinioRuntimeCredentials,
    StoredObjectRef,
    VerifiedMinioObjectStore,
    create_minio_client,
)

__all__ = [
    "AGENT_RUN_QUEUE",
    "AGENT_RUN_TASK",
    "CeleryAgentRunQueue",
    "CeleryQueueConfig",
    "MinioObjectStoreConfig",
    "MinioRuntimeCredentials",
    "StoredObjectRef",
    "VerifiedMinioObjectStore",
    "create_celery_app",
    "create_minio_client",
]
