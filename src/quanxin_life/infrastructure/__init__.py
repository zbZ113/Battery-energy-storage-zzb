"""Optional infrastructure adapters with explicit runtime dependency injection."""

from quanxin_life.infrastructure.object_store import (
    MinioObjectStoreConfig,
    MinioRuntimeCredentials,
    StoredObjectRef,
    VerifiedMinioObjectStore,
    create_minio_client,
)

__all__ = [
    "MinioObjectStoreConfig",
    "MinioRuntimeCredentials",
    "StoredObjectRef",
    "VerifiedMinioObjectStore",
    "create_minio_client",
]
