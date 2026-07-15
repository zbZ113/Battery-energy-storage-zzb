"""Content-addressed MinIO storage with mandatory SHA-256 verification."""

from __future__ import annotations

import hashlib
import importlib
import io
import re
from collections.abc import Callable
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9_-]*(?:/[a-z0-9][a-z0-9_-]*)*\Z")


class MinioObjectStoreConfig(BaseModel):
    """Non-secret MinIO connection and bucket settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str = Field(min_length=1)
    bucket: str
    secure: bool = True
    region: str | None = None

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "://" in normalized:
            raise ValueError("MinIO endpoint must be a nonblank host[:port] without a scheme")
        return normalized

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        if not _BUCKET.fullmatch(value) or ".." in value:
            raise ValueError("MinIO bucket name is invalid")
        return value

    @field_validator("region")
    @classmethod
    def normalize_region(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("MinIO region must not be blank when provided")
        return normalized


class MinioRuntimeCredentials(BaseModel):
    """Runtime-only credentials whose representations remain redacted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    access_key: SecretStr
    secret_key: SecretStr

    @field_validator("access_key", "secret_key")
    @classmethod
    def reject_blank_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("MinIO credentials must not be blank")
        return value


class StoredObjectRef(BaseModel):
    """Verified immutable object address safe to persist in PostgreSQL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(ge=0)
    content_type: str = Field(min_length=1)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value


class _ReadableResponse(Protocol):
    def read(self) -> bytes: ...

    def close(self) -> None: ...

    def release_conn(self) -> None: ...


class MinioClient(Protocol):
    """Subset of the MinIO SDK used by the verified adapter."""

    def bucket_exists(self, bucket_name: str) -> bool: ...

    def make_bucket(self, bucket_name: str, *, location: str | None = None) -> None: ...

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: io.BytesIO,
        length: int,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> object: ...

    def get_object(self, bucket_name: str, object_name: str) -> _ReadableResponse: ...


MinioClientFactory = Callable[..., MinioClient]


def create_minio_client(
    config: MinioObjectStoreConfig,
    credentials: MinioRuntimeCredentials,
    *,
    client_factory: MinioClientFactory | None = None,
) -> MinioClient:
    """Construct a client explicitly without reading environment variables."""

    factory = client_factory
    if factory is None:
        try:
            minio_module = importlib.import_module("minio")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "MinIO support requires installing the infrastructure dependency group"
            ) from exc
        factory = cast(MinioClientFactory, cast(Any, minio_module).Minio)
    return factory(
        config.endpoint,
        access_key=credentials.access_key.get_secret_value(),
        secret_key=credentials.secret_key.get_secret_value(),
        secure=config.secure,
        region=config.region,
    )


class VerifiedMinioObjectStore:
    """Store immutable bytes under digest-derived keys and verify every read."""

    def __init__(self, *, client: MinioClient, config: MinioObjectStoreConfig) -> None:
        self._client = client
        self._config = config

    def ensure_bucket(self) -> None:
        """Create the configured bucket only when this explicit method is called."""

        if not self._client.bucket_exists(self._config.bucket):
            self._client.make_bucket(self._config.bucket, location=self._config.region)

    def put_bytes(
        self,
        *,
        namespace: str,
        payload: bytes,
        expected_sha256: str,
        content_type: str,
    ) -> StoredObjectRef:
        """Upload caller-verified bytes to a content-addressed object key."""

        self._validate_namespace(namespace)
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if not _SHA256.fullmatch(expected_sha256):
            raise ValueError("expected SHA-256 must be a lowercase digest")
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("payload SHA-256 does not match expected SHA-256")
        normalized_content_type = content_type.strip()
        if not normalized_content_type:
            raise ValueError("content_type must not be blank")

        object_name = f"{namespace}/{actual_sha256}"
        self._client.put_object(
            self._config.bucket,
            object_name,
            io.BytesIO(payload),
            len(payload),
            content_type=normalized_content_type,
            metadata={"sha256": actual_sha256},
        )
        return StoredObjectRef(
            uri=f"minio://{self._config.bucket}/{object_name}",
            sha256=actual_sha256,
            size_bytes=len(payload),
            content_type=normalized_content_type,
        )

    def get_bytes(self, stored: StoredObjectRef) -> bytes:
        """Read one object and reject wrong buckets, sizes, or modified content."""

        parsed = urlsplit(stored.uri)
        if parsed.scheme != "minio" or parsed.netloc != self._config.bucket:
            raise ValueError("stored object URI does not target the configured bucket")
        if parsed.query or parsed.fragment or not parsed.path.startswith("/"):
            raise ValueError("stored object URI is not canonical")
        object_name = parsed.path.removeprefix("/")
        try:
            namespace, digest = object_name.rsplit("/", maxsplit=1)
        except ValueError as exc:
            raise ValueError("stored object URI is not canonical") from exc
        if not _NAMESPACE.fullmatch(namespace) or not _SHA256.fullmatch(digest):
            raise ValueError("stored object URI is not canonical")
        if digest != stored.sha256:
            raise ValueError("stored object URI is not bound to its SHA-256")
        if stored.uri != f"minio://{self._config.bucket}/{namespace}/{digest}":
            raise ValueError("stored object URI is not canonical")

        response = self._client.get_object(self._config.bucket, object_name)
        try:
            payload = response.read()
        finally:
            try:
                response.close()
            finally:
                response.release_conn()
        if len(payload) != stored.size_bytes:
            raise ValueError("stored object size does not match its verified reference")
        if hashlib.sha256(payload).hexdigest() != stored.sha256:
            raise ValueError("stored object SHA-256 does not match its verified reference")
        return payload

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not isinstance(namespace, str) or not _NAMESPACE.fullmatch(namespace):
            raise ValueError("object namespace is unsafe")


__all__ = [
    "MinioClient",
    "MinioObjectStoreConfig",
    "MinioRuntimeCredentials",
    "StoredObjectRef",
    "VerifiedMinioObjectStore",
    "create_minio_client",
]
