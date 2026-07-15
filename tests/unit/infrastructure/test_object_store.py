from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from quanxin_life.infrastructure.object_store import (
    MinioObjectStoreConfig,
    MinioRuntimeCredentials,
    StoredObjectRef,
    VerifiedMinioObjectStore,
    create_minio_client,
)


@dataclass
class _Response:
    payload: bytes
    closed: bool = False
    released: bool = False

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


class _CloseFailResponse(_Response):
    def close(self) -> None:
        self.closed = True
        raise RuntimeError("simulated close failure")


class _FakeMinioClient:
    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], bytes] = {}
        self.last_response: _Response | None = None

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str, *, location: str | None = None) -> None:
        del location
        self.buckets.add(bucket_name)

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: io.BytesIO,
        length: int,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> object:
        del content_type
        payload = data.read()
        assert len(payload) == length
        assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
        self.objects[(bucket_name, object_name)] = payload
        return object()

    def get_object(self, bucket_name: str, object_name: str) -> _Response:
        response = _Response(self.objects[(bucket_name, object_name)])
        self.last_response = response
        return response


class _CloseFailMinioClient(_FakeMinioClient):
    def get_object(self, bucket_name: str, object_name: str) -> _Response:
        response = _CloseFailResponse(self.objects[(bucket_name, object_name)])
        self.last_response = response
        return response


def test_runtime_credentials_are_redacted() -> None:
    credentials = MinioRuntimeCredentials(
        access_key="access-that-must-not-leak",
        secret_key="secret-that-must-not-leak",
    )

    rendered = repr(credentials) + str(credentials)
    assert "access-that-must-not-leak" not in rendered
    assert "secret-that-must-not-leak" not in rendered


def test_client_factory_receives_credentials_only_at_runtime() -> None:
    captured: dict[str, object] = {}
    expected_client = _FakeMinioClient()

    def factory(endpoint: str, **kwargs: object) -> _FakeMinioClient:
        captured["endpoint"] = endpoint
        captured.update(kwargs)
        return expected_client

    client = create_minio_client(
        MinioObjectStoreConfig(
            endpoint="minio.internal:9000",
            bucket="quanxin-artifacts",
            secure=False,
            region="cn-shenzhen",
        ),
        MinioRuntimeCredentials(access_key="runtime-access", secret_key="runtime-secret"),
        client_factory=factory,
    )

    assert client is expected_client
    assert captured == {
        "endpoint": "minio.internal:9000",
        "access_key": "runtime-access",
        "secret_key": "runtime-secret",
        "secure": False,
        "region": "cn-shenzhen",
    }


@pytest.mark.parametrize("bucket", ["", "UPPERCASE", "contains_underscore", "ab"])
def test_config_rejects_invalid_bucket_names(bucket: str) -> None:
    with pytest.raises(ValidationError):
        MinioObjectStoreConfig(endpoint="minio:9000", bucket=bucket)


def test_verified_store_puts_content_addressed_object_and_reads_it_back() -> None:
    client = _FakeMinioClient()
    store = VerifiedMinioObjectStore(
        client=client,
        config=MinioObjectStoreConfig(endpoint="minio:9000", bucket="quanxin-artifacts"),
    )
    payload = b"trusted parquet bytes"
    sha256 = hashlib.sha256(payload).hexdigest()

    store.ensure_bucket()
    stored = store.put_bytes(
        namespace="datasets/canonical",
        payload=payload,
        expected_sha256=sha256,
        content_type="application/vnd.apache.parquet",
    )

    assert stored == StoredObjectRef(
        uri=f"minio://quanxin-artifacts/datasets/canonical/{sha256}",
        sha256=sha256,
        size_bytes=len(payload),
        content_type="application/vnd.apache.parquet",
    )
    assert store.get_bytes(stored) == payload
    assert client.last_response is not None
    assert client.last_response.closed is True
    assert client.last_response.released is True


def test_verified_store_rejects_source_hash_mismatch_before_upload() -> None:
    client = _FakeMinioClient()
    store = VerifiedMinioObjectStore(
        client=client,
        config=MinioObjectStoreConfig(endpoint="minio:9000", bucket="quanxin-artifacts"),
    )

    with pytest.raises(ValueError, match="SHA-256"):
        store.put_bytes(
            namespace="reports",
            payload=b"real payload",
            expected_sha256="0" * 64,
            content_type="text/markdown",
        )

    assert client.objects == {}


@pytest.mark.parametrize("namespace", ["../escape", "/absolute", "reports//draft", "报告"])
def test_verified_store_rejects_unsafe_namespaces(namespace: str) -> None:
    store = VerifiedMinioObjectStore(
        client=_FakeMinioClient(),
        config=MinioObjectStoreConfig(endpoint="minio:9000", bucket="quanxin-artifacts"),
    )

    with pytest.raises(ValueError, match="namespace"):
        store.put_bytes(
            namespace=namespace,
            payload=b"payload",
            expected_sha256=hashlib.sha256(b"payload").hexdigest(),
            content_type="application/octet-stream",
        )


def test_verified_store_detects_tampering_on_download() -> None:
    client = _FakeMinioClient()
    store = VerifiedMinioObjectStore(
        client=client,
        config=MinioObjectStoreConfig(endpoint="minio:9000", bucket="quanxin-artifacts"),
    )
    payload = b"original"
    sha256 = hashlib.sha256(payload).hexdigest()
    stored = store.put_bytes(
        namespace="models",
        payload=payload,
        expected_sha256=sha256,
        content_type="application/octet-stream",
    )
    client.objects[("quanxin-artifacts", f"models/{sha256}")] = b"tampered"

    with pytest.raises(ValueError, match="SHA-256"):
        store.get_bytes(stored)


def test_verified_store_rejects_reference_for_another_bucket() -> None:
    store = VerifiedMinioObjectStore(
        client=_FakeMinioClient(),
        config=MinioObjectStoreConfig(endpoint="minio:9000", bucket="quanxin-artifacts"),
    )

    with pytest.raises(ValueError, match="configured bucket"):
        store.get_bytes(
            StoredObjectRef(
                uri=f"minio://other-bucket/reports/{'a' * 64}",
                sha256="a" * 64,
                size_bytes=1,
                content_type="text/plain",
            )
        )


@pytest.mark.parametrize(
    "uri_template",
    [
        "minio://quanxin-artifacts/../{sha256}",
        "minio://quanxin-artifacts/reports//{sha256}",
        "minio://quanxin-artifacts/%2e%2e/{sha256}",
        "minio://quanxin-artifacts/reports/{sha256}?download=1",
        "minio://quanxin-artifacts/reports/{sha256}#fragment",
    ],
)
def test_verified_store_rejects_noncanonical_download_uri(uri_template: str) -> None:
    store = VerifiedMinioObjectStore(
        client=_FakeMinioClient(),
        config=MinioObjectStoreConfig(endpoint="minio:9000", bucket="quanxin-artifacts"),
    )
    sha256 = "a" * 64

    with pytest.raises(ValueError, match="canonical"):
        store.get_bytes(
            StoredObjectRef(
                uri=uri_template.format(sha256=sha256),
                sha256=sha256,
                size_bytes=1,
                content_type="text/plain",
            )
        )


def test_verified_store_releases_connection_even_when_close_fails() -> None:
    client = _CloseFailMinioClient()
    payload = b"verified"
    sha256 = hashlib.sha256(payload).hexdigest()
    client.objects[("quanxin-artifacts", f"reports/{sha256}")] = payload
    store = VerifiedMinioObjectStore(
        client=client,
        config=MinioObjectStoreConfig(endpoint="minio:9000", bucket="quanxin-artifacts"),
    )

    with pytest.raises(RuntimeError, match="close failure"):
        store.get_bytes(
            StoredObjectRef(
                uri=f"minio://quanxin-artifacts/reports/{sha256}",
                sha256=sha256,
                size_bytes=len(payload),
                content_type="text/plain",
            )
        )

    assert client.last_response is not None
    assert client.last_response.released is True
