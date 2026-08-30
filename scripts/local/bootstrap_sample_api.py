"""Initialize the reviewed MATR sample through the running product API."""

from __future__ import annotations

import argparse
import base64
import getpass
import http.cookiejar
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CUTOFFS = (20, 50, 100, 150)
REGISTRY_ID = "c31f62e68faa66e56b16d21ebdd3067d5dea0c8408bb3ad6baa73e05a42824be"
PROJECT_NAME = "Hiro 本地验证"
DATASET_NAME = "MATR_b3c34 本地验证样例"
DATASET_SCHEMA_VERSION = "cycle-record-v1"


def required_calibration_routes() -> tuple[tuple[str, int, str], ...]:
    return (
        ("RUL", 20, "DEFAULT"),
        ("RUL", 50, "COVERAGE"),
        ("RUL", 100, "COVERAGE"),
        ("RUL", 150, "COVERAGE"),
        ("SOH", 20, "MEAN_ACCURACY"),
        ("SOH", 20, "TAIL_EFFICIENCY"),
        ("SOH", 50, "MEAN_ACCURACY"),
        ("SOH", 50, "TAIL_EFFICIENCY"),
        ("SOH", 100, "MEAN_ACCURACY"),
        ("SOH", 100, "TAIL_EFFICIENCY"),
        ("SOH", 150, "MEAN_ACCURACY"),
        ("SOH", 150, "TAIL_EFFICIENCY"),
    )


def unique_match(records: Iterable[Any], *, label: str) -> Any | None:
    matches = tuple(records)
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"multiple {label} records match the bootstrap identity")
    return matches[0]


def object_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} API response must be an object")
    return value


def calibration_source_registration_id(runtime_root: Path) -> str:
    registry_path = runtime_root / "config" / "calibration-sources.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("reviewed calibration source registry is invalid") from exc
    sources = registry.get("sources") if isinstance(registry, dict) else None
    if not isinstance(sources, list) or len(sources) != 1:
        raise RuntimeError("local runtime requires exactly one calibration source")
    source = sources[0]
    registration_id = (
        source.get("registration_id") if isinstance(source, dict) else None
    )
    if (
        not isinstance(registration_id, str)
        or not registration_id
        or registration_id != registration_id.strip()
    ):
        raise RuntimeError("calibration source registration ID is invalid")
    return registration_id


class ApiClient:
    def __init__(self, base_url: str) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("local API must use a loopback HTTP origin")
        self.base_url = base_url.rstrip("/")
        self.origin = self.base_url
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        headers = {"Accept": "application/json", "Origin": self.origin}
        body = None
        if payload is not None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=300) as response:
                content = response.read()
                return json.loads(content) if content else None
        except urllib.error.HTTPError as exc:
            detail = f"http_{exc.code}"
            try:
                error_payload = json.loads(exc.read())
                if isinstance(error_payload, dict):
                    candidate = error_payload.get("detail")
                    if isinstance(candidate, str) and len(candidate) <= 120:
                        detail = candidate
            except (UnicodeError, json.JSONDecodeError):
                pass
            raise RuntimeError(
                f"{method} {path} failed with HTTP {exc.code}: {detail}"
            ) from None


def _project(client: ApiClient) -> dict[str, Any]:
    projects = client.request("GET", "/v1/projects")
    existing = unique_match(
        (item for item in projects if item.get("name") == PROJECT_NAME),
        label="project",
    )
    if existing is not None:
        return object_record(existing, label="project")
    return object_record(
        client.request("POST", "/v1/projects", payload={"name": PROJECT_NAME}),
        label="project",
    )


def _dataset(
    client: ApiClient,
    *,
    project_id: str,
    data_version: str,
) -> dict[str, Any]:
    path = f"/v1/projects/{urllib.parse.quote(project_id, safe='')}/datasets"
    datasets = client.request("GET", path)
    existing = unique_match(
        (
            item
            for item in datasets
            if item.get("name") == DATASET_NAME
            and item.get("data_version") == data_version
            and item.get("schema_version") == DATASET_SCHEMA_VERSION
        ),
        label="dataset",
    )
    if existing is not None:
        return object_record(existing, label="dataset")
    return object_record(
        client.request(
            "POST",
            "/v1/datasets",
            payload={
                "project_id": project_id,
                "name": DATASET_NAME,
                "data_version": data_version,
                "schema_version": DATASET_SCHEMA_VERSION,
            },
        ),
        label="dataset",
    )


def _record_batches(
    client: ApiClient,
    *,
    dataset: dict[str, Any],
    demo_root: Path,
) -> tuple[dict[str, Any], ...]:
    dataset_id = str(dataset["dataset_id"])
    list_path = f"/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}/batches"
    existing = client.request("GET", list_path)
    by_cutoff = {int(item["cutoff_cycle"]): item for item in existing}
    if dataset.get("status") == "DRAFT":
        for cutoff in CUTOFFS:
            if cutoff in by_cutoff:
                continue
            csv_path = demo_root / f"MATR_b3c34-cutoff-{cutoff}.csv"
            registration_path = (
                demo_root / f"MATR_b3c34-cutoff-{cutoff}.registration.json"
            )
            payload = {
                "payload_base64": base64.b64encode(csv_path.read_bytes()).decode(
                    "ascii"
                ),
                "registration": json.loads(
                    registration_path.read_text(encoding="utf-8")
                ),
            }
            created = client.request(
                "POST",
                f"/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}/batches/canonical-csv",
                payload=payload,
            )
            by_cutoff[int(created["cutoff_cycle"])] = created
        client.request(
            "POST",
            f"/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}/freeze",
        )
    if set(by_cutoff) != set(CUTOFFS):
        raise RuntimeError("reviewed MATR sample does not contain four cutoffs")
    return tuple(by_cutoff[cutoff] for cutoff in CUTOFFS)


def _activate_routes(client: ApiClient, *, project_id: str, runtime_root: Path) -> None:
    registry_index = (
        runtime_root
        / "registry"
        / "bundles"
        / REGISTRY_ID[:16]
        / "deployment_bundle_index.json"
    )
    index = json.loads(registry_index.read_text(encoding="utf-8"))
    routes = index.get("routes")
    if not isinstance(routes, list) or len(routes) != 15:
        raise RuntimeError("reviewed deployment registry must contain 15 routes")
    active_path = (
        f"/v1/projects/{urllib.parse.quote(project_id, safe='')}/model-routes/active"
    )
    active = client.request("GET", active_path)
    by_coordinate = {
        (item["task"], int(item["cutoff_cycle"]), item["route_role"]): item
        for item in active
    }
    for route in routes:
        coordinate = (
            str(route["task"]),
            int(route["cutoff_cycle"]),
            str(route["role"]),
        )
        artifact_id = str(route["deep_artifact_id"])
        current = by_coordinate.get(coordinate)
        if current is not None:
            if current.get("artifact_id") != artifact_id:
                raise RuntimeError("active model route differs from reviewed registry")
            continue
        task, cutoff, role = coordinate
        client.request(
            "POST",
            "/v1/admin/model-routes/activations",
            payload={
                "project_id": project_id,
                "task": task,
                "cutoff_cycle": cutoff,
                "role": role,
                "artifact_id": artifact_id,
                "reason": "Local cloud-parity bootstrap from reviewed registry",
                "expected_previous_event_sha256": "0" * 64,
            },
            idempotency_key=f"local-route-{task.lower()}-{cutoff}-{role.lower()}",
        )
    active = client.request("GET", active_path)
    if len(active) != 15:
        raise RuntimeError("exactly 15 reviewed model routes must be active")


def _materialize_calibration(
    client: ApiClient,
    *,
    project_id: str,
    source_registration_id: str,
    timeout_seconds: int,
) -> None:
    path = (
        f"/v1/projects/{urllib.parse.quote(project_id, safe='')}"
        "/advanced-calibration/materializations"
    )
    for task, cutoff, role in required_calibration_routes():
        client.request(
            "POST",
            path,
            payload={
                "task": task,
                "cutoff_cycle": cutoff,
                "route_role": role,
                "source_registration_id": source_registration_id,
            },
            idempotency_key=(
                f"local-calibration-{task.lower()}-{cutoff}-{role.lower()}"
            ),
        )

    expected = set(required_calibration_routes())
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        records = client.request("GET", path)
        current = {
            (item["task"], int(item["cutoff_cycle"]), item["route_role"]): item
            for item in records
            if (item["task"], int(item["cutoff_cycle"]), item["route_role"])
            in expected
        }
        failed = [
            coordinate
            for coordinate, item in current.items()
            if item["status"] in {"FAILED", "STALE"}
        ]
        if failed:
            raise RuntimeError(f"calibration materialization failed: {failed}")
        if set(current) == expected and all(
            item["status"] == "READY" for item in current.values()
        ):
            return
        print(f"calibration_ready={sum(item['status'] == 'READY' for item in current.values())}/12")
        time.sleep(5)
    raise RuntimeError("calibration materialization did not become READY in time")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--runtime-root", type=Path, default=Path("D:/QuanxinRuntime/runtime"))
    parser.add_argument("--username", default="admin@quanxin.local")
    parser.add_argument("--calibration-timeout", type=int, default=3600)
    args = parser.parse_args()

    client = ApiClient(args.base_url)
    secret: str | None = getpass.getpass("Current local website password: ")
    try:
        principal = client.request(
            "POST",
            "/v1/auth/login",
            payload={"username": args.username, "password": secret},
        )
    finally:
        secret = None
    if principal.get("role") != "ADMIN":
        raise RuntimeError("local bootstrap requires an ADMIN account")
    if principal.get("must_change_password"):
        raise RuntimeError("change the temporary password in the browser first")

    demo_root = args.runtime_root / "demo-target"
    first_registration = json.loads(
        (demo_root / "MATR_b3c34-cutoff-20.registration.json").read_text(
            encoding="utf-8"
        )
    )
    project = _project(client)
    project_id = str(project["project_id"])
    dataset = _dataset(
        client,
        project_id=project_id,
        data_version=str(first_registration["data_version"]),
    )
    batches = _record_batches(client, dataset=dataset, demo_root=demo_root)
    client.request(
        "POST",
        "/v1/admin/model-artifacts/advanced-candidates",
        payload={"project_id": project_id},
    )
    _activate_routes(client, project_id=project_id, runtime_root=args.runtime_root)
    _materialize_calibration(
        client,
        project_id=project_id,
        source_registration_id=calibration_source_registration_id(
            args.runtime_root
        ),
        timeout_seconds=args.calibration_timeout,
    )

    print(f"project_id={project_id}")
    print(f"dataset_id={dataset['dataset_id']}")
    print(f"record_batches={len(batches)}")
    print("active_routes=15")
    print("calibration_routes_ready=12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
