from __future__ import annotations

from typing import Any

from quanxin_life.application.lifetime_workflow import LifetimeDecisionWorkflowRequest


def test_competition_http_factory_shares_service_batch_store_and_verified_workflow(
    monkeypatch,
) -> None:
    from quanxin_life.application import http_application

    service = object()
    dependencies = object()
    calls: dict[str, Any] = {}

    class BatchStore:
        def register_canonical_csv(self, payload, *, registration):
            calls["registration"] = (payload, registration)
            return "batch-id"

    store = BatchStore()

    monkeypatch.setattr(
        http_application,
        "create_competition_tool_invocation_service",
        lambda received: service if received is dependencies else None,
    )

    def workflow(received_service, request, *, batch_resolver):
        calls["workflow"] = (received_service, request, batch_resolver)
        return "workflow-result"

    monkeypatch.setattr(http_application, "run_lifetime_decision_workflow", workflow)

    def api_factory(
        received_service,
        *,
        lifetime_workflow_runner,
        canonical_csv_registrar,
    ):
        calls["api"] = received_service
        calls["runner"] = lifetime_workflow_runner
        calls["registrar"] = canonical_csv_registrar
        return "fastapi-app"

    monkeypatch.setattr(http_application, "create_fastapi_app", api_factory)

    app = http_application.create_competition_fastapi_app(
        dependencies,  # type: ignore[arg-type]
        batch_store=store,  # type: ignore[arg-type]
    )
    request = LifetimeDecisionWorkflowRequest(
        record_batch_id="batch-id",
        calibration_cohort_id="calibration-id",
        policy_id="policy-id",
    )

    assert app == "fastapi-app"
    assert calls["api"] is service
    assert calls["runner"](service, request) == "workflow-result"
    assert calls["workflow"] == (service, request, store)
    assert calls["registrar"](b"payload", "registration") == "batch-id"
    assert calls["registration"] == (b"payload", "registration")
