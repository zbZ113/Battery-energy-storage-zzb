from __future__ import annotations

from typing import Any

import pytest

from quanxin_life.application.lifetime_workflow import LifetimeDecisionWorkflowRequest


def test_competition_http_factory_shares_service_and_project_batch_adapter(
    monkeypatch,
) -> None:
    from quanxin_life.application import http_application

    service = object()
    dependencies = object()
    calls: dict[str, Any] = {}

    store = object()
    auth_adapter = object()
    project_adapter = object()
    dataset_adapter = object()
    record_batch_adapter = object()
    experiment_adapter = object()
    model_artifact_adapter = object()
    model_route_adapter = object()
    agent_run_adapter = object()
    knowledge_adapter = object()

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
        auth_adapter,
        project_adapter,
        dataset_adapter,
        record_batch_adapter,
        experiment_adapter,
        model_artifact_adapter,
        model_route_adapter,
        agent_run_adapter,
        knowledge_adapter,
    ):
        calls["api"] = received_service
        calls["runner"] = lifetime_workflow_runner
        calls["auth_adapter"] = auth_adapter
        calls["project_adapter"] = project_adapter
        calls["dataset_adapter"] = dataset_adapter
        calls["record_batch_adapter"] = record_batch_adapter
        calls["experiment_adapter"] = experiment_adapter
        calls["model_artifact_adapter"] = model_artifact_adapter
        calls["model_route_adapter"] = model_route_adapter
        calls["agent_run_adapter"] = agent_run_adapter
        calls["knowledge_adapter"] = knowledge_adapter
        return "fastapi-app"

    monkeypatch.setattr(http_application, "create_fastapi_app", api_factory)

    app = http_application.create_competition_fastapi_app(
        dependencies,  # type: ignore[arg-type]
        batch_store=store,  # type: ignore[arg-type]
        auth_adapter=auth_adapter,  # type: ignore[arg-type]
        project_adapter=project_adapter,  # type: ignore[arg-type]
        dataset_adapter=dataset_adapter,  # type: ignore[arg-type]
        record_batch_adapter=record_batch_adapter,  # type: ignore[arg-type]
        experiment_adapter=experiment_adapter,  # type: ignore[arg-type]
        model_artifact_adapter=model_artifact_adapter,  # type: ignore[arg-type]
        model_route_adapter=model_route_adapter,  # type: ignore[arg-type]
        agent_run_adapter=agent_run_adapter,  # type: ignore[arg-type]
        knowledge_adapter=knowledge_adapter,  # type: ignore[arg-type]
    )
    request = LifetimeDecisionWorkflowRequest(
        record_batch_id="batch-id",
        calibration_cohort_id="calibration-id",
        policy_id="policy-id",
    )

    assert app == "fastapi-app"
    assert calls["api"] is service
    assert calls["auth_adapter"] is auth_adapter
    assert calls["project_adapter"] is project_adapter
    assert calls["dataset_adapter"] is dataset_adapter
    assert calls["record_batch_adapter"] is record_batch_adapter
    assert calls["experiment_adapter"] is experiment_adapter
    assert calls["model_artifact_adapter"] is model_artifact_adapter
    assert calls["model_route_adapter"] is model_route_adapter
    assert calls["agent_run_adapter"] is agent_run_adapter
    assert calls["knowledge_adapter"] is knowledge_adapter
    assert calls["runner"](service, request) == "workflow-result"
    assert calls["workflow"] == (service, request, store)


def test_competition_http_factory_rejects_an_explicitly_missing_auth_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="auth_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=None,  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_an_explicitly_missing_project_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="project_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=None,  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_an_explicitly_missing_dataset_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="dataset_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=None,  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_missing_record_batch_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="record_batch_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=None,  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_an_explicitly_missing_experiment_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="experiment_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=None,  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_an_explicitly_missing_agent_run_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="agent_run_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=None,  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_an_explicitly_missing_knowledge_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="knowledge_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=None,  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_missing_model_artifact_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="model_artifact_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=None,  # type: ignore[arg-type]
            model_route_adapter=object(),  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )


def test_competition_http_factory_rejects_missing_model_route_adapter() -> None:
    from quanxin_life.application.http_application import create_competition_fastapi_app

    with pytest.raises(ValueError, match="model_route_adapter"):
        create_competition_fastapi_app(
            object(),  # type: ignore[arg-type]
            batch_store=object(),  # type: ignore[arg-type]
            auth_adapter=object(),  # type: ignore[arg-type]
            project_adapter=object(),  # type: ignore[arg-type]
            dataset_adapter=object(),  # type: ignore[arg-type]
            record_batch_adapter=object(),  # type: ignore[arg-type]
            experiment_adapter=object(),  # type: ignore[arg-type]
            model_artifact_adapter=object(),  # type: ignore[arg-type]
            model_route_adapter=None,  # type: ignore[arg-type]
            agent_run_adapter=object(),  # type: ignore[arg-type]
            knowledge_adapter=object(),  # type: ignore[arg-type]
        )
