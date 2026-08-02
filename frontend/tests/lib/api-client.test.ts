import { afterEach, describe, expect, it, vi } from "vitest";

import {
  apiRequest,
  approveAgentRun,
  changePassword,
  createAdvancedCalibrationMaterialization,
  createAdvancedAnalysis,
  createAgentRun,
  getAnalysisInputs,
  getCurrentPrincipal,
  getAgentRunResult,
  getProjectResult,
  invokeProjectTool,
  listActiveModelRoutes,
  listAdvancedCalibrationMaterializations,
  listAgentRunResults,
  listDatasetBatches,
  listProjectAgentRuns,
  listProjectDatasets,
  logout,
  rejectAgentRun,
  resolveApiBaseUrl,
} from "@/lib/api-client";

describe("apiRequest", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("always sends the opaque session cookie and requests fresh data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiRequest<{ status: string }>("/health");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/health",
      expect.objectContaining({ credentials: "include", cache: "no-store" }),
    );
  });

  it("turns backend failures into a safe typed error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "authentication_failed" }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(apiRequest("/v1/auth/me")).rejects.toEqual(
      expect.objectContaining({ status: 401, code: "authentication_failed" }),
    );
  });

  it("sends password changes through the protected auth endpoint", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(
      new Response(
        JSON.stringify({
          user_id: "user-1",
          username: "researcher",
          role: "MEMBER",
          must_change_password: false,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    ));
    vi.stubGlobal("fetch", fetchMock);

    await changePassword("temporary password", "a much safer password");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/v1/auth/change-password",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        body: JSON.stringify({
          current_password: "temporary password",
          new_password: "a much safer password",
        }),
      }),
    );
  });

  it("loads a result only through its project-authorized Agent run", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          result_id: "result-1",
          tool_name: "validate_battery_data",
          tool_version: "1.0.0",
          input_hash: "sha256:abc",
          values: {},
          provenance: [],
          warnings: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await getAgentRunResult("run/1", "result/1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/v1/agent/runs/run%2F1/results/result%2F1",
      expect.objectContaining({ credentials: "include" }),
    );
  });

  it("loads and invokes ToolResults only through the project-scoped API", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(
      new Response(
        JSON.stringify({
          result_id: "result-1",
          tool_name: "extract_early_cycle_features",
          tool_version: "advanced-input-tool-v1",
          input_hash: "a".repeat(64),
          values: {},
          uncertainty: null,
          provenance: [],
          warnings: [],
          created_at: "2026-07-26T00:00:00Z",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    ));
    vi.stubGlobal("fetch", fetchMock);

    await getProjectResult("project/1", "result/1");
    await invokeProjectTool(
      "project/1",
      "extract_early_cycle_features",
      { record_batch_id: "batch-1" },
    );

    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://localhost:8000/v1/projects/project%2F1/results/result%2F1",
    );
    expect(fetchMock.mock.calls[1]).toEqual([
      "http://localhost:8000/v1/projects/project%2F1/tools/extract_early_cycle_features",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        body: JSON.stringify({ record_batch_id: "batch-1" }),
      }),
    ]);
  });

  it("creates an Agent run with an idempotency key and bounded intent", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ run_id: "run-1" }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await createAgentRun(
      {
        project_id: "project-1",
        user_goal: "分析这批电芯的循环寿命",
        dataset_ids: ["dataset-1"],
        requested_outputs: ["cycle_life"],
      },
      "agent-run-unique-key-1",
    );

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(new Headers(init.headers).get("Idempotency-Key")).toBe("agent-run-unique-key-1");
    expect(init.method).toBe("POST");
  });

  it("uses the Task 3 catalogs and fixed advanced-analysis endpoint", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(
        jsonResponse({ project_id: "project/1", datasets: [], batches: [] }),
      )
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(
        jsonResponse({
          run_id: "run-1",
          project_id: "project/1",
          status: "RUNNING",
          created_at: "2026-08-02T10:00:00Z",
          updated_at: "2026-08-02T10:00:00Z",
        }, 202),
      );
    vi.stubGlobal("fetch", fetchMock);

    await listProjectDatasets("project/1");
    await listDatasetBatches("dataset/1");
    await getAnalysisInputs("project/1");
    await listProjectAgentRuns("project/1");
    await listAgentRunResults("run/1");
    await createAdvancedAnalysis(
      "project/1",
      {
        record_batch_id: "batch/1",
        cell_id: "cell-1",
        cutoff_cycle: 20,
      },
      "advanced-analysis-key-0001",
    );

    expect(fetchMock.mock.calls.slice(0, 5).map(([url]) => url)).toEqual([
      "http://localhost:8000/v1/projects/project%2F1/datasets",
      "http://localhost:8000/v1/datasets/dataset%2F1/batches",
      "http://localhost:8000/v1/projects/project%2F1/analysis-inputs",
      "http://localhost:8000/v1/projects/project%2F1/agent/runs",
      "http://localhost:8000/v1/agent/runs/run%2F1/results",
    ]);
    const [createUrl, createInit] = fetchMock.mock.calls[5] as [string, RequestInit];
    expect(createUrl).toBe(
      "http://localhost:8000/v1/projects/project%2F1/advanced-analyses",
    );
    expect(createInit.method).toBe("POST");
    expect(new Headers(createInit.headers).get("Idempotency-Key")).toBe(
      "advanced-analysis-key-0001",
    );
    expect(JSON.parse(String(createInit.body))).toEqual({
      record_batch_id: "batch/1",
      cell_id: "cell-1",
      cutoff_cycle: 20,
    });
  });

  it("posts approval actions and logout to their protected endpoints", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ run_id: "run-1" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));
    vi.stubGlobal("fetch", fetchMock);

    await approveAgentRun("run-1", "approval-1", "证据已复核");
    await rejectAgentRun("run-1", "approval-2", null);
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
    await logout();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://localhost:8000/v1/agent/runs/run-1/approve",
      "http://localhost:8000/v1/agent/runs/run-1/reject",
      "http://localhost:8000/v1/auth/logout",
    ]);
  });

  it("rejects an unsafe or missing production API origin", () => {
    expect(() => resolveApiBaseUrl(undefined, "production")).toThrow("NEXT_PUBLIC_API_BASE_URL");
    expect(() => resolveApiBaseUrl("http://api.example.com", "production")).toThrow("HTTPS");
    expect(resolveApiBaseUrl("http://localhost:8000", "development")).toBe(
      "http://localhost:8000",
    );
  });

  it("loads the current principal and strict active calibration routes", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            user_id: "user-1",
            username: "member@example.test",
            role: "MEMBER",
            must_change_password: false,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify([activeRoute()]),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getCurrentPrincipal()).resolves.toEqual(
      expect.objectContaining({ role: "MEMBER" }),
    );
    await expect(listActiveModelRoutes("project/1")).resolves.toEqual([
      activeRoute(),
    ]);

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://localhost:8000/v1/auth/me",
      "http://localhost:8000/v1/projects/project%2F1/model-routes/active",
    ]);
  });

  it("creates and lists identity-only calibration materializations", async () => {
    const materialization = readyMaterialization();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            materialization,
            dispatch: {
              materialization_id: materialization.materialization_id,
              task_id: "task-1",
            },
          }),
          { status: 202, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify([materialization]),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    await createAdvancedCalibrationMaterialization(
      "project/1",
      {
        task: "RUL",
        cutoff_cycle: 100,
        route_role: "COVERAGE",
      },
      "calibration-key-0001",
    );
    await listAdvancedCalibrationMaterializations("project/1");

    const [createUrl, createInit] = fetchMock.mock.calls[0] as [
      string,
      RequestInit,
    ];
    expect(createUrl).toBe(
      "http://localhost:8000/v1/projects/project%2F1/advanced-calibration/materializations",
    );
    expect(new Headers(createInit.headers).get("Idempotency-Key")).toBe(
      "calibration-key-0001",
    );
    expect(JSON.parse(String(createInit.body))).toEqual({
      task: "RUL",
      cutoff_cycle: 100,
      route_role: "COVERAGE",
    });
    expect(JSON.stringify(createInit.body)).not.toContain("observed");
    expect(JSON.stringify(createInit.body)).not.toContain("predicted");
    expect(fetchMock.mock.calls[1][0]).toBe(createUrl);
  });

  it("rejects malformed calibration readiness instead of fabricating READY", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify([
            { ...readyMaterialization(), status: "ALMOST_READY" },
          ]),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    await expect(
      listAdvancedCalibrationMaterializations("project-1"),
    ).rejects.toThrow("invalid advanced calibration");
  });
});

function activeRoute() {
  return {
    task: "RUL",
    cutoff_cycle: 100,
    route_role: "COVERAGE",
    artifact_id: "artifact-1",
    artifact_kind: "cyclepatch-direct-official-cycle-life",
    artifact_manifest_sha256: "1".repeat(64),
    model_version: "model-v1",
    data_version: "data-v1",
    split_version: "split-v1",
    feature_version: "feature-v1",
    normalization_statistics_sha256: "2".repeat(64),
    decision_event_id: "event-1",
    ledger_sequence_number: 3,
    ledger_head_sha256: "3".repeat(64),
  };
}

function readyMaterialization() {
  return {
    materialization_id: "materialization-1",
    project_id: "project-1",
    task: "RUL",
    cutoff_cycle: 100,
    route_role: "COVERAGE",
    status: "READY",
    data_version: "data-v1",
    split_version: "split-v1",
    feature_version: "feature-v1",
    artifact_id: "artifact-1",
    artifact_manifest_sha256: "1".repeat(64),
    normalization_statistics_sha256: "2".repeat(64),
    decision_event_id: "event-1",
    ledger_sequence_number: 3,
    ledger_head_sha256: "3".repeat(64),
    source_registration_id: "matr-three-batch-final-v1",
    source_identity_sha256: "4".repeat(64),
    sample_manifest_sha256: "5".repeat(64),
    sample_count: 18,
    created_at: "2026-07-27T08:00:00Z",
    started_at: "2026-07-27T08:01:00Z",
    completed_at: "2026-07-27T08:02:00Z",
    failure_code: null,
  };
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
