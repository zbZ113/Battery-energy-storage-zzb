import { afterEach, describe, expect, it, vi } from "vitest";

import {
  apiRequest,
  approveAgentRun,
  changePassword,
  createAgentRun,
  getAgentRunResult,
  getProjectResult,
  invokeProjectTool,
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
});
