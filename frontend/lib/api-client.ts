import type {
  AgentRunResultRecord,
  AgentRunCreateRequest,
  AgentRunRecord,
  AnalysisInputsRecord,
  AuthPrincipal,
  CreateAdvancedAnalysisRequest,
  DatasetRecord,
  ProjectRecord,
  RecordBatchBindingRecord,
  ToolResult,
} from "./types";
import {
  decodeActiveModelRouteList,
  decodeAdvancedCalibrationMaterialization,
  decodeAdvancedCalibrationMaterializationList,
  decodeCreateAdvancedCalibrationMaterializationResponse,
  type ActiveModelRouteSummary,
  type AdvancedCalibrationMaterialization,
  type CreateAdvancedCalibrationMaterializationRequest,
  type CreateAdvancedCalibrationMaterializationResponse,
} from "@/components/calibration-materialization-contract";

export function resolveApiBaseUrl(
  configuredUrl: string | undefined = process.env.NEXT_PUBLIC_API_BASE_URL,
  environment: string | undefined = process.env.NODE_ENV,
): string {
  const value = configuredUrl?.trim();
  if (!value) {
    if (environment === "production") {
      throw new Error("生产环境必须配置 NEXT_PUBLIC_API_BASE_URL");
    }
    return "http://localhost:8000";
  }
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("NEXT_PUBLIC_API_BASE_URL 不是有效地址");
  }
  const localHostnames = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);
  if (parsed.protocol !== "https:") {
    if (environment === "production" || parsed.protocol !== "http:" || !localHostnames.has(parsed.hostname)) {
      throw new Error("生产或非本机 API 地址必须使用 HTTPS");
    }
  }
  return value.replace(/\/$/, "");
}

function apiBaseUrl(): string {
  return resolveApiBaseUrl();
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
  ) {
    super("泉芯智寿服务暂时无法完成请求");
    this.name = "ApiError";
  }
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    ...init,
    headers,
    credentials: "include",
    cache: "no-store",
  });

  if (!response.ok) {
    let code = `http_${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string" && payload.detail.length <= 120) {
        code = payload.detail;
      }
    } catch {
      // Keep the status-only code when an upstream proxy does not return JSON.
    }
    throw new ApiError(response.status, code);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function login(username: string, password: string): Promise<AuthPrincipal> {
  return apiRequest<AuthPrincipal>("/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function getCurrentPrincipal(): Promise<AuthPrincipal> {
  return apiRequest<AuthPrincipal>("/v1/auth/me");
}

export function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<AuthPrincipal> {
  return apiRequest<AuthPrincipal>("/v1/auth/change-password", {
    method: "POST",
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
}

export function logout(): Promise<void> {
  return apiRequest<void>("/v1/auth/logout", { method: "POST" });
}

export function listProjects(): Promise<ProjectRecord[]> {
  return apiRequest<ProjectRecord[]>("/v1/projects");
}

export function getProject(projectId: string): Promise<ProjectRecord> {
  return apiRequest<ProjectRecord>(`/v1/projects/${encodeURIComponent(projectId)}`);
}

export function listProjectDatasets(projectId: string): Promise<DatasetRecord[]> {
  return apiRequest<DatasetRecord[]>(
    `/v1/projects/${encodeURIComponent(projectId)}/datasets`,
  );
}

export function listDatasetBatches(
  datasetId: string,
): Promise<RecordBatchBindingRecord[]> {
  return apiRequest<RecordBatchBindingRecord[]>(
    `/v1/datasets/${encodeURIComponent(datasetId)}/batches`,
  );
}

export function getAnalysisInputs(projectId: string): Promise<AnalysisInputsRecord> {
  return apiRequest<AnalysisInputsRecord>(
    `/v1/projects/${encodeURIComponent(projectId)}/analysis-inputs`,
  );
}

export function listProjectAgentRuns(projectId: string): Promise<AgentRunRecord[]> {
  return apiRequest<AgentRunRecord[]>(
    `/v1/projects/${encodeURIComponent(projectId)}/agent/runs`,
  );
}

export function listAgentRunResults(
  runId: string,
): Promise<AgentRunResultRecord[]> {
  return apiRequest<AgentRunResultRecord[]>(
    `/v1/agent/runs/${encodeURIComponent(runId)}/results`,
  );
}

export function createAdvancedAnalysis(
  projectId: string,
  payload: CreateAdvancedAnalysisRequest,
  idempotencyKey: string,
): Promise<AgentRunRecord> {
  return apiRequest<AgentRunRecord>(
    `/v1/projects/${encodeURIComponent(projectId)}/advanced-analyses`,
    {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: JSON.stringify(payload),
    },
  );
}

export function getAgentRun(runId: string): Promise<AgentRunRecord> {
  return apiRequest<AgentRunRecord>(`/v1/agent/runs/${encodeURIComponent(runId)}`);
}

export function createAgentRun(
  payload: AgentRunCreateRequest,
  idempotencyKey: string,
): Promise<AgentRunRecord> {
  return apiRequest<AgentRunRecord>("/v1/agent/runs", {
    method: "POST",
    headers: { "Idempotency-Key": idempotencyKey },
    body: JSON.stringify(payload),
  });
}

function actOnApproval(
  action: "approve" | "reject",
  runId: string,
  approvalId: string,
  reason: string | null,
): Promise<AgentRunRecord> {
  return apiRequest<AgentRunRecord>(
    `/v1/agent/runs/${encodeURIComponent(runId)}/${action}`,
    {
      method: "POST",
      body: JSON.stringify({ approval_id: approvalId, reason }),
    },
  );
}

export function approveAgentRun(
  runId: string,
  approvalId: string,
  reason: string | null,
): Promise<AgentRunRecord> {
  return actOnApproval("approve", runId, approvalId, reason);
}

export function rejectAgentRun(
  runId: string,
  approvalId: string,
  reason: string | null,
): Promise<AgentRunRecord> {
  return actOnApproval("reject", runId, approvalId, reason);
}

export function getAgentRunResult(runId: string, resultId: string): Promise<ToolResult> {
  return apiRequest<ToolResult>(
    `/v1/agent/runs/${encodeURIComponent(runId)}/results/${encodeURIComponent(resultId)}`,
  );
}

export function getProjectResult(
  projectId: string,
  resultId: string,
): Promise<ToolResult> {
  return apiRequest<ToolResult>(
    `/v1/projects/${encodeURIComponent(projectId)}/results/${encodeURIComponent(resultId)}`,
  );
}

export function invokeProjectTool(
  projectId: string,
  toolName: string,
  payload: Record<string, unknown>,
): Promise<ToolResult> {
  return apiRequest<ToolResult>(
    `/v1/projects/${encodeURIComponent(projectId)}/tools/${encodeURIComponent(toolName)}`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export async function listActiveModelRoutes(
  projectId: string,
): Promise<ActiveModelRouteSummary[]> {
  const payload = await apiRequest<unknown>(
    `/v1/projects/${encodeURIComponent(projectId)}/model-routes/active`,
  );
  return decodeActiveModelRouteList(payload);
}

export async function listAdvancedCalibrationMaterializations(
  projectId: string,
): Promise<AdvancedCalibrationMaterialization[]> {
  const payload = await apiRequest<unknown>(
    advancedCalibrationCollectionPath(projectId),
  );
  return decodeAdvancedCalibrationMaterializationList(payload);
}

export async function getAdvancedCalibrationMaterialization(
  projectId: string,
  materializationId: string,
): Promise<AdvancedCalibrationMaterialization> {
  const payload = await apiRequest<unknown>(
    `${advancedCalibrationCollectionPath(projectId)}/${encodeURIComponent(materializationId)}`,
  );
  return decodeAdvancedCalibrationMaterialization(payload);
}

export async function createAdvancedCalibrationMaterialization(
  projectId: string,
  payload: CreateAdvancedCalibrationMaterializationRequest,
  idempotencyKey: string,
): Promise<CreateAdvancedCalibrationMaterializationResponse> {
  const response = await apiRequest<unknown>(
    advancedCalibrationCollectionPath(projectId),
    {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: JSON.stringify(payload),
    },
  );
  return decodeCreateAdvancedCalibrationMaterializationResponse(response);
}

export function agentEventsUrl(runId: string): string {
  return `${apiBaseUrl()}/v1/agent/runs/${encodeURIComponent(runId)}/events`;
}

function advancedCalibrationCollectionPath(projectId: string): string {
  return `/v1/projects/${encodeURIComponent(projectId)}/advanced-calibration/materializations`;
}
