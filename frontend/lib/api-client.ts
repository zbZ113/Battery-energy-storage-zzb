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
  runtimeProfile: string | undefined = process.env.NEXT_PUBLIC_RUNTIME_PROFILE,
): string {
  const profile = runtimeProfile?.trim();
  if (profile && profile !== "local") {
    throw new Error("NEXT_PUBLIC_RUNTIME_PROFILE 不是受支持的运行配置");
  }
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
    if (parsed.protocol !== "http:" || !localHostnames.has(parsed.hostname)) {
      throw new Error("非本机 API 地址必须使用 HTTPS");
    }
    if (environment === "production" && profile !== "local") {
      throw new Error("生产 API 地址必须使用 HTTPS");
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

export type ReportExportFormat =
  | "markdown"
  | "json"
  | "pdf"
  | "docx"
  | "soh_csv"
  | "tool_results_json"
  | "zip";

export type ReportExportStatus =
  | "PENDING"
  | "RUNNING"
  | "READY"
  | "FAILED"
  | "EXPIRED";

export type ProjectReportStatus = "PENDING" | "RUNNING" | "READY" | "FAILED";

export interface ProjectReportExportRecord {
  export_id: string;
  report_id: string;
  format: ReportExportFormat;
  status: ReportExportStatus;
  filename: string | null;
  media_type: string | null;
  size_bytes: number | null;
  sha256: string | null;
  created_at: string;
  completed_at: string | null;
  expires_at: string | null;
}

export interface ProjectReportExportCatalog {
  report_id: string;
  project_id: string;
  run_id: string;
  report_result_id: string;
  status: ProjectReportStatus;
  template_version: string;
  created_at: string;
  completed_at: string | null;
  exports: ProjectReportExportRecord[];
}

export type ProjectReportExportCreator = (
  projectId: string,
  runId: string,
  reportResultId: string,
) => Promise<ProjectReportExportCatalog>;

export type ProjectReportExportLoader = ProjectReportExportCreator;

export async function createProjectReportExports(
  projectId: string,
  runId: string,
  reportResultId: string,
): Promise<ProjectReportExportCatalog> {
  const payload = await apiRequest<unknown>(
    projectReportExportsPath(projectId, runId, reportResultId),
    { method: "POST" },
  );
  return decodeProjectReportExportCatalog(payload);
}

export async function getProjectReportExports(
  projectId: string,
  runId: string,
  reportResultId: string,
): Promise<ProjectReportExportCatalog> {
  const payload = await apiRequest<unknown>(
    projectReportExportsPath(projectId, runId, reportResultId),
  );
  return decodeProjectReportExportCatalog(payload);
}

export function projectReportExportDownloadUrl(
  projectId: string,
  runId: string,
  reportResultId: string,
  format: ReportExportFormat,
): string {
  return `${apiBaseUrl()}${projectReportExportsPath(
    projectId,
    runId,
    reportResultId,
  )}/${encodeURIComponent(format)}`;
}

export function projectToolResultDownloadUrl(
  projectId: string,
  runId: string,
  reportResultId: string,
  resultId: string,
): string {
  return `${apiBaseUrl()}${projectReportExportsPath(
    projectId,
    runId,
    reportResultId,
  )}/tool-results/${encodeURIComponent(resultId)}.json`;
}

function projectReportExportsPath(
  projectId: string,
  runId: string,
  reportResultId: string,
): string {
  return (
    `/v1/projects/${encodeURIComponent(projectId)}`
    + `/agent/runs/${encodeURIComponent(runId)}`
    + `/reports/${encodeURIComponent(reportResultId)}/exports`
  );
}

const REPORT_EXPORT_FORMATS = new Set<ReportExportFormat>([
  "markdown",
  "json",
  "pdf",
  "docx",
  "soh_csv",
  "tool_results_json",
  "zip",
]);
const REPORT_EXPORT_STATUSES = new Set<ReportExportStatus>([
  "PENDING",
  "RUNNING",
  "READY",
  "FAILED",
  "EXPIRED",
]);
const PROJECT_REPORT_STATUSES = new Set<ProjectReportStatus>([
  "PENDING",
  "RUNNING",
  "READY",
  "FAILED",
]);
const SAFE_EXPORT_FILENAME = /^[A-Za-z0-9_.-]{1,255}$/;
const SHA256 = /^[a-f0-9]{64}$/;

function decodeProjectReportExportCatalog(value: unknown): ProjectReportExportCatalog {
  if (!isRecord(value) || !Array.isArray(value.exports)) invalidReportCatalog();
  const status = projectReportStatus(value.status);
  const reportId = requiredString(value.report_id);
  const formats = new Set<string>();
  const exports = value.exports.map((raw) => {
    if (!isRecord(raw)) invalidReportCatalog();
    const format = reportFormat(raw.format);
    if (formats.has(format)) invalidReportCatalog();
    formats.add(format);
    const exportStatus = reportExportStatus(raw.status);
    const record: ProjectReportExportRecord = {
      export_id: requiredString(raw.export_id),
      report_id: requiredString(raw.report_id),
      format,
      status: exportStatus,
      filename: nullableString(raw.filename),
      media_type: nullableString(raw.media_type),
      size_bytes: nullablePositiveInteger(raw.size_bytes),
      sha256: nullableString(raw.sha256),
      created_at: timestamp(raw.created_at),
      completed_at: nullableTimestamp(raw.completed_at),
      expires_at: nullableTimestamp(raw.expires_at),
    };
    if (record.report_id !== reportId) invalidReportCatalog();
    if (
      exportStatus === "READY"
      && (
        record.filename === null
        || !SAFE_EXPORT_FILENAME.test(record.filename)
        || record.media_type === null
        || record.media_type.length > 200
        || /[\r\n]/.test(record.media_type)
        || record.size_bytes === null
        || record.sha256 === null
        || !SHA256.test(record.sha256)
        || record.completed_at === null
        || record.expires_at === null
      )
    ) invalidReportCatalog();
    return record;
  });
  return {
    report_id: reportId,
    project_id: requiredString(value.project_id),
    run_id: requiredString(value.run_id),
    report_result_id: requiredString(value.report_result_id),
    status,
    template_version: requiredString(value.template_version),
    created_at: timestamp(value.created_at),
    completed_at: nullableTimestamp(value.completed_at),
    exports,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredString(value: unknown): string {
  if (typeof value !== "string" || value.length === 0) invalidReportCatalog();
  return value;
}

function nullableString(value: unknown): string | null {
  if (value === null) return null;
  return requiredString(value);
}

function timestamp(value: unknown): string {
  const text = requiredString(value);
  if (!Number.isFinite(Date.parse(text))) invalidReportCatalog();
  return text;
}

function nullableTimestamp(value: unknown): string | null {
  if (value === null) return null;
  return timestamp(value);
}

function nullablePositiveInteger(value: unknown): number | null {
  if (value === null) return null;
  if (!Number.isSafeInteger(value) || (value as number) < 1) invalidReportCatalog();
  return value as number;
}

function reportFormat(value: unknown): ReportExportFormat {
  if (typeof value !== "string" || !REPORT_EXPORT_FORMATS.has(value as ReportExportFormat)) {
    invalidReportCatalog();
  }
  return value as ReportExportFormat;
}

function reportExportStatus(value: unknown): ReportExportStatus {
  if (typeof value !== "string" || !REPORT_EXPORT_STATUSES.has(value as ReportExportStatus)) {
    invalidReportCatalog();
  }
  return value as ReportExportStatus;
}

function projectReportStatus(value: unknown): ProjectReportStatus {
  if (
    typeof value !== "string"
    || !PROJECT_REPORT_STATUSES.has(value as ProjectReportStatus)
  ) {
    invalidReportCatalog();
  }
  return value as ProjectReportStatus;
}

function invalidReportCatalog(): never {
  throw new Error("invalid project report export catalog");
}

function advancedCalibrationCollectionPath(projectId: string): string {
  return `/v1/projects/${encodeURIComponent(projectId)}/advanced-calibration/materializations`;
}
