export type AdvancedModelTask = "RUL" | "SOH";

export type SupportedCutoffCycle = 20 | 50 | 100 | 150;

export type AdvancedModelRouteRole =
  | "DEFAULT"
  | "POINT_ACCURACY"
  | "COVERAGE"
  | "MEAN_ACCURACY"
  | "TAIL_EFFICIENCY";

export type AdvancedCalibrationMaterializationStatus =
  | "PENDING"
  | "RUNNING"
  | "READY"
  | "FAILED"
  | "STALE";

export interface ActiveModelRouteSummary {
  task: AdvancedModelTask;
  cutoff_cycle: SupportedCutoffCycle;
  route_role: AdvancedModelRouteRole;
  artifact_id: string;
  artifact_kind: string;
  artifact_manifest_sha256: string;
  model_version: string;
  data_version: string;
  split_version: string;
  feature_version: string;
  normalization_statistics_sha256: string;
  decision_event_id: string;
  ledger_sequence_number: number;
  ledger_head_sha256: string;
}

export interface AdvancedCalibrationMaterialization {
  materialization_id: string;
  project_id: string;
  task: AdvancedModelTask;
  cutoff_cycle: SupportedCutoffCycle;
  route_role: AdvancedModelRouteRole;
  status: AdvancedCalibrationMaterializationStatus;
  data_version: string;
  split_version: string;
  feature_version: string;
  artifact_id: string;
  artifact_manifest_sha256: string;
  normalization_statistics_sha256: string;
  decision_event_id: string;
  ledger_sequence_number: number;
  ledger_head_sha256: string;
  source_registration_id: string;
  source_identity_sha256: string;
  sample_manifest_sha256: string | null;
  sample_count: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  failure_code: string | null;
}

export interface CreateAdvancedCalibrationMaterializationRequest {
  task: AdvancedModelTask;
  cutoff_cycle: SupportedCutoffCycle;
  route_role: AdvancedModelRouteRole;
}

export interface AdvancedCalibrationDispatchReceipt {
  materialization_id: string;
  task_id: string;
}

export interface CreateAdvancedCalibrationMaterializationResponse {
  materialization: AdvancedCalibrationMaterialization;
  dispatch: AdvancedCalibrationDispatchReceipt;
}

const TASKS = new Set<AdvancedModelTask>(["RUL", "SOH"]);
const ROUTE_ROLES = new Set<AdvancedModelRouteRole>([
  "DEFAULT",
  "POINT_ACCURACY",
  "COVERAGE",
  "MEAN_ACCURACY",
  "TAIL_EFFICIENCY",
]);
const STATUSES = new Set<AdvancedCalibrationMaterializationStatus>([
  "PENDING",
  "RUNNING",
  "READY",
  "FAILED",
  "STALE",
]);
const SUPPORTED_CUTOFF_CYCLES = new Set<number>([20, 50, 100, 150]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const TIMESTAMP_PATTERN =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$/;

const ACTIVE_ROUTE_KEYS = [
  "artifact_id",
  "artifact_kind",
  "artifact_manifest_sha256",
  "cutoff_cycle",
  "data_version",
  "decision_event_id",
  "feature_version",
  "ledger_head_sha256",
  "ledger_sequence_number",
  "model_version",
  "normalization_statistics_sha256",
  "route_role",
  "split_version",
  "task",
] as const;

const MATERIALIZATION_KEYS = [
  "artifact_id",
  "artifact_manifest_sha256",
  "completed_at",
  "created_at",
  "cutoff_cycle",
  "data_version",
  "decision_event_id",
  "failure_code",
  "feature_version",
  "ledger_head_sha256",
  "ledger_sequence_number",
  "materialization_id",
  "normalization_statistics_sha256",
  "project_id",
  "route_role",
  "sample_count",
  "sample_manifest_sha256",
  "source_identity_sha256",
  "source_registration_id",
  "split_version",
  "started_at",
  "status",
  "task",
] as const;

function invalid(detail: string): never {
  throw new Error(`invalid advanced calibration: ${detail}`);
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return invalid(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
  label: string,
): void {
  const allowed = new Set(keys);
  const actual = Object.keys(value);
  if (
    actual.length !== keys.length
    || actual.some((key) => !allowed.has(key))
  ) {
    invalid(`${label} contains missing or unsupported fields`);
  }
}

function nonblank(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() !== value || value.length === 0) {
    return invalid(`${label} must be a nonblank string`);
  }
  return value;
}

function nullableNonblank(value: unknown, label: string): string | null {
  return value === null ? null : nonblank(value, label);
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    return invalid(`${label} must be a positive integer`);
  }
  return value as number;
}

export function isSupportedCutoffCycle(
  value: number,
): value is SupportedCutoffCycle {
  return SUPPORTED_CUTOFF_CYCLES.has(value);
}

function cutoffCycle(value: unknown): SupportedCutoffCycle {
  if (
    !Number.isInteger(value)
    || !isSupportedCutoffCycle(value as number)
  ) {
    return invalid("cutoff_cycle is unsupported");
  }
  return value as SupportedCutoffCycle;
}

function nonnegativeInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    return invalid(`${label} must be a nonnegative integer`);
  }
  return value as number;
}

function sha256(value: unknown, label: string): string {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    return invalid(`${label} must be a canonical SHA-256`);
  }
  return value;
}

function nullableSha256(value: unknown, label: string): string | null {
  return value === null ? null : sha256(value, label);
}

function timestamp(value: unknown, label: string): string {
  if (
    typeof value !== "string"
    || !TIMESTAMP_PATTERN.test(value)
    || Number.isNaN(Date.parse(value))
  ) {
    return invalid(`${label} must be a timezone-aware ISO timestamp`);
  }
  return value;
}

function nullableTimestamp(value: unknown, label: string): string | null {
  return value === null ? null : timestamp(value, label);
}

function task(value: unknown): AdvancedModelTask {
  if (typeof value !== "string" || !TASKS.has(value as AdvancedModelTask)) {
    return invalid("task is unsupported");
  }
  return value as AdvancedModelTask;
}

function routeRole(value: unknown): AdvancedModelRouteRole {
  if (
    typeof value !== "string"
    || !ROUTE_ROLES.has(value as AdvancedModelRouteRole)
  ) {
    return invalid("route_role is unsupported");
  }
  return value as AdvancedModelRouteRole;
}

function status(value: unknown): AdvancedCalibrationMaterializationStatus {
  if (
    typeof value !== "string"
    || !STATUSES.has(value as AdvancedCalibrationMaterializationStatus)
  ) {
    return invalid("status is unsupported");
  }
  return value as AdvancedCalibrationMaterializationStatus;
}

export function decodeActiveModelRouteList(
  value: unknown,
): ActiveModelRouteSummary[] {
  if (!Array.isArray(value)) invalid("active route list must be an array");
  return value.map((item) => decodeActiveModelRoute(item));
}

function decodeActiveModelRoute(value: unknown): ActiveModelRouteSummary {
  const item = record(value, "active route");
  exactKeys(item, ACTIVE_ROUTE_KEYS, "active route");
  return {
    artifact_id: nonblank(item.artifact_id, "artifact_id"),
    artifact_kind: nonblank(item.artifact_kind, "artifact_kind"),
    artifact_manifest_sha256: sha256(
      item.artifact_manifest_sha256,
      "artifact_manifest_sha256",
    ),
    cutoff_cycle: cutoffCycle(item.cutoff_cycle),
    data_version: nonblank(item.data_version, "data_version"),
    decision_event_id: nonblank(item.decision_event_id, "decision_event_id"),
    feature_version: nonblank(item.feature_version, "feature_version"),
    ledger_head_sha256: sha256(
      item.ledger_head_sha256,
      "ledger_head_sha256",
    ),
    ledger_sequence_number: positiveInteger(
      item.ledger_sequence_number,
      "ledger_sequence_number",
    ),
    model_version: nonblank(item.model_version, "model_version"),
    normalization_statistics_sha256: sha256(
      item.normalization_statistics_sha256,
      "normalization_statistics_sha256",
    ),
    route_role: routeRole(item.route_role),
    split_version: nonblank(item.split_version, "split_version"),
    task: task(item.task),
  };
}

export function decodeAdvancedCalibrationMaterialization(
  value: unknown,
): AdvancedCalibrationMaterialization {
  const item = record(value, "materialization");
  exactKeys(item, MATERIALIZATION_KEYS, "materialization");
  const decoded: AdvancedCalibrationMaterialization = {
    artifact_id: nonblank(item.artifact_id, "artifact_id"),
    artifact_manifest_sha256: sha256(
      item.artifact_manifest_sha256,
      "artifact_manifest_sha256",
    ),
    completed_at: nullableTimestamp(item.completed_at, "completed_at"),
    created_at: timestamp(item.created_at, "created_at"),
    cutoff_cycle: cutoffCycle(item.cutoff_cycle),
    data_version: nonblank(item.data_version, "data_version"),
    decision_event_id: nonblank(item.decision_event_id, "decision_event_id"),
    failure_code: nullableNonblank(item.failure_code, "failure_code"),
    feature_version: nonblank(item.feature_version, "feature_version"),
    ledger_head_sha256: sha256(
      item.ledger_head_sha256,
      "ledger_head_sha256",
    ),
    ledger_sequence_number: positiveInteger(
      item.ledger_sequence_number,
      "ledger_sequence_number",
    ),
    materialization_id: nonblank(
      item.materialization_id,
      "materialization_id",
    ),
    normalization_statistics_sha256: sha256(
      item.normalization_statistics_sha256,
      "normalization_statistics_sha256",
    ),
    project_id: nonblank(item.project_id, "project_id"),
    route_role: routeRole(item.route_role),
    sample_count: nonnegativeInteger(item.sample_count, "sample_count"),
    sample_manifest_sha256: nullableSha256(
      item.sample_manifest_sha256,
      "sample_manifest_sha256",
    ),
    source_identity_sha256: sha256(
      item.source_identity_sha256,
      "source_identity_sha256",
    ),
    source_registration_id: nonblank(
      item.source_registration_id,
      "source_registration_id",
    ),
    split_version: nonblank(item.split_version, "split_version"),
    started_at: nullableTimestamp(item.started_at, "started_at"),
    status: status(item.status),
    task: task(item.task),
  };
  validateMaterializationState(decoded);
  return decoded;
}

function validateMaterializationState(
  value: AdvancedCalibrationMaterialization,
): void {
  const retainsAuditedSamples =
    value.status === "READY" || value.status === "STALE";
  if (
    retainsAuditedSamples
    && (
      value.completed_at === null
      || value.sample_manifest_sha256 === null
      || value.sample_count <= 0
    )
  ) {
    invalid("audited materialization lacks complete sample evidence");
  }
  if (
    !retainsAuditedSamples
    && (value.sample_manifest_sha256 !== null || value.sample_count !== 0)
  ) {
    invalid("unfinished materialization exposes sample evidence");
  }
  if (value.status === "READY" && value.failure_code !== null) {
    invalid("READY materialization contains a failure code");
  }
  if (value.status === "FAILED" && value.failure_code === null) {
    invalid("FAILED materialization lacks a failure code");
  }
  if (value.status === "STALE" && value.failure_code === null) {
    invalid("STALE materialization lacks an obsolescence code");
  }
}

export function decodeAdvancedCalibrationMaterializationList(
  value: unknown,
): AdvancedCalibrationMaterialization[] {
  if (!Array.isArray(value)) invalid("materialization list must be an array");
  return value.map((item) => decodeAdvancedCalibrationMaterialization(item));
}

export function decodeCreateAdvancedCalibrationMaterializationResponse(
  value: unknown,
): CreateAdvancedCalibrationMaterializationResponse {
  const response = record(value, "create response");
  exactKeys(response, ["dispatch", "materialization"], "create response");
  const materialization = decodeAdvancedCalibrationMaterialization(
    response.materialization,
  );
  const dispatch = record(response.dispatch, "dispatch");
  exactKeys(dispatch, ["materialization_id", "task_id"], "dispatch");
  const decodedDispatch: AdvancedCalibrationDispatchReceipt = {
    materialization_id: nonblank(
      dispatch.materialization_id,
      "dispatch.materialization_id",
    ),
    task_id: nonblank(dispatch.task_id, "dispatch.task_id"),
  };
  if (
    decodedDispatch.materialization_id !== materialization.materialization_id
  ) {
    invalid("dispatch identity does not match materialization");
  }
  return { dispatch: decodedDispatch, materialization };
}
