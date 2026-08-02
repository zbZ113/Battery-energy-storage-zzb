import type {
  AgentRunRecord,
  AgentRunResultRecord,
  AnalysisInputsRecord,
  DatasetRecord,
  ProvenanceRecord,
  RecordBatchBindingRecord,
  ToolResult,
} from "@/lib/types";

export const FIXED_ADVANCED_STEP_IDS = [
  "advanced-input",
  "rul-point",
  "rul-coverage",
  "soh",
  "rul-calibration",
  "rul-interval",
  "soh-calibration",
  "soh-band",
  "report",
] as const;

export type FixedAdvancedStepId = (typeof FIXED_ADVANCED_STEP_IDS)[number];

type AgentRunStatus =
  | "PLANNING"
  | "RUNNING"
  | "AWAITING_APPROVAL"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED"
  | "FALLBACK";

interface FixedAdvancedIntent {
  intent_id: string;
  project_id: string;
  goal: string;
  dataset_ids: [string];
  requested_outputs: ["advanced_single_cell_analysis"];
  created_at: string;
}

interface FixedAdvancedPlanStep {
  step_id: FixedAdvancedStepId;
  role: "data_quality" | "lifetime" | "supervisor";
  tool_name: string;
  depends_on: string[];
  input_references: Record<string, string>;
  requires_approval: false;
  failure_policy: "STOP";
}

interface FixedAdvancedPlan {
  plan_version: "supervisor-planner-v1";
  intent_id: string;
  steps: FixedAdvancedPlanStep[];
  planning_mode: "FIXED_FALLBACK";
  plan_hash: string;
  created_at: string;
}

export interface FixedAdvancedAgentRun extends AgentRunRecord {
  run_id: string;
  project_id: string;
  created_by_user_id: string;
  status: AgentRunStatus;
  planning_mode: "FIXED_FALLBACK";
  intent: FixedAdvancedIntent;
  plan: FixedAdvancedPlan;
  dispatch_status: "PENDING" | "DISPATCHED";
  dispatch_task_id: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface AdvancedAnalysisResultIds {
  rulResultId: string | null;
  sohResultId: string | null;
  rulConformalResultId: string | null;
  sohConformalResultId: string | null;
  reportResultId: string | null;
}

const FIXED_STEP_TEMPLATES: readonly FixedAdvancedPlanStep[] = [
  {
    step_id: "advanced-input",
    role: "data_quality",
    tool_name: "extract_early_cycle_features",
    depends_on: [],
    input_references: { record_batch_id: "intent.dataset_ids[0]" },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "rul-point",
    role: "lifetime",
    tool_name: "predict_cycle_life",
    depends_on: ["advanced-input"],
    input_references: {
      upstream_result_id: "step.advanced-input.result_id",
      route_role: "context.rul_point_route_role",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "rul-coverage",
    role: "lifetime",
    tool_name: "predict_cycle_life",
    depends_on: ["advanced-input"],
    input_references: {
      upstream_result_id: "step.advanced-input.result_id",
      route_role: "context.rul_coverage_route_role",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "soh",
    role: "lifetime",
    tool_name: "predict_soh_trajectory",
    depends_on: ["advanced-input"],
    input_references: {
      upstream_result_id: "step.advanced-input.result_id",
      route_role: "context.soh_route_role",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "rul-calibration",
    role: "lifetime",
    tool_name: "calibrate_prediction_interval",
    depends_on: [],
    input_references: {
      operation: "context.conformal_calibrate_operation",
      task: "context.rul_task",
      route_role: "context.rul_coverage_route_role",
      alpha: "context.conformal_alpha",
      calibration_sample_result_ids: "context.rul_calibration_sample_result_ids",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "rul-interval",
    role: "lifetime",
    tool_name: "calibrate_prediction_interval",
    depends_on: ["rul-coverage", "rul-calibration"],
    input_references: {
      operation: "context.conformal_issue_operation",
      task: "context.rul_task",
      route_role: "context.rul_coverage_route_role",
      prediction_result_id: "step.rul-coverage.result_id",
      calibration_result_id: "step.rul-calibration.result_id",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "soh-calibration",
    role: "lifetime",
    tool_name: "calibrate_prediction_interval",
    depends_on: [],
    input_references: {
      operation: "context.conformal_calibrate_operation",
      task: "context.soh_task",
      route_role: "context.soh_route_role",
      alpha: "context.conformal_alpha",
      calibration_sample_result_ids: "context.soh_calibration_sample_result_ids",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "soh-band",
    role: "lifetime",
    tool_name: "calibrate_prediction_interval",
    depends_on: ["soh", "soh-calibration"],
    input_references: {
      operation: "context.conformal_issue_operation",
      task: "context.soh_task",
      route_role: "context.soh_route_role",
      prediction_result_id: "step.soh.result_id",
      calibration_result_id: "step.soh-calibration.result_id",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
  {
    step_id: "report",
    role: "supervisor",
    tool_name: "generate_audited_report",
    depends_on: ["rul-point", "soh", "rul-interval", "soh-band"],
    input_references: {
      rul_result_id: "step.rul-point.result_id",
      soh_result_id: "step.soh.result_id",
      rul_conformal_result_id: "step.rul-interval.result_id",
      soh_conformal_result_id: "step.soh-band.result_id",
    },
    requires_approval: false,
    failure_policy: "STOP",
  },
];

const RUN_STATUSES = new Set<AgentRunStatus>([
  "PLANNING",
  "RUNNING",
  "AWAITING_APPROVAL",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
  "FALLBACK",
]);
const SOURCE_KINDS = new Set(["OBSERVED", "PREDICTED", "NEWLY_OBSERVED", "SIMULATED"]);
const TOOL_RESULT_KEYS = [
  "result_id",
  "tool_name",
  "tool_version",
  "model_version",
  "data_version",
  "feature_version",
  "input_hash",
  "values",
  "uncertainty",
  "warnings",
  "provenance",
  "created_at",
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function record(value: unknown, keys: readonly string[], label: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${label} must be an object`);
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} contains missing or unknown fields`);
  }
  return value;
}

function nonblank(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a nonblank string`);
  }
  return value;
}

function nullableString(value: unknown, label: string): string | null {
  return value === null ? null : nonblank(value, label);
}

function timestamp(value: unknown, label: string): string {
  const normalized = nonblank(value, label);
  if (!/(?:Z|[+-]\d{2}:\d{2})$/.test(normalized) || Number.isNaN(Date.parse(normalized))) {
    throw new Error(`${label} must include a timezone`);
  }
  return normalized;
}

function sha256(value: unknown, label: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256`);
  }
  return value;
}

function uuid(value: unknown, label: string): string {
  const normalized = nonblank(value, label);
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(normalized)) {
    throw new Error(`${label} must be a UUID`);
  }
  return normalized;
}

function integer(value: unknown, label: string, minimum = 1): number {
  if (!Number.isInteger(value) || (value as number) < minimum) {
    throw new Error(`${label} must be an integer >= ${minimum}`);
  }
  return value as number;
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value.map((item, index) => nonblank(item, `${label}[${index}]`));
}

function isJsonValue(value: unknown): boolean {
  if (value === null || typeof value === "string" || typeof value === "boolean") return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(isJsonValue);
  return isRecord(value) && Object.values(value).every(isJsonValue);
}

function jsonRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value) || !isJsonValue(value)) throw new Error(`${label} must be a JSON object`);
  return value;
}

function canonicalJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalJson);
  if (!isRecord(value)) return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, canonicalJson(item)]),
  );
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonicalJson(left)) === JSON.stringify(canonicalJson(right));
}

function decodeDataset(value: unknown, expectedProjectId: string): DatasetRecord {
  const item = record(value, [
    "dataset_id", "project_id", "name", "data_version", "schema_version", "status",
    "manifest_uri", "manifest_sha256", "created_at", "frozen_at",
  ], "analysis dataset");
  const projectId = nonblank(item.project_id, "dataset project_id");
  if (projectId !== expectedProjectId) throw new Error("dataset belongs to another project");
  if (item.status !== "FROZEN") throw new Error("analysis dataset must be FROZEN");
  const frozenAt = timestamp(item.frozen_at, "dataset frozen_at");
  const manifestSha = item.manifest_sha256 === null
    ? null
    : sha256(item.manifest_sha256, "dataset manifest_sha256");
  return {
    dataset_id: nonblank(item.dataset_id, "dataset_id"),
    project_id: projectId,
    name: nonblank(item.name, "dataset name"),
    data_version: nonblank(item.data_version, "dataset data_version"),
    schema_version: nonblank(item.schema_version, "dataset schema_version"),
    status: "FROZEN",
    manifest_uri: nullableString(item.manifest_uri, "dataset manifest_uri"),
    manifest_sha256: manifestSha,
    created_at: timestamp(item.created_at, "dataset created_at"),
    frozen_at: frozenAt,
  };
}

function decodeBatch(value: unknown, expectedProjectId: string): RecordBatchBindingRecord {
  const item = record(value, [
    "record_batch_id", "binding_schema_version", "project_id", "dataset_id",
    "source_manifest_sha256", "content_dataset_id", "dataset_schema_version", "cell_id",
    "cutoff_cycle", "data_version", "split_version", "feature_version", "created_at",
  ], "analysis record batch");
  const projectId = nonblank(item.project_id, "record batch project_id");
  if (projectId !== expectedProjectId) throw new Error("record batch belongs to another project");
  if (item.binding_schema_version !== "record-batch-binding-v1") {
    throw new Error("record batch binding schema is unsupported");
  }
  return {
    record_batch_id: nonblank(item.record_batch_id, "record_batch_id"),
    binding_schema_version: "record-batch-binding-v1",
    project_id: projectId,
    dataset_id: nonblank(item.dataset_id, "record batch dataset_id"),
    source_manifest_sha256: sha256(item.source_manifest_sha256, "source_manifest_sha256"),
    content_dataset_id: nonblank(item.content_dataset_id, "content_dataset_id"),
    dataset_schema_version: nonblank(item.dataset_schema_version, "dataset_schema_version"),
    cell_id: nonblank(item.cell_id, "cell_id"),
    cutoff_cycle: integer(item.cutoff_cycle, "cutoff_cycle"),
    data_version: nonblank(item.data_version, "record batch data_version"),
    split_version: nonblank(item.split_version, "split_version"),
    feature_version: nonblank(item.feature_version, "feature_version"),
    created_at: timestamp(item.created_at, "record batch created_at"),
  };
}

export function decodeAnalysisInputs(value: unknown, expectedProjectId: string): AnalysisInputsRecord {
  const projectId = nonblank(expectedProjectId, "expected project_id");
  const payload = record(value, ["project_id", "datasets", "batches"], "analysis inputs");
  if (payload.project_id !== projectId) throw new Error("analysis inputs belong to another project");
  if (!Array.isArray(payload.datasets) || !Array.isArray(payload.batches)) {
    throw new Error("analysis input catalogs must be arrays");
  }
  const datasets = payload.datasets.map((item) => decodeDataset(item, projectId));
  const batches = payload.batches.map((item) => decodeBatch(item, projectId));
  const datasetsById = new Map<string, DatasetRecord>();
  for (const dataset of datasets) {
    if (datasetsById.has(dataset.dataset_id)) throw new Error("analysis dataset IDs must be unique");
    datasetsById.set(dataset.dataset_id, dataset);
  }
  const batchIds = new Set<string>();
  const referencedDatasetIds = new Set<string>();
  for (const batch of batches) {
    if (batchIds.has(batch.record_batch_id)) throw new Error("record batch IDs must be unique");
    batchIds.add(batch.record_batch_id);
    const dataset = datasetsById.get(batch.dataset_id);
    if (!dataset) throw new Error("record batch references an unknown dataset");
    if (
      batch.data_version !== dataset.data_version
      || batch.dataset_schema_version !== dataset.schema_version
    ) {
      throw new Error("record batch snapshot differs from its frozen dataset");
    }
    referencedDatasetIds.add(batch.dataset_id);
  }
  if (datasets.some((dataset) => !referencedDatasetIds.has(dataset.dataset_id))) {
    throw new Error("analysis dataset has no selectable record batch");
  }
  return { project_id: projectId, datasets, batches };
}

function decodeStringMapping(value: unknown, label: string): Record<string, string> {
  const mapping = jsonRecord(value, label);
  const decoded: Record<string, string> = {};
  for (const [key, item] of Object.entries(mapping)) {
    decoded[nonblank(key, `${label} key`)] = nonblank(item, `${label}.${key}`);
  }
  return decoded;
}

function decodeRunShape(value: unknown): FixedAdvancedAgentRun {
  const item = record(value, [
    "run_id", "project_id", "created_by_user_id", "status", "planning_mode", "intent",
    "plan", "dispatch_status", "dispatch_task_id", "created_at", "updated_at", "completed_at",
  ], "Agent run");
  const status = nonblank(item.status, "Agent run status") as AgentRunStatus;
  if (!RUN_STATUSES.has(status)) throw new Error("Agent run status is unsupported");
  if (item.planning_mode !== "FIXED_FALLBACK") throw new Error("advanced run must use fixed planning");

  const rawIntent = record(item.intent, [
    "intent_id", "project_id", "goal", "dataset_ids", "requested_outputs", "created_at",
  ], "Agent intent");
  const datasetIds = stringArray(rawIntent.dataset_ids, "intent dataset_ids");
  const requestedOutputs = stringArray(rawIntent.requested_outputs, "intent requested_outputs");
  if (datasetIds.length !== 1) throw new Error("advanced run must bind exactly one record batch");
  if (requestedOutputs.length !== 1 || requestedOutputs[0] !== "advanced_single_cell_analysis") {
    throw new Error("Agent run is not an advanced single-cell analysis");
  }
  const intent: FixedAdvancedIntent = {
    intent_id: uuid(rawIntent.intent_id, "intent_id"),
    project_id: nonblank(rawIntent.project_id, "intent project_id"),
    goal: nonblank(rawIntent.goal, "intent goal"),
    dataset_ids: [datasetIds[0]],
    requested_outputs: ["advanced_single_cell_analysis"],
    created_at: timestamp(rawIntent.created_at, "intent created_at"),
  };

  const rawPlan = record(item.plan, [
    "plan_version", "intent_id", "steps", "planning_mode", "plan_hash", "created_at",
  ], "Agent plan");
  if (rawPlan.plan_version !== "supervisor-planner-v1" || rawPlan.planning_mode !== "FIXED_FALLBACK") {
    throw new Error("advanced Agent plan identity is unsupported");
  }
  if (rawPlan.intent_id !== intent.intent_id) throw new Error("Agent plan intent_id mismatch");
  if (!Array.isArray(rawPlan.steps) || rawPlan.steps.length !== FIXED_STEP_TEMPLATES.length) {
    throw new Error("advanced Agent plan must contain all nine steps");
  }
  const steps = rawPlan.steps.map((rawStep, index): FixedAdvancedPlanStep => {
    const step = record(rawStep, [
      "step_id", "role", "tool_name", "depends_on", "input_references",
      "requires_approval", "failure_policy",
    ], `Agent plan step ${index + 1}`);
    const decoded: FixedAdvancedPlanStep = {
      step_id: nonblank(step.step_id, "step_id") as FixedAdvancedStepId,
      role: nonblank(step.role, "step role") as FixedAdvancedPlanStep["role"],
      tool_name: nonblank(step.tool_name, "step tool_name"),
      depends_on: stringArray(step.depends_on, "step depends_on"),
      input_references: decodeStringMapping(step.input_references, "step input_references"),
      requires_approval: step.requires_approval as false,
      failure_policy: step.failure_policy as "STOP",
    };
    if (step.requires_approval !== false || step.failure_policy !== "STOP") {
      throw new Error("advanced Agent steps must stop without approval");
    }
    if (!sameJson(decoded, FIXED_STEP_TEMPLATES[index])) {
      throw new Error("advanced Agent plan step differs from the fixed template");
    }
    return decoded;
  });
  const plan: FixedAdvancedPlan = {
    plan_version: "supervisor-planner-v1",
    intent_id: intent.intent_id,
    steps,
    planning_mode: "FIXED_FALLBACK",
    plan_hash: sha256(rawPlan.plan_hash, "plan_hash"),
    created_at: timestamp(rawPlan.created_at, "plan created_at"),
  };

  const projectId = nonblank(item.project_id, "Agent run project_id");
  if (intent.project_id !== projectId) throw new Error("Agent intent belongs to another project");
  const completedAt = item.completed_at === null
    ? null
    : timestamp(item.completed_at, "Agent run completed_at");
  if ((status === "COMPLETED" || status === "FAILED" || status === "CANCELLED") !== (completedAt !== null)) {
    throw new Error("Agent run terminal timestamp is inconsistent with status");
  }
  if (item.dispatch_status !== "PENDING" && item.dispatch_status !== "DISPATCHED") {
    throw new Error("Agent dispatch status is unsupported");
  }
  return {
    run_id: nonblank(item.run_id, "run_id"),
    project_id: projectId,
    created_by_user_id: nonblank(item.created_by_user_id, "created_by_user_id"),
    status,
    planning_mode: "FIXED_FALLBACK",
    intent,
    plan,
    dispatch_status: item.dispatch_status,
    dispatch_task_id: nullableString(item.dispatch_task_id, "dispatch_task_id"),
    created_at: timestamp(item.created_at, "Agent run created_at"),
    updated_at: timestamp(item.updated_at, "Agent run updated_at"),
    completed_at: completedAt,
  };
}

export function isFixedAdvancedRun(value: unknown): value is FixedAdvancedAgentRun {
  try {
    decodeRunShape(value);
    return true;
  } catch {
    return false;
  }
}

export function decodeAgentRun(
  value: unknown,
  expectedProjectId: string,
  expectedRecordBatchId: string,
): FixedAdvancedAgentRun {
  const run = decodeRunShape(value);
  if (run.project_id !== nonblank(expectedProjectId, "expected project_id")) {
    throw new Error("Agent run belongs to another project");
  }
  if (run.intent.dataset_ids[0] !== nonblank(expectedRecordBatchId, "expected record_batch_id")) {
    throw new Error("Agent run belongs to another record batch");
  }
  return run;
}

function decodeProvenance(value: unknown): ProvenanceRecord {
  const item = record(value, [
    "source_id", "source_kind", "uri", "sha256", "description", "created_at",
  ], "ToolResult provenance");
  const sourceKind = nonblank(item.source_kind, "source_kind");
  if (!SOURCE_KINDS.has(sourceKind)) throw new Error("ToolResult source_kind is unsupported");
  return {
    source_id: nonblank(item.source_id, "source_id"),
    source_kind: sourceKind,
    uri: nonblank(item.uri, "provenance uri"),
    sha256: sha256(item.sha256, "provenance sha256"),
    description: nonblank(item.description, "provenance description"),
    created_at: timestamp(item.created_at, "provenance created_at"),
  };
}

function decodeToolResult(value: unknown): ToolResult {
  const item = record(value, TOOL_RESULT_KEYS, "ToolResult");
  if (!Array.isArray(item.provenance) || item.provenance.length === 0) {
    throw new Error("ToolResult provenance must be nonempty");
  }
  const uncertainty = item.uncertainty === null
    ? null
    : jsonRecord(item.uncertainty, "ToolResult uncertainty");
  return {
    result_id: uuid(item.result_id, "result_id"),
    tool_name: nonblank(item.tool_name, "tool_name"),
    tool_version: nonblank(item.tool_version, "tool_version"),
    model_version: nullableString(item.model_version, "model_version"),
    data_version: nullableString(item.data_version, "data_version"),
    feature_version: nullableString(item.feature_version, "feature_version"),
    input_hash: sha256(item.input_hash, "input_hash"),
    values: jsonRecord(item.values, "ToolResult values"),
    uncertainty,
    warnings: stringArray(item.warnings, "ToolResult warnings"),
    provenance: item.provenance.map(decodeProvenance),
    created_at: timestamp(item.created_at, "ToolResult created_at"),
  };
}

function artifact(result: ToolResult, label: string): Record<string, unknown> {
  nonblank(result.values.artifact_type, `${label} artifact_type`);
  return jsonRecord(result.values.artifact, `${label} artifact`);
}

function identity(
  value: Record<string, unknown>,
  label: string,
): { datasetId: string; cellId: string; cutoffCycle: number } {
  return {
    datasetId: nonblank(value.dataset_id, `${label} dataset_id`),
    cellId: nonblank(value.cell_id, `${label} cell_id`),
    cutoffCycle: integer(value.cutoff_cycle, `${label} cutoff_cycle`),
  };
}

function requireResultReference(
  value: Record<string, unknown>,
  field: string,
  expected: AgentRunResultRecord | undefined,
  label: string,
): void {
  if (!expected || value[field] !== expected.result.result_id) {
    throw new Error(`${label} ${field} does not match the result catalog`);
  }
}

export function decodeAgentRunResults(
  value: unknown,
  run: FixedAdvancedAgentRun,
): AgentRunResultRecord[] {
  if (!isFixedAdvancedRun(run)) throw new Error("result catalog requires a fixed advanced run");
  if (!Array.isArray(value)) throw new Error("Agent run results must be an array");
  const results: AgentRunResultRecord[] = [];
  const byStep = new Map<string, AgentRunResultRecord>();
  const resultIds = new Set<string>();
  let priorOrdinal = 0;
  for (const rawItem of value) {
    const item = record(rawItem, ["step_id", "ordinal", "result"], "Agent run result");
    const stepId = nonblank(item.step_id, "result step_id");
    const expectedIndex = FIXED_ADVANCED_STEP_IDS.indexOf(stepId as FixedAdvancedStepId);
    if (expectedIndex < 0) throw new Error("result step_id is not part of Runtime V7");
    const ordinal = integer(item.ordinal, "result ordinal");
    if (ordinal !== expectedIndex + 1 || ordinal <= priorOrdinal) {
      throw new Error("result ordinal differs from the fixed Agent plan");
    }
    priorOrdinal = ordinal;
    if (byStep.has(stepId)) throw new Error("result step_id values must be unique");
    const result = decodeToolResult(item.result);
    if (resultIds.has(result.result_id)) throw new Error("ToolResult IDs must be unique");
    resultIds.add(result.result_id);
    if (result.tool_name !== FIXED_STEP_TEMPLATES[expectedIndex].tool_name) {
      throw new Error("ToolResult tool_name differs from its Agent step");
    }
    const decoded = { step_id: stepId, ordinal, result };
    byStep.set(stepId, decoded);
    results.push(decoded);
  }

  const expectedBatchId = run.intent.dataset_ids[0];
  let expectedIdentity: { datasetId: string; cellId: string; cutoffCycle: number } | null = null;
  for (const stepId of [
    "advanced-input", "rul-point", "rul-coverage", "soh", "rul-interval", "soh-band", "report",
  ] as const) {
    const entry = byStep.get(stepId);
    if (!entry) continue;
    const values = artifact(entry.result, stepId);
    const currentIdentity = identity(values, stepId);
    if (expectedIdentity && !sameJson(currentIdentity, expectedIdentity)) {
      throw new Error("advanced ToolResults do not share one analysis identity");
    }
    expectedIdentity = currentIdentity;
    if (["advanced-input", "rul-point", "rul-coverage", "soh"].includes(stepId)) {
      if (values.record_batch_id !== expectedBatchId) {
        throw new Error("advanced ToolResult belongs to another record batch");
      }
    }
  }

  const rulInterval = byStep.get("rul-interval");
  if (rulInterval) {
    const values = artifact(rulInterval.result, "rul-interval");
    requireResultReference(values, "prediction_result_id", byStep.get("rul-coverage"), "rul-interval");
    requireResultReference(values, "calibration_result_id", byStep.get("rul-calibration"), "rul-interval");
  }
  const sohBand = byStep.get("soh-band");
  if (sohBand) {
    const values = artifact(sohBand.result, "soh-band");
    requireResultReference(values, "prediction_result_id", byStep.get("soh"), "soh-band");
    requireResultReference(values, "calibration_result_id", byStep.get("soh-calibration"), "soh-band");
  }
  const report = byStep.get("report");
  if (report) {
    const upstream = artifact(report.result, "report").upstream_result_ids;
    const expectedUpstream = ["rul-point", "soh", "rul-interval", "soh-band"].map((stepId) =>
      byStep.get(stepId)?.result.result_id
    );
    if (!Array.isArray(upstream) || expectedUpstream.some((item) => item === undefined) || !sameJson(upstream, expectedUpstream)) {
      throw new Error("report upstream_result_ids do not match the primary results");
    }
  }
  if (run.status === "COMPLETED" && results.length !== FIXED_ADVANCED_STEP_IDS.length) {
    throw new Error("completed advanced runs require all nine ToolResults");
  }
  if (run.status === "COMPLETED" && Object.values(resultIdsFromRunResults(results)).some((id) => id === null)) {
    throw new Error("completed advanced runs lack primary ToolResults");
  }
  return results;
}

export function resultIdsFromRunResults(results: readonly AgentRunResultRecord[]): AdvancedAnalysisResultIds {
  const byStep = new Map(results.map((item) => [item.step_id, item.result.result_id]));
  return {
    rulResultId: byStep.get("rul-point") ?? null,
    sohResultId: byStep.get("soh") ?? null,
    rulConformalResultId: byStep.get("rul-interval") ?? null,
    sohConformalResultId: byStep.get("soh-band") ?? null,
    reportResultId: byStep.get("report") ?? null,
  };
}
