import { describe, expect, it } from "vitest";

import { RUNTIME_V7_STEPS } from "@/components/nine-step-tracker";
import {
  FIXED_ADVANCED_STEP_IDS,
  decodeAgentRun,
  decodeAgentRunResults,
  decodeAnalysisInputs,
  isFixedAdvancedRun,
  resultIdsFromRunResults,
} from "@/lib/advanced-analysis-contract";

const PROJECT_ID = "project-1";
const DATASET_ID = "dataset-1";
const RECORD_BATCH_ID = "batch-1";
const CELL_ID = "cell-1";
const CUTOFF_CYCLE = 20;
const CREATED_AT = "2026-08-02T04:00:00Z";
const SHA = "a".repeat(64);

const STEP_TEMPLATES = [
  {
    step_id: "advanced-input",
    role: "data_quality",
    tool_name: "extract_early_cycle_features",
    depends_on: [],
    input_references: { record_batch_id: "intent.dataset_ids[0]" },
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
  },
] as const;

function analysisInputs() {
  return {
    project_id: PROJECT_ID,
    datasets: [{
      dataset_id: DATASET_ID,
      project_id: PROJECT_ID,
      name: "MATR sample",
      data_version: "data-v1",
      schema_version: "dataset-v1",
      status: "FROZEN",
      manifest_uri: "minio://datasets/manifest.json",
      manifest_sha256: SHA,
      created_at: CREATED_AT,
      frozen_at: CREATED_AT,
    }],
    batches: [{
      record_batch_id: RECORD_BATCH_ID,
      binding_schema_version: "record-batch-binding-v1",
      project_id: PROJECT_ID,
      dataset_id: DATASET_ID,
      source_manifest_sha256: SHA,
      content_dataset_id: "MATR",
      dataset_schema_version: "dataset-v1",
      cell_id: CELL_ID,
      cutoff_cycle: CUTOFF_CYCLE,
      data_version: "data-v1",
      split_version: "split-v1",
      feature_version: "feature-v1",
      created_at: CREATED_AT,
    }],
  };
}

function agentRun(status = "RUNNING") {
  const intentId = "00000000-0000-4000-8000-000000000001";
  return {
    run_id: "run-1",
    project_id: PROJECT_ID,
    created_by_user_id: "user-1",
    status,
    planning_mode: "FIXED_FALLBACK",
    intent: {
      intent_id: intentId,
      project_id: PROJECT_ID,
      goal: "Run the fixed advanced single-cell analysis workflow",
      dataset_ids: [RECORD_BATCH_ID],
      requested_outputs: ["advanced_single_cell_analysis"],
      created_at: CREATED_AT,
    },
    plan: {
      plan_version: "supervisor-planner-v1",
      intent_id: intentId,
      steps: STEP_TEMPLATES.map((step) => ({
        ...step,
        requires_approval: false,
        failure_policy: "STOP",
      })),
      planning_mode: "FIXED_FALLBACK",
      plan_hash: SHA,
      created_at: CREATED_AT,
    },
    dispatch_status: "DISPATCHED",
    dispatch_task_id: "task-1",
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    completed_at: status === "COMPLETED" ? CREATED_AT : null,
  };
}

const RESULT_IDS = STEP_TEMPLATES.map((_, index) =>
  `00000000-0000-4000-8000-${String(index + 10).padStart(12, "0")}`
);

function artifactFor(stepId: string): Record<string, unknown> {
  const common = {
    dataset_id: "MATR",
    cell_id: CELL_ID,
    cutoff_cycle: CUTOFF_CYCLE,
  };
  if (["advanced-input", "rul-point", "rul-coverage", "soh"].includes(stepId)) {
    return { ...common, record_batch_id: RECORD_BATCH_ID };
  }
  if (stepId === "rul-interval") {
    return {
      ...common,
      prediction_result_id: RESULT_IDS[2],
      calibration_result_id: RESULT_IDS[4],
    };
  }
  if (stepId === "soh-band") {
    return {
      ...common,
      prediction_result_id: RESULT_IDS[3],
      calibration_result_id: RESULT_IDS[6],
    };
  }
  if (stepId === "report") {
    return {
      ...common,
      upstream_result_ids: [RESULT_IDS[1], RESULT_IDS[3], RESULT_IDS[5], RESULT_IDS[7]],
    };
  }
  return common;
}

function runResults() {
  return STEP_TEMPLATES.map((step, index) => ({
    step_id: step.step_id,
    ordinal: index + 1,
    result: {
      result_id: RESULT_IDS[index],
      tool_name: step.tool_name,
      tool_version: `${step.step_id}-tool-v1`,
      model_version: null,
      data_version: "data-v1",
      feature_version: "feature-v1",
      input_hash: SHA,
      values: {
        artifact_type: `quanxin_life.${step.step_id}.v1`,
        artifact: artifactFor(step.step_id),
      },
      uncertainty: null,
      warnings: [],
      provenance: [{
        source_id: `source-${index + 1}`,
        source_kind: "PREDICTED",
        uri: `tool-result://${RESULT_IDS[index]}`,
        sha256: SHA,
        description: "Audited source",
        created_at: CREATED_AT,
      }],
      created_at: CREATED_AT,
    },
  }));
}

describe("advanced analysis runtime contract", () => {
  it("keeps the catalog contract aligned with the visible Runtime V7 order", () => {
    expect(FIXED_ADVANCED_STEP_IDS).toEqual(RUNTIME_V7_STEPS.map((step) => step.id));
  });

  it("decodes only frozen project-scoped datasets and their bound batches", () => {
    const decoded = decodeAnalysisInputs(analysisInputs(), PROJECT_ID);
    expect(decoded.batches[0].record_batch_id).toBe(RECORD_BATCH_ID);

    expect(() => decodeAnalysisInputs({ ...analysisInputs(), extra: true }, PROJECT_ID)).toThrow();
    expect(() => decodeAnalysisInputs({
      ...analysisInputs(),
      datasets: [{ ...analysisInputs().datasets[0], status: "DRAFT", frozen_at: null }],
    }, PROJECT_ID)).toThrow();
    expect(() => decodeAnalysisInputs({
      ...analysisInputs(),
      batches: [{ ...analysisInputs().batches[0], dataset_id: "missing-dataset" }],
    }, PROJECT_ID)).toThrow();
    expect(() => decodeAnalysisInputs({
      ...analysisInputs(),
      batches: [{ ...analysisInputs().batches[0], binding_schema_version: "unknown-binding" }],
    }, PROJECT_ID)).toThrow();
    expect(() => decodeAnalysisInputs(analysisInputs(), "another-project")).toThrow();
  });

  it("decodes only the exact fixed advanced run identity and nine-step template", () => {
    const decoded = decodeAgentRun(agentRun(), PROJECT_ID, RECORD_BATCH_ID);
    expect(isFixedAdvancedRun(decoded)).toBe(true);

    expect(() => decodeAgentRun({ ...agentRun(), extra: true }, PROJECT_ID, RECORD_BATCH_ID)).toThrow();
    expect(() => decodeAgentRun(agentRun(), "another-project", RECORD_BATCH_ID)).toThrow();
    expect(() => decodeAgentRun(agentRun(), PROJECT_ID, "another-batch")).toThrow();
    expect(() => decodeAgentRun({
      ...agentRun(),
      intent: { ...agentRun().intent, requested_outputs: ["something_else"] },
    }, PROJECT_ID, RECORD_BATCH_ID)).toThrow();
    expect(() => decodeAgentRun({
      ...agentRun(),
      plan: {
        ...agentRun().plan,
        steps: agentRun().plan.steps.map((step, index) => (
          index === 1 ? { ...step, role: "supervisor" } : step
        )),
      },
    }, PROJECT_ID, RECORD_BATCH_ID)).toThrow();
  });

  it("accepts semantically identical input references regardless of JSON key order", () => {
    const raw = agentRun();
    const reordered = {
      ...raw,
      plan: {
        ...raw.plan,
        steps: raw.plan.steps.map((step, index) => index === 4 ? {
          ...step,
          input_references: Object.fromEntries(
            Object.entries(step.input_references).reverse(),
          ),
        } : step),
      },
    };

    expect(decodeAgentRun(reordered, PROJECT_ID, RECORD_BATCH_ID).plan.steps[4].step_id)
      .toBe("rul-calibration");
  });

  it("decodes the result catalog and maps the five primary result IDs", () => {
    const run = decodeAgentRun(agentRun("COMPLETED"), PROJECT_ID, RECORD_BATCH_ID);
    const decoded = decodeAgentRunResults(runResults(), run);

    expect(resultIdsFromRunResults(decoded)).toEqual({
      rulResultId: RESULT_IDS[1],
      sohResultId: RESULT_IDS[3],
      rulConformalResultId: RESULT_IDS[5],
      sohConformalResultId: RESULT_IDS[7],
      reportResultId: RESULT_IDS[8],
    });
  });

  it("allows a valid non-terminal subset without inventing missing results", () => {
    const run = decodeAgentRun(agentRun("RUNNING"), PROJECT_ID, RECORD_BATCH_ID);
    const decoded = decodeAgentRunResults(runResults().slice(0, 4), run);

    expect(resultIdsFromRunResults(decoded)).toEqual({
      rulResultId: RESULT_IDS[1],
      sohResultId: RESULT_IDS[3],
      rulConformalResultId: null,
      sohConformalResultId: null,
      reportResultId: null,
    });
  });

  it("rejects duplicate steps, ordinal drift, wrong batches, and unknown fields", () => {
    const run = decodeAgentRun(agentRun(), PROJECT_ID, RECORD_BATCH_ID);
    const duplicate = [...runResults().slice(0, 2), runResults()[1]];
    expect(() => decodeAgentRunResults(duplicate, run)).toThrow();

    const drifted = runResults().slice(0, 2);
    drifted[1] = { ...drifted[1], ordinal: 3 };
    expect(() => decodeAgentRunResults(drifted, run)).toThrow();

    const wrongBatch = runResults().slice(0, 2);
    wrongBatch[1] = {
      ...wrongBatch[1],
      result: {
        ...wrongBatch[1].result,
        values: {
          ...wrongBatch[1].result.values,
          artifact: { ...artifactFor("rul-point"), record_batch_id: "another-batch" },
        },
      },
    };
    expect(() => decodeAgentRunResults(wrongBatch, run)).toThrow();

    const unknown = runResults().slice(0, 1);
    unknown[0] = { ...unknown[0], extra: true } as typeof unknown[number];
    expect(() => decodeAgentRunResults(unknown, run)).toThrow();
  });

  it("fails closed when a completed run lacks any of its nine persisted results", () => {
    const run = decodeAgentRun(agentRun("COMPLETED"), PROJECT_ID, RECORD_BATCH_ID);
    expect(() => decodeAgentRunResults(runResults().filter((item) => item.step_id !== "rul-calibration"), run)).toThrow();
  });
});
