import type { ToolResult } from "@/lib/types";

const ARTIFACT_TYPES = {
  report: "quanxin_life.advanced_cell_report.v1",
  rul: "quanxin_life.advanced_rul_prediction.v1",
  rulConformal: "quanxin_life.advanced_rul_split_interval.v1",
  soh: "quanxin_life.advanced_soh_trajectory.v1",
  sohConformal: "quanxin_life.advanced_soh_split_band.v1",
} as const;

export type SingleCellResultIds = {
  reportResultId: string | null;
  rulConformalResultId: string | null;
  rulResultId: string | null;
  sohConformalResultId: string | null;
  sohResultId: string | null;
};

export type ResultKey = keyof SingleCellResultIds;
export type ProjectResultLoader = (
  projectId: string,
  resultId: string,
) => Promise<ToolResult>;
export type LoadedResults = Partial<Record<ResultKey, ToolResult>>;

type RuntimeIdentity = {
  artifactId: string;
  artifactKind: string;
  artifactManifestSha256: string;
  cellId: string;
  cutoffCycle: number;
  dataVersion: string;
  datasetId: string;
  featureVersion: string;
  modelVersion: string;
  normalizationStatisticsSha256: string;
  outputTarget: string;
  routeRole: string;
  splitVersion: string;
};

export type ParsedRul = RuntimeIdentity & {
  derivedRemainingCycles: number;
  predictedCycle: number;
  recordBatchId: string;
};

export type ParsedSoh = RuntimeIdentity & {
  predictedSoh: number[];
  predictionCycles: number[];
  recordBatchId: string;
};

export type ParsedRulConformal = RuntimeIdentity & {
  coverageTarget: number;
  derivedRulCycle: number;
  lowerCycle: number;
  lowerRulCycle: number;
  pointPredictionCycle: number;
  predictionResultId: string;
  upperCycle: number;
  upperRulCycle: number;
};

export type ParsedSohConformal = RuntimeIdentity & {
  coverageScope: "simultaneous_finite_trajectory";
  coverageTarget: number;
  finiteHorizonOnly: true;
  lowerSoh: number[];
  predictedSoh: number[];
  predictionCycles: number[];
  predictionResultId: string;
  upperSoh: number[];
};

export type ParsedReport = {
  cellId: string;
  cutoffCycle: number;
  datasetId: string;
  markdown: string;
  upstreamResultIds: string[];
};

export type ParsedAnalysis = {
  report?: ParsedReport;
  rul?: ParsedRul;
  rulConformal?: ParsedRulConformal;
  soh?: ParsedSoh;
  sohConformal?: ParsedSohConformal;
};

export const RESULT_LABELS: Record<ResultKey, string> = {
  reportResultId: "审计报告",
  rulConformalResultId: "RUL Split Conformal",
  rulResultId: "MATR 官方 cycle life",
  sohConformalResultId: "SOH simultaneous band",
  sohResultId: "SOH finite-horizon trajectory",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredRecord(value: unknown, field: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${field} 不是对象`);
  return value;
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${field} 不是有效字符串`);
  }
  return value;
}

function requiredNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${field} 不是有限数值`);
  }
  return value;
}

function requiredNumberArray(value: unknown, field: string): number[] {
  if (
    !Array.isArray(value)
    || value.length === 0
    || !value.every((item) => typeof item === "number" && Number.isFinite(item))
  ) {
    throw new Error(`${field} 不是有效数值数组`);
  }
  return value;
}

function requiredStringArray(value: unknown, field: string): string[] {
  if (
    !Array.isArray(value)
    || !value.every((item) => typeof item === "string" && item.length > 0)
  ) {
    throw new Error(`${field} 不是有效字符串数组`);
  }
  return value;
}

function validateToolResult(result: ToolResult): void {
  if (
    typeof result.result_id !== "string"
    || typeof result.tool_name !== "string"
    || typeof result.tool_version !== "string"
    || typeof result.input_hash !== "string"
    || !isRecord(result.values)
    || !Array.isArray(result.provenance)
    || !result.provenance.every(
      (item) => (
        isRecord(item)
        && typeof item.source_id === "string"
        && typeof item.source_kind === "string"
        && typeof item.uri === "string"
        && typeof item.sha256 === "string"
        && typeof item.description === "string"
        && typeof item.created_at === "string"
      ),
    )
    || !Array.isArray(result.warnings)
    || !result.warnings.every((warning) => typeof warning === "string")
  ) {
    throw new Error("ToolResult envelope 不完整");
  }
}

export function isSafeToolResult(result: ToolResult): boolean {
  try {
    validateToolResult(result);
    return true;
  } catch {
    return false;
  }
}

function artifact(
  result: ToolResult,
  expectedType: string,
  expectedToolName: string,
): Record<string, unknown> {
  validateToolResult(result);
  if (
    result.tool_name !== expectedToolName
    || result.values.artifact_type !== expectedType
  ) {
    throw new Error("ToolResult 类型与结果槽位不匹配");
  }
  return requiredRecord(result.values.artifact, "artifact");
}

function runtimeIdentity(
  value: Record<string, unknown>,
  result: ToolResult,
): RuntimeIdentity {
  return {
    artifactId: requiredString(value.artifact_id, "artifact_id"),
    artifactKind: requiredString(value.artifact_kind, "artifact_kind"),
    artifactManifestSha256: requiredString(
      value.artifact_manifest_sha256,
      "artifact_manifest_sha256",
    ),
    cellId: requiredString(value.cell_id, "cell_id"),
    cutoffCycle: requiredNumber(value.cutoff_cycle, "cutoff_cycle"),
    dataVersion: requiredString(result.data_version, "data_version"),
    datasetId: requiredString(value.dataset_id, "dataset_id"),
    featureVersion: requiredString(result.feature_version, "feature_version"),
    modelVersion: requiredString(result.model_version, "model_version"),
    normalizationStatisticsSha256: requiredString(
      value.normalization_statistics_sha256,
      "normalization_statistics_sha256",
    ),
    outputTarget: requiredString(value.output_target, "output_target"),
    routeRole: requiredString(value.route_role, "route_role"),
    splitVersion: requiredString(value.split_version, "split_version"),
  };
}

function parseRul(result: ToolResult): ParsedRul {
  const value = artifact(result, ARTIFACT_TYPES.rul, "predict_cycle_life");
  const prediction = requiredRecord(
    value.cycle_life_prediction,
    "cycle_life_prediction",
  );
  if (
    prediction.target !== "matr_official_cycle_life"
    || prediction.right_censored !== true
    || prediction.observed_cycle !== null
  ) {
    throw new Error("RUL target 不是 label-free MATR 官方 cycle life");
  }
  const identity = runtimeIdentity(value, result);
  if (
    identity.outputTarget !== "matr_official_cycle_life"
    || prediction.cell_id !== identity.cellId
    || prediction.cutoff_cycle !== identity.cutoffCycle
    || prediction.dataset_id !== identity.datasetId
    || prediction.split_version !== identity.splitVersion
  ) {
    throw new Error("RUL identity 不一致");
  }
  return {
    ...identity,
    derivedRemainingCycles: requiredNumber(
      value.derived_remaining_cycles,
      "derived_remaining_cycles",
    ),
    predictedCycle: requiredNumber(prediction.predicted_cycle, "predicted_cycle"),
    recordBatchId: requiredString(value.record_batch_id, "record_batch_id"),
  };
}

function parseSoh(result: ToolResult): ParsedSoh {
  const value = artifact(result, ARTIFACT_TYPES.soh, "predict_soh_trajectory");
  const identity = runtimeIdentity(value, result);
  const predictionCycles = requiredNumberArray(
    value.prediction_cycles,
    "prediction_cycles",
  );
  const predictedSoh = requiredNumberArray(value.predicted_soh, "predicted_soh");
  if (
    identity.outputTarget !== "soh_trajectory"
    || predictionCycles.length !== predictedSoh.length
    || !predictionCycles.every(
      (cycle, index) => (
        Number.isInteger(cycle)
        && (index === 0 || cycle > predictionCycles[index - 1])
      ),
    )
    || value.horizon_end_cycle !== predictionCycles.at(-1)
  ) {
    throw new Error("SOH finite horizon 证据不一致");
  }
  return {
    ...identity,
    predictedSoh,
    predictionCycles,
    recordBatchId: requiredString(value.record_batch_id, "record_batch_id"),
  };
}

function parseRulConformal(result: ToolResult): ParsedRulConformal {
  const value = artifact(
    result,
    ARTIFACT_TYPES.rulConformal,
    "calibrate_prediction_interval",
  );
  const parsed = {
    ...runtimeIdentity(value, result),
    coverageTarget: requiredNumber(value.coverage_target, "coverage_target"),
    derivedRulCycle: requiredNumber(value.derived_rul_cycle, "derived_rul_cycle"),
    lowerCycle: requiredNumber(value.lower_cycle, "lower_cycle"),
    lowerRulCycle: requiredNumber(value.lower_rul_cycle, "lower_rul_cycle"),
    pointPredictionCycle: requiredNumber(
      value.point_prediction_cycle,
      "point_prediction_cycle",
    ),
    predictionResultId: requiredString(
      value.prediction_result_id,
      "prediction_result_id",
    ),
    upperCycle: requiredNumber(value.upper_cycle, "upper_cycle"),
    upperRulCycle: requiredNumber(value.upper_rul_cycle, "upper_rul_cycle"),
  };
  if (
    parsed.outputTarget !== "matr_official_cycle_life"
    || parsed.coverageTarget <= 0
    || parsed.coverageTarget >= 1
    || !(
      parsed.lowerCycle <= parsed.pointPredictionCycle
      && parsed.pointPredictionCycle <= parsed.upperCycle
    )
    || !(
      parsed.lowerRulCycle <= parsed.derivedRulCycle
      && parsed.derivedRulCycle <= parsed.upperRulCycle
    )
  ) {
    throw new Error("RUL Split Conformal interval 不一致");
  }
  return parsed;
}

function parseSohConformal(result: ToolResult): ParsedSohConformal {
  const value = artifact(
    result,
    ARTIFACT_TYPES.sohConformal,
    "calibrate_prediction_interval",
  );
  const predictionCycles = requiredNumberArray(
    value.prediction_cycles,
    "prediction_cycles",
  );
  const predictedSoh = requiredNumberArray(value.predicted_soh, "predicted_soh");
  const lowerSoh = requiredNumberArray(value.lower_soh, "lower_soh");
  const upperSoh = requiredNumberArray(value.upper_soh, "upper_soh");
  const lengths = new Set([
    predictionCycles.length,
    predictedSoh.length,
    lowerSoh.length,
    upperSoh.length,
  ]);
  if (
    value.finite_horizon_only !== true
    || value.coverage_scope !== "simultaneous_finite_trajectory"
    || lengths.size !== 1
    || !predictionCycles.every(
      (cycle, index) => (
        Number.isInteger(cycle)
        && (index === 0 || cycle > predictionCycles[index - 1])
      ),
    )
    || !predictedSoh.every(
      (point, index) => lowerSoh[index] <= point && point <= upperSoh[index],
    )
  ) {
    throw new Error("SOH simultaneous finite band 不一致");
  }
  const parsed: ParsedSohConformal = {
    ...runtimeIdentity(value, result),
    coverageScope: "simultaneous_finite_trajectory",
    coverageTarget: requiredNumber(value.coverage_target, "coverage_target"),
    finiteHorizonOnly: true,
    lowerSoh,
    predictedSoh,
    predictionCycles,
    predictionResultId: requiredString(
      value.prediction_result_id,
      "prediction_result_id",
    ),
    upperSoh,
  };
  if (
    parsed.outputTarget !== "soh_trajectory"
    || parsed.coverageTarget <= 0
    || parsed.coverageTarget >= 1
  ) {
    throw new Error("SOH Split Conformal target 不一致");
  }
  return parsed;
}

function parseReport(result: ToolResult): ParsedReport {
  const value = artifact(
    result,
    ARTIFACT_TYPES.report,
    "generate_audited_report",
  );
  return {
    cellId: requiredString(value.cell_id, "cell_id"),
    cutoffCycle: requiredNumber(value.cutoff_cycle, "cutoff_cycle"),
    datasetId: requiredString(value.dataset_id, "dataset_id"),
    markdown: requiredString(result.values.markdown, "markdown"),
    upstreamResultIds: requiredStringArray(
      value.upstream_result_ids,
      "upstream_result_ids",
    ),
  };
}

function sameIdentity(left: RuntimeIdentity, right: RuntimeIdentity): boolean {
  return (
    left.cellId === right.cellId
    && left.cutoffCycle === right.cutoffCycle
    && left.dataVersion === right.dataVersion
    && left.datasetId === right.datasetId
    && left.featureVersion === right.featureVersion
    && left.splitVersion === right.splitVersion
  );
}

function sameRuntimeArtifact(
  left: RuntimeIdentity,
  right: RuntimeIdentity,
): boolean {
  return (
    left.modelVersion === right.modelVersion
    && left.artifactKind === right.artifactKind
    && left.artifactId === right.artifactId
    && left.artifactManifestSha256 === right.artifactManifestSha256
    && left.normalizationStatisticsSha256
      === right.normalizationStatisticsSha256
  );
}

function sameNumberArray(left: number[], right: number[]): boolean {
  return (
    left.length === right.length
    && left.every((value, index) => value === right[index])
  );
}

export function parseAnalysis(
  results: LoadedResults,
  resultIds: SingleCellResultIds,
  recordBatchId: string,
): ParsedAnalysis {
  const parsed: ParsedAnalysis = {};
  if (results.rulResultId) parsed.rul = parseRul(results.rulResultId);
  if (results.sohResultId) parsed.soh = parseSoh(results.sohResultId);
  if (results.rulConformalResultId) {
    parsed.rulConformal = parseRulConformal(results.rulConformalResultId);
  }
  if (results.sohConformalResultId) {
    parsed.sohConformal = parseSohConformal(results.sohConformalResultId);
  }
  if (results.reportResultId) parsed.report = parseReport(results.reportResultId);

  if (
    (parsed.rul && parsed.rul.recordBatchId !== recordBatchId)
    || (parsed.soh && parsed.soh.recordBatchId !== recordBatchId)
    || (parsed.rul && parsed.soh && !sameIdentity(parsed.rul, parsed.soh))
  ) {
    throw new Error("record batch、cell 或 dataset 身份不一致");
  }
  if (parsed.rul && parsed.rulConformal) {
    if (!sameIdentity(parsed.rul, parsed.rulConformal)) {
      throw new Error("RUL point 与 route-specific interval 身份不一致");
    }
    if (parsed.rul.cutoffCycle === 20) {
      if (
        parsed.rul.routeRole !== "DEFAULT"
        || parsed.rulConformal.routeRole !== "DEFAULT"
        || !sameRuntimeArtifact(parsed.rul, parsed.rulConformal)
        || parsed.rulConformal.pointPredictionCycle !== parsed.rul.predictedCycle
        || parsed.rulConformal.derivedRulCycle !== parsed.rul.derivedRemainingCycles
      ) {
        throw new Error("cutoff 20 RUL DEFAULT point 与 interval 不一致");
      }
    } else if (
      parsed.rul.routeRole !== "POINT_ACCURACY"
      || parsed.rulConformal.routeRole !== "COVERAGE"
      || parsed.rulConformal.predictionResultId === resultIds.rulResultId
    ) {
      throw new Error("RUL point-accuracy 与 coverage route 绑定不一致");
    }
  }
  if (
    parsed.soh
    && parsed.sohConformal
    && (
      !sameIdentity(parsed.soh, parsed.sohConformal)
      || parsed.soh.routeRole !== parsed.sohConformal.routeRole
      || parsed.sohConformal.predictionResultId !== resultIds.sohResultId
      || !sameNumberArray(
        parsed.soh.predictionCycles,
        parsed.sohConformal.predictionCycles,
      )
      || !sameNumberArray(
        parsed.soh.predictedSoh,
        parsed.sohConformal.predictedSoh,
      )
    )
  ) {
    throw new Error("SOH point 与 route-specific simultaneous band 不一致");
  }
  if (parsed.report) {
    const identities: RuntimeIdentity[] = [];
    if (parsed.rul) identities.push(parsed.rul);
    if (parsed.soh) identities.push(parsed.soh);
    if (parsed.rulConformal) identities.push(parsed.rulConformal);
    if (parsed.sohConformal) identities.push(parsed.sohConformal);
    if (
      identities.some(
        (identity) => (
          identity.cellId !== parsed.report?.cellId
          || identity.cutoffCycle !== parsed.report?.cutoffCycle
          || identity.datasetId !== parsed.report?.datasetId
        ),
      )
    ) {
      throw new Error("报告与分析结果身份不一致");
    }
    const expectedUpstream = [
      resultIds.rulResultId,
      resultIds.sohResultId,
      resultIds.rulConformalResultId,
      resultIds.sohConformalResultId,
    ].filter((value): value is string => value !== null);
    if (
      expectedUpstream.length === 4
      && (
        parsed.report.upstreamResultIds.length !== expectedUpstream.length
        || expectedUpstream.some(
          (id) => !parsed.report?.upstreamResultIds.includes(id),
        )
      )
    ) {
      throw new Error("报告未绑定当前四项 ToolResult");
    }
  }
  return parsed;
}
