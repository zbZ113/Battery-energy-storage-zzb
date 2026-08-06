import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  SingleCellAnalysis,
  type SingleCellResultIds,
} from "@/components/single-cell-analysis";
import type { ToolResult } from "@/lib/types";
import type { ProjectReportExportCatalog } from "@/lib/api-client";

const resultIds: SingleCellResultIds = {
  reportResultId: "report-result",
  rulConformalResultId: "rul-conformal-result",
  rulResultId: "rul-result",
  sohConformalResultId: "soh-conformal-result",
  sohResultId: "soh-result",
};

function result(
  resultId: string,
  toolName: string,
  artifactType: string,
  artifact: Record<string, unknown>,
  warnings: string[] = [],
  values: Record<string, unknown> = {},
): ToolResult {
  return {
    created_at: "2026-07-26T12:00:00Z",
    data_version: "matr-three-batch-v1",
    feature_version: "cyclepatch-multichannel-v1",
    input_hash: `${resultId}-input-sha256`,
    model_version: "advanced-model-v1",
    provenance: [
      {
        created_at: "2026-07-26T12:00:00Z",
        description: `${resultId} verified source`,
        sha256: "a".repeat(64),
        source_id: `${resultId}-source`,
        source_kind: "PREDICTED",
        uri: `artifact://advanced/${resultId}`,
      },
    ],
    result_id: resultId,
    tool_name: toolName,
    tool_version: `${toolName}-v1`,
    uncertainty: null,
    values: {
      artifact,
      artifact_type: artifactType,
      ...values,
    },
    warnings,
  };
}

function validResults(): Record<string, ToolResult> {
  const runtime = {
    artifact_id: "artifact-1",
    artifact_kind: "cyclepatch_direct",
    artifact_manifest_sha256: "b".repeat(64),
    cell_id: "cell-target",
    cutoff_cycle: 20,
    dataset_id: "MATR",
    decision_event_id: "decision-1",
    ledger_head_sha256: "c".repeat(64),
    ledger_sequence_number: 7,
    normalization_statistics_sha256: "d".repeat(64),
    output_target: "matr_official_cycle_life",
    route_role: "DEFAULT",
    split_version: "matr-cell-split-v1",
    task: "RUL",
  };
  const sohRuntime = {
    ...runtime,
    artifact_id: "artifact-2",
    artifact_kind: "current_hybrid",
    output_target: "soh_trajectory",
    route_role: "TAIL_EFFICIENCY",
    task: "SOH",
  };
  const rul = result(
    "rul-result",
    "predict_cycle_life",
    "quanxin_life.advanced_rul_prediction.v1",
    {
      ...runtime,
      cycle_life_prediction: {
        cell_id: "cell-target",
        cutoff_cycle: 20,
        data_version: "matr-three-batch-v1",
        dataset_id: "MATR",
        feature_version: "cyclepatch-multichannel-v1",
        model_version: "advanced-model-v1",
        observed_cycle: null,
        predicted_cycle: 812.5,
        right_censored: true,
        split_version: "matr-cell-split-v1",
        target: "matr_official_cycle_life",
      },
      derived_remaining_cycles: 792.5,
      record_batch_id: "batch-1",
    },
    ["RUL_RESULT_WARNING"],
  );
  const soh = result(
    "soh-result",
    "predict_soh_trajectory",
    "quanxin_life.advanced_soh_trajectory.v1",
    {
      ...sohRuntime,
      horizon_end_cycle: 23,
      predicted_soh: [0.99, 0.98, 0.97],
      prediction_cycles: [21, 22, 23],
      record_batch_id: "batch-1",
    },
    ["SOH_RESULT_WARNING"],
  );
  const rulConformal = result(
    "rul-conformal-result",
    "calibrate_prediction_interval",
    "quanxin_life.advanced_rul_split_interval.v1",
    {
      ...runtime,
      calibration_result_id: "rul-calibration",
      coverage_target: 0.9,
      derived_rul_cycle: 792.5,
      lower_cycle: 780,
      lower_rul_cycle: 760,
      point_prediction_cycle: 812.5,
      prediction_result_id: "rul-result",
      upper_cycle: 845,
      upper_rul_cycle: 825,
    },
    ["RUL_CONFORMAL_WARNING"],
  );
  const sohConformal = result(
    "soh-conformal-result",
    "calibrate_prediction_interval",
    "quanxin_life.advanced_soh_split_band.v1",
    {
      ...sohRuntime,
      calibration_result_id: "soh-calibration",
      coverage_scope: "simultaneous_finite_trajectory",
      coverage_target: 0.9,
      finite_horizon_only: true,
      lower_soh: [0.94, 0.93, 0.92],
      predicted_soh: [0.99, 0.98, 0.97],
      prediction_cycles: [21, 22, 23],
      prediction_result_id: "soh-result",
      upper_soh: [1.04, 1.03, 1.02],
    },
    ["SOH_CONFORMAL_WARNING"],
  );
  const report = result(
    "report-result",
    "generate_audited_report",
    "quanxin_life.advanced_cell_report.v1",
    {
      cell_id: "cell-target",
      cutoff_cycle: 20,
      dataset_id: "MATR",
      split_version: "matr-cell-split-v1",
      upstream_result_ids: [
        "rul-result",
        "soh-result",
        "rul-conformal-result",
        "soh-conformal-result",
      ],
    },
    ["REPORT_WARNING"],
    {
      markdown: [
        "# Advanced single-cell audited report",
        "MATR official cycle life; not a unified EOL80 definition.",
        "SOH result: finite horizon trajectory only.",
      ].join("\n"),
    },
  );
  return {
    "report-result": report,
    "rul-conformal-result": rulConformal,
    "rul-result": rul,
    "soh-conformal-result": sohConformal,
    "soh-result": soh,
  };
}

function cutoff50Results(): Record<string, ToolResult> {
  const results = validResults();
  const updateArtifact = (
    resultId: string,
    update: Record<string, unknown>,
  ) => {
    const current = results[resultId];
    const currentArtifact = current.values.artifact as Record<string, unknown>;
    results[resultId] = {
      ...current,
      values: {
        ...current.values,
        artifact: { ...currentArtifact, ...update },
      },
    };
  };
  updateArtifact("rul-result", {
    cutoff_cycle: 50,
    route_role: "POINT_ACCURACY",
    cycle_life_prediction: {
      ...((results["rul-result"].values.artifact as Record<string, unknown>)
        .cycle_life_prediction as Record<string, unknown>),
      cutoff_cycle: 50,
      predicted_cycle: 900,
    },
    derived_remaining_cycles: 850,
  });
  updateArtifact("soh-result", {
    cutoff_cycle: 50,
    horizon_end_cycle: 53,
    predicted_soh: [0.98, 0.97, 0.96],
    prediction_cycles: [51, 52, 53],
  });
  updateArtifact("rul-conformal-result", {
    artifact_id: "coverage-artifact",
    artifact_kind: "cyclepatch_batlinet",
    cutoff_cycle: 50,
    derived_rul_cycle: 790,
    lower_cycle: 810,
    lower_rul_cycle: 760,
    point_prediction_cycle: 840,
    prediction_result_id: "coverage-point-result",
    route_role: "COVERAGE",
    upper_cycle: 870,
    upper_rul_cycle: 820,
  });
  updateArtifact("soh-conformal-result", {
    cutoff_cycle: 50,
    lower_soh: [0.93, 0.92, 0.91],
    predicted_soh: [0.98, 0.97, 0.96],
    prediction_cycles: [51, 52, 53],
    upper_soh: [1.03, 1.02, 1.01],
  });
  updateArtifact("report-result", { cutoff_cycle: 50 });
  return results;
}

describe("SingleCellAnalysis", () => {
  it("shows a waiting state and makes no request when no ToolResult IDs were supplied", () => {
    const loadResult = vi.fn();
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={{
          reportResultId: null,
          rulConformalResultId: null,
          rulResultId: null,
          sohConformalResultId: null,
          sohResultId: null,
        }}
      />,
    );

    expect(loadResult).not.toHaveBeenCalled();
    expect(screen.getByText("等待服务端签发分析结果")).toBeInTheDocument();
    expect(screen.getByText("尚缺 5 项 ToolResult")).toBeInTheDocument();
  });

  it("does not tell users to keep waiting after a terminal run", () => {
    render(
      <SingleCellAnalysis
        loadResult={vi.fn()}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={{
          reportResultId: null,
          rulConformalResultId: null,
          rulResultId: null,
          sohConformalResultId: null,
          sohResultId: null,
        }}
        runTerminal
      />,
    );

    expect(screen.getByText("运行已结束，部分结果未签发")).toBeInTheDocument();
    expect(screen.queryByText("等待服务端签发分析结果")).not.toBeInTheDocument();
  });

  it("keeps export unavailable until the server provides a signed artifact", () => {
    render(
      <SingleCellAnalysis
        loadResult={vi.fn()}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={{
          reportResultId: null,
          rulConformalResultId: null,
          rulResultId: null,
          sohConformalResultId: null,
          sohResultId: null,
        }}
      />,
    );

    const exportStatus = screen.getByRole("status", { name: "下载状态" });
    expect(exportStatus).toHaveTextContent("待服务端提供导出");
    expect(screen.queryByRole("link", { name: /下载/ })).not.toBeInTheDocument();
  });

  it("loads every supplied result through the project-scoped loader and renders issued values", async () => {
    const results = validResults();
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
      />,
    );

    await waitFor(() => expect(loadResult).toHaveBeenCalledTimes(5));
    expect(loadResult.mock.calls).toEqual(
      expect.arrayContaining([
        ["project-1", "rul-result"],
        ["project-1", "soh-result"],
        ["project-1", "rul-conformal-result"],
        ["project-1", "soh-conformal-result"],
        ["project-1", "report-result"],
      ]),
    );

    expect(await screen.findByText("MATR 官方 cycle life（非统一 EOL80）")).toBeInTheDocument();
    expect(screen.getAllByText("812.5").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("792.5").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("[780, 845]")).toBeInTheDocument();
    expect(screen.getByText("[760, 825]")).toBeInTheDocument();
    expect(screen.getAllByText("DEFAULT").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("TAIL_EFFICIENCY").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("finite_horizon_only = true")).toBeInTheDocument();
    expect(screen.getByText("simultaneous_finite_trajectory")).toBeInTheDocument();
    expect(screen.getAllByText("RUL_RESULT_WARNING").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("SOH_RESULT_WARNING").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("RUL_CONFORMAL_WARNING").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("SOH_CONFORMAL_WARNING").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("REPORT_WARNING").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText(/Advanced single-cell audited report/).length,
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("已签发工具结果")).toHaveLength(5);
  });

  it("dispatches once, polls pending exports, and renders only complete READY links", async () => {
    const results = validResults();
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    const pending = reportCatalog("PENDING", [
      reportExport("markdown", "PENDING"),
      reportExport("pdf", "PENDING"),
    ]);
    const ready = reportCatalog("READY", [
      reportExport("markdown", "READY"),
      reportExport("pdf", "READY"),
      reportExport("docx", "FAILED"),
    ]);
    const createExports = vi.fn().mockResolvedValue(pending);
    const loadExports = vi.fn().mockResolvedValue(ready);

    render(
      <SingleCellAnalysis
        createReportExports={createExports}
        exportPollIntervalMs={1}
        loadReportExports={loadExports}
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
        runCompleted
        runId="run-export-1"
        runTerminal
        toolResultIds={["tool-result-1", "tool-result-2"]}
      />,
    );

    await waitFor(() => expect(loadExports).toHaveBeenCalledTimes(1));
    expect(createExports).toHaveBeenCalledTimes(1);
    expect(createExports).toHaveBeenCalledWith(
      "project-1",
      "run-export-1",
      "report-result",
    );
    expect(loadExports).toHaveBeenCalledWith(
      "project-1",
      "run-export-1",
      "report-result",
    );

    const markdown = await screen.findByRole("link", { name: /Markdown/ });
    expect(markdown).toHaveAttribute(
      "href",
      "http://localhost:8000/v1/projects/project-1/agent/runs/run-export-1/reports/report-result/exports/markdown",
    );
    expect(markdown).toHaveAttribute("download", "report-markdown.md");
    expect(screen.getByRole("link", { name: /PDF/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /DOCX/ })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "下载 ToolResult 1 JSON" })).toHaveAttribute(
      "href",
      "http://localhost:8000/v1/projects/project-1/agent/runs/run-export-1/reports/report-result/exports/tool-results/tool-result-1.json",
    );
    expect(screen.getByRole("link", { name: "下载 ToolResult 2 JSON" })).toBeInTheDocument();
    const downloadCenter = screen.getByRole("region", { name: "报告下载中心" });
    expect(downloadCenter).toHaveTextContent("128 B");
    expect(downloadCenter).toHaveTextContent("a".repeat(64));
    expect(downloadCenter).toHaveTextContent("2026-08-04T08:01:00Z");
  });

  it("reports a failed server export job instead of presenting an empty catalog", async () => {
    const results = validResults();
    const failed = reportCatalog("FAILED", [reportExport("markdown", "FAILED")]);
    render(
      <SingleCellAnalysis
        createReportExports={vi.fn().mockResolvedValue(failed)}
        loadReportExports={vi.fn()}
        loadResult={vi.fn(async (_projectId: string, resultId: string) => results[resultId])}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
        runCompleted
        runId="run-export-1"
        runTerminal
      />,
    );

    expect(await screen.findByRole("alert", { name: "导出任务失败" })).toHaveTextContent(
      "服务端未能生成报告",
    );
  });

  it("fails closed when a catalog identity does not match the validated run", async () => {
    const results = validResults();
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    const createExports = vi.fn().mockResolvedValue({
      ...reportCatalog("READY", [reportExport("markdown", "READY")]),
      run_id: "another-run",
    });

    render(
      <SingleCellAnalysis
        createReportExports={createExports}
        loadReportExports={vi.fn()}
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
        runCompleted
        runId="run-export-mismatch"
        runTerminal
      />,
    );

    expect(await screen.findByRole("alert", { name: "下载错误" })).toHaveTextContent(
      "导出目录身份核验失败",
    );
    expect(screen.queryByRole("link", { name: /Markdown/ })).not.toBeInTheDocument();
  });

  it("does not dispatch exports before a completed report passes ToolResult validation", async () => {
    const results = validResults();
    const values = results["report-result"].values;
    results["report-result"] = {
      ...results["report-result"],
      values: {
        ...values,
        artifact: {
          ...(values.artifact as Record<string, unknown>),
          cell_id: "another-cell",
        },
      },
    };
    const createExports = vi.fn();
    render(
      <SingleCellAnalysis
        createReportExports={createExports}
        loadReportExports={vi.fn()}
        loadResult={vi.fn(async (_projectId: string, resultId: string) => results[resultId])}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
        runCompleted
        runId="run-invalid-results"
        runTerminal
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("分析结果暂不可用");
    expect(createExports).not.toHaveBeenCalled();
  });

  it("fails closed when loaded artifacts do not describe the same cell", async () => {
    const results = validResults();
    const values = results["soh-result"].values;
    results["soh-result"] = {
      ...results["soh-result"],
      values: {
        ...values,
        artifact: {
          ...(values.artifact as Record<string, unknown>),
          cell_id: "another-cell",
        },
      },
    };
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("分析结果暂不可用");
    expect(screen.getByRole("alert")).toHaveTextContent("身份或证据链不一致");
    expect(screen.queryByText("812.5")).not.toBeInTheDocument();
  });

  it("does not pass a malformed ToolResult envelope into the evidence view", async () => {
    const results = validResults();
    results["report-result"] = {
      ...results["report-result"],
      provenance: null as unknown as ToolResult["provenance"],
    };
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("分析结果暂不可用");
    expect(screen.queryByText("report-result")).not.toBeInTheDocument();
  });

  it("keeps verified results visible when one project-scoped request fails", async () => {
    const results = validResults();
    const loadResult = vi.fn(async (_projectId: string, resultId: string) => {
      if (resultId === "report-result") throw new Error("not available");
      return results[resultId];
    });
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("审计报告读取失败");
    expect(screen.getByText("MATR 官方 cycle life（非统一 EOL80）")).toBeInTheDocument();
    expect(screen.getAllByText("已签发工具结果")).toHaveLength(4);
  });

  it("keeps cutoff-above-20 point accuracy separate from the coverage-route interval center", async () => {
    const results = cutoff50Results();
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
      />,
    );

    expect(await screen.findByText("MATR 官方 cycle life（非统一 EOL80）")).toBeInTheDocument();
    expect(screen.queryByText(/分析结果暂不可用/)).not.toBeInTheDocument();
    expect(screen.getAllByText("900").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("840").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("POINT_ACCURACY").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("COVERAGE").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("coverage route interval center")).toBeInTheDocument();
  });

  it("accepts distinct cutoff-20 DEFAULT result IDs when runtime and values are identical", async () => {
    const results = validResults();
    const conformal = results["rul-conformal-result"];
    results["rul-conformal-result"] = {
      ...conformal,
      values: {
        ...conformal.values,
        artifact: {
          ...(conformal.values.artifact as Record<string, unknown>),
          prediction_result_id: "rul-default-coverage-copy",
        },
      },
    };
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
      />,
    );

    expect(await screen.findByText("MATR 官方 cycle life（非统一 EOL80）")).toBeInTheDocument();
    expect(screen.queryByText(/分析结果暂不可用/)).not.toBeInTheDocument();
  });

  it("rejects a cutoff-20 DEFAULT interval whose server-issued center was changed", async () => {
    const results = validResults();
    const conformal = results["rul-conformal-result"];
    results["rul-conformal-result"] = {
      ...conformal,
      values: {
        ...conformal.values,
        artifact: {
          ...(conformal.values.artifact as Record<string, unknown>),
          point_prediction_cycle: 813.5,
          prediction_result_id: "rul-default-coverage-copy",
        },
      },
    };
    const loadResult = vi.fn(
      async (_projectId: string, resultId: string) => results[resultId],
    );
    render(
      <SingleCellAnalysis
        loadResult={loadResult}
        projectId="project-1"
        recordBatchId="batch-1"
        resultIds={resultIds}
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("分析结果暂不可用");
    expect(screen.queryByText("MATR 官方 cycle life（非统一 EOL80）")).not.toBeInTheDocument();
  });
});

function reportCatalog(
  status: ProjectReportExportCatalog["status"],
  exports: ProjectReportExportCatalog["exports"],
): ProjectReportExportCatalog {
  return {
    report_id: "report-1",
    project_id: "project-1",
    run_id: "run-export-1",
    report_result_id: "report-result",
    status,
    template_version: "advanced-cell-report-template-v1",
    created_at: "2026-08-03T08:00:00Z",
    completed_at: status === "READY" ? "2026-08-03T08:01:00Z" : null,
    exports,
  };
}

function reportExport(
  format: ProjectReportExportCatalog["exports"][number]["format"],
  status: ProjectReportExportCatalog["exports"][number]["status"],
): ProjectReportExportCatalog["exports"][number] {
  const ready = status === "READY";
  return {
    export_id: `export-${format}`,
    report_id: "report-1",
    format,
    status,
    filename: ready ? `report-${format}.${format === "markdown" ? "md" : format}` : null,
    media_type: ready ? "application/octet-stream" : null,
    size_bytes: ready ? 128 : null,
    sha256: ready ? "a".repeat(64) : null,
    created_at: "2026-08-03T08:00:00Z",
    completed_at: ready ? "2026-08-03T08:01:00Z" : null,
    expires_at: ready ? "2026-08-04T08:01:00Z" : null,
  };
}
