import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RunAnalysisLoader } from "@/components/run-analysis-loader";
import type { AgentRunRecord, AgentRunResultRecord } from "@/lib/types";

vi.mock("@/components/single-cell-analysis", () => ({
  SingleCellAnalysis: ({ projectId, recordBatchId, resultIds, runTerminal, showEvidence }: {
    projectId: string;
    recordBatchId: string;
    resultIds: Record<string, string | null>;
    runTerminal?: boolean;
    showEvidence?: boolean;
  }) => (
    <div data-testid="analysis">
      {projectId}|{recordBatchId}|{JSON.stringify(resultIds)}|terminal={String(runTerminal)}|evidence={String(showEvidence)}
    </div>
  ),
}));

vi.mock("@/lib/advanced-analysis-contract", () => ({
  decodeAgentRun: (value: unknown) => value,
  decodeAgentRunResults: (value: unknown) => value,
  resultIdsFromRunResults: () => ({
    reportResultId: "report-result",
    rulConformalResultId: "rul-conformal-result",
    rulResultId: "rul-result",
    sohConformalResultId: "soh-conformal-result",
    sohResultId: "soh-result",
  }),
}));

const run: AgentRunRecord = {
  run_id: "run-1",
  project_id: "project-1",
  status: "COMPLETED",
  created_at: "2026-08-02T02:00:00Z",
  updated_at: "2026-08-02T02:01:00Z",
};

describe("RunAnalysisLoader", () => {
  it("discovers result IDs from the run catalog and passes them to the verified analysis", async () => {
    const loadRun = vi.fn().mockResolvedValue(run);
    const loadResults = vi.fn().mockResolvedValue([{
      step_id: "advanced-input",
      ordinal: 1,
      result: {
        result_id: "00000000-0000-4000-8000-000000000010",
        tool_name: "extract_early_cycle_features",
        tool_version: "tool-v1",
        input_hash: "a".repeat(64),
        values: { artifact: { record_batch_id: "record-batch-1" } },
        uncertainty: null,
        provenance: [],
        warnings: [],
        created_at: "2026-08-02T04:00:00Z",
      },
    }] satisfies AgentRunResultRecord[]);
    render(
      <RunAnalysisLoader
        loadResults={loadResults}
        loadRun={loadRun}
        projectId="project-1"
        recordBatchId="record-batch-1"
        runId="run-1"
      />,
    );

    await waitFor(() => expect(loadRun).toHaveBeenCalledWith("run-1"));
    expect(loadResults).toHaveBeenCalledWith("run-1");
    expect(await screen.findByTestId("analysis")).toHaveTextContent("project-1|record-batch-1");
    expect(screen.getByTestId("analysis")).toHaveTextContent('"reportResultId":"report-result"');
    expect(screen.getByTestId("analysis")).toHaveTextContent("terminal=true");
    expect(screen.getByTestId("analysis")).toHaveTextContent("evidence=false");
    expect(screen.getByRole("heading", { name: "九步 ToolResult 证据目录" })).toBeInTheDocument();
  });

  it("fails closed when the run or result catalog cannot be verified", async () => {
    render(
      <RunAnalysisLoader
        loadResults={vi.fn().mockResolvedValue([])}
        loadRun={vi.fn().mockRejectedValue(new Error("invalid run"))}
        projectId="project-1"
        recordBatchId="record-batch-1"
        runId="run-1"
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("结果目录无法通过身份核验");
    expect(screen.queryByTestId("analysis")).not.toBeInTheDocument();
  });
});
