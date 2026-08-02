import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AdvancedAnalysisLauncher } from "@/components/advanced-analysis-launcher";
import type { AnalysisInputsRecord, AgentRunRecord } from "@/lib/types";

vi.mock("@/lib/advanced-analysis-contract", () => ({
  decodeAgentRun: (value: unknown) => value,
}));

const inputs: AnalysisInputsRecord = {
  project_id: "project-1",
  datasets: [
    {
      dataset_id: "dataset-1",
      project_id: "project-1",
      name: "MATR_b3c34",
      data_version: "matr-v1",
      schema_version: "canonical-v1",
      status: "FROZEN",
      manifest_uri: null,
      manifest_sha256: "a".repeat(64),
      created_at: "2026-08-02T01:00:00Z",
      frozen_at: "2026-08-02T01:01:00Z",
    },
  ],
  batches: [20, 50].map((cutoff) => ({
    record_batch_id: `batch-${cutoff}`,
    binding_schema_version: "binding-v1",
    project_id: "project-1",
    dataset_id: "dataset-1",
    source_manifest_sha256: "b".repeat(64),
    content_dataset_id: `content-${cutoff}`,
    dataset_schema_version: "canonical-v1",
    cell_id: "b3c34",
    cutoff_cycle: cutoff,
    data_version: "matr-v1",
    split_version: "cell-split-v1",
    feature_version: "features-v1",
    created_at: "2026-08-02T01:02:00Z",
  })),
};

function createdRun(): AgentRunRecord {
  return {
    run_id: "run-1",
    project_id: "project-1",
    status: "PENDING",
    created_at: "2026-08-02T02:00:00Z",
    updated_at: "2026-08-02T02:00:00Z",
  };
}

describe("AdvancedAnalysisLauncher", () => {
  it("launches only a server-listed cell and cutoff without a free UUID field", async () => {
    const user = userEvent.setup();
    const createAnalysis = vi.fn().mockResolvedValue(createdRun());
    const onCreated = vi.fn();
    render(
      <AdvancedAnalysisLauncher
        createAnalysis={createAnalysis}
        inputs={inputs}
        onCreated={onCreated}
        projectId="project-1"
      />,
    );

    expect(screen.queryByLabelText(/UUID|数据集 ID|record batch/i)).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("cutoff"), "batch-50");
    await user.click(screen.getByRole("button", { name: "启动全新九步 Agent" }));

    expect(createAnalysis).toHaveBeenCalledWith(
      "project-1",
      { record_batch_id: "batch-50", cell_id: "b3c34", cutoff_cycle: 50 },
      expect.stringMatching(/^advanced-analysis-/),
    );
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith(createdRun()));
  });

  it("reuses the same idempotency key when a failed launch is retried", async () => {
    const user = userEvent.setup();
    const createAnalysis = vi.fn()
      .mockRejectedValueOnce(new Error("network"))
      .mockResolvedValueOnce(createdRun());
    render(
      <AdvancedAnalysisLauncher
        createAnalysis={createAnalysis}
        inputs={inputs}
        onCreated={vi.fn()}
        projectId="project-1"
      />,
    );

    const launch = screen.getByRole("button", { name: "启动全新九步 Agent" });
    await user.click(launch);
    expect(await screen.findByRole("alert")).toHaveTextContent("启动失败");
    await user.click(launch);

    await waitFor(() => expect(createAnalysis).toHaveBeenCalledTimes(2));
    expect(createAnalysis.mock.calls[1][2]).toBe(createAnalysis.mock.calls[0][2]);
  });

  it("does not allow duplicate submissions while creation is pending", async () => {
    const user = userEvent.setup();
    let resolveRun!: (run: AgentRunRecord) => void;
    const createAnalysis = vi.fn(() => new Promise<AgentRunRecord>((resolve) => {
      resolveRun = resolve;
    }));
    render(
      <AdvancedAnalysisLauncher
        createAnalysis={createAnalysis}
        inputs={inputs}
        onCreated={vi.fn()}
        projectId="project-1"
      />,
    );

    const launch = screen.getByRole("button", { name: "启动全新九步 Agent" });
    await user.click(launch);
    expect(launch).toBeDisabled();
    screen.getAllByRole("combobox").forEach((selector) => expect(selector).toBeDisabled());
    await user.click(launch);
    expect(createAnalysis).toHaveBeenCalledOnce();
    resolveRun(createdRun());
  });
});
