import { act, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ProjectPage, { ProjectWorkspace } from "@/app/projects/[projectId]/page";
import { getAnalysisInputs, getProject } from "@/lib/api-client";

vi.mock("react", async () => {
  const actual = await vi.importActual<typeof import("react")>("react");
  return { ...actual, use: () => ({ projectId: "project-1" }) };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

vi.mock("@/lib/advanced-analysis-contract", async () => {
  const actual = await vi.importActual<typeof import("@/lib/advanced-analysis-contract")>(
    "@/lib/advanced-analysis-contract",
  );
  return { ...actual, isFixedAdvancedRun: () => true };
});

vi.mock("@/lib/api-client", () => ({
  createAdvancedAnalysis: vi.fn(),
  getAnalysisInputs: vi.fn().mockResolvedValue({
    project_id: "project-1",
    datasets: [{
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
    }],
    batches: [{
      record_batch_id: "batch-20",
      binding_schema_version: "record-batch-binding-v1",
      project_id: "project-1",
      dataset_id: "dataset-1",
      source_manifest_sha256: "b".repeat(64),
      content_dataset_id: "content-20",
      dataset_schema_version: "canonical-v1",
      cell_id: "b3c34",
      cutoff_cycle: 20,
      data_version: "matr-v1",
      split_version: "cell-split-v1",
      feature_version: "features-v1",
      created_at: "2026-08-02T01:02:00Z",
    }],
  }),
  getProject: vi.fn().mockResolvedValue({
    project_id: "project-1",
    name: "内置样例验证",
    owner_user_id: "user-1",
    created_at: "2026-08-02T01:00:00Z",
  }),
  listProjectAgentRuns: vi.fn().mockResolvedValue([{
    run_id: "run-1",
    project_id: "project-1",
    status: "COMPLETED",
    created_at: "2026-08-02T02:00:00Z",
    updated_at: "2026-08-02T02:01:00Z",
  }]),
  logout: vi.fn().mockResolvedValue(undefined),
}));

describe("Project workspace", () => {
  it("organizes trusted analysis and run status into the approved dual-pane layout", async () => {
    render(<ProjectPage params={Promise.resolve({ projectId: "project-1" })} />);

    expect(await screen.findByRole("heading", { name: "内置样例验证" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "完整流程" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "分析与结果" })).toHaveTextContent("等待 ToolResult");
    expect(screen.getByRole("region", { name: "运行与进度" })).toHaveTextContent("九步实时进度");
    expect(screen.getByRole("region", { name: "运行与进度" })).toHaveTextContent("冻结分析输入");
    expect(screen.getByRole("region", { name: "运行与进度" })).toHaveTextContent("RUL Conformal");
    expect(screen.getByRole("button", { name: /内置样例/ })).toBeEnabled();
    expect(screen.getAllByText("MATR_b3c34")).not.toHaveLength(0);
    expect(screen.getByText("b3c34 / 20 cycles")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /run-1/ })).toHaveAttribute("href", "/agent/runs/run-1");
    expect(screen.getByText("上传能力尚未接入")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /上传数据/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "启动全新九步 Agent" })).toBeInTheDocument();
    expect(screen.queryByText("运行 ID")).not.toBeInTheDocument();
    expect(screen.queryByText("结果 ID")).not.toBeInTheDocument();
  });

  it("disables the built-in sample when the project has no frozen analysis input", async () => {
    vi.mocked(getAnalysisInputs).mockResolvedValueOnce({
      project_id: "project-1",
      datasets: [],
      batches: [],
    });

    render(<ProjectWorkspace projectId="project-1" />);

    expect(await screen.findByRole("button", { name: /内置样例/ })).toBeDisabled();
    expect(screen.getByText("暂无可用冻结数据")).toBeInTheDocument();
  });

  it("clears the previous project while a route change is loading", async () => {
    let resolveSecondProject!: (value: {
      project_id: string;
      name: string;
      owner_user_id: string;
      created_at: string;
    }) => void;
    vi.mocked(getProject)
      .mockResolvedValueOnce({
        project_id: "project-1",
        name: "项目一",
        owner_user_id: "user-1",
        created_at: "2026-08-02T01:00:00Z",
      })
      .mockReturnValueOnce(new Promise((resolve) => { resolveSecondProject = resolve; }));

    const { rerender } = render(<ProjectWorkspace projectId="project-1" />);
    expect(await screen.findByRole("heading", { name: "项目一" })).toBeInTheDocument();

    rerender(<ProjectWorkspace projectId="project-2" />);

    expect(screen.queryByRole("heading", { name: "项目一" })).not.toBeInTheDocument();
    expect(screen.getByText("正在读取项目…")).toBeInTheDocument();

    await act(async () => {
      resolveSecondProject({
        project_id: "project-2",
        name: "项目二",
        owner_user_id: "user-1",
        created_at: "2026-08-02T01:00:00Z",
      });
    });
    await waitFor(() => expect(screen.getByRole("heading", { name: "项目二" })).toBeInTheDocument());
  });
});
