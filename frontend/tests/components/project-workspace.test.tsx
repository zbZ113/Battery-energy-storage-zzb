import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ProjectPage from "@/app/projects/[projectId]/page";

vi.mock("react", async () => {
  const actual = await vi.importActual<typeof import("react")>("react");
  return { ...actual, use: () => ({ projectId: "project-1" }) };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

vi.mock("@/lib/api-client", () => ({
  createAgentRun: vi.fn(),
  getProject: vi.fn().mockResolvedValue({
    project_id: "project-1",
    name: "内置样例验证",
    owner_user_id: "user-1",
    created_at: "2026-08-02T01:00:00Z",
  }),
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
    expect(screen.getByText("上传能力尚未接入")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /内置样例/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /上传数据/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "启动全新九步 Agent" })).toBeInTheDocument();
  });
});
