import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AppShell } from "@/components/app-shell";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), refresh: vi.fn() }),
}));

describe("AppShell", () => {
  it("renders the approved top navigation and trusted boundary", () => {
    render(<AppShell><p>工作区</p></AppShell>);

    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "主导航" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "分析工作台" })).toHaveAttribute("href", "/projects");
    expect(screen.getByText("可信边界已开启")).toBeInTheDocument();
    expect(screen.getByRole("main")).toHaveTextContent("工作区");
  });

  it("shows the six-stage workflow with an explicit current stage", () => {
    render(
      <AppShell workflowStage="source">
        <p>项目工作区</p>
      </AppShell>,
    );

    const workflow = screen.getByRole("navigation", { name: "完整流程" });
    expect(workflow).toHaveTextContent("登录 / 项目");
    expect(workflow).toHaveTextContent("样例 / 上传");
    expect(workflow).toHaveTextContent("电芯 / cutoff");
    expect(workflow).toHaveTextContent("九步 Agent");
    expect(workflow).toHaveTextContent("结果 / 报告");
    expect(workflow).toHaveTextContent("下载");
    expect(screen.getByText("样例 / 上传").closest("li")).toHaveAttribute("aria-current", "step");
  });

  it("exposes a compact navigation toggle for small screens", async () => {
    const user = userEvent.setup();
    render(<AppShell><p>工作区</p></AppShell>);

    const toggle = screen.getByRole("button", { name: "展开主导航" });
    const navigation = screen.getByRole("navigation", { name: "主导航" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(navigation).not.toHaveClass("main-nav-open");

    await user.click(toggle);

    expect(screen.getByRole("button", { name: "收起主导航" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(navigation).toHaveClass("main-nav-open");
  });

  it("logs out before returning to the login page", async () => {
    const onLogout = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<AppShell onLogout={onLogout}><p>工作区</p></AppShell>);
    await user.click(screen.getByRole("button", { name: "退出登录" }));
    expect(onLogout).toHaveBeenCalledOnce();
  });
});
