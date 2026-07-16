import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CreateAgentRunForm } from "@/components/create-agent-run-form";

describe("CreateAgentRunForm", () => {
  it("normalizes multiline dataset identifiers and uses the safe cycle-life output", async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<CreateAgentRunForm onCreate={onCreate} projectId="project-1" />);

    await user.type(screen.getByLabelText("你想让 Agent 完成什么"), "分析电芯寿命风险");
    await user.type(screen.getByLabelText("冻结数据集 ID"), "dataset-1, dataset-2\ndataset-1");
    await user.click(screen.getByRole("button", { name: "创建分析任务" }));

    expect(onCreate).toHaveBeenCalledWith({
      project_id: "project-1",
      user_goal: "分析电芯寿命风险",
      dataset_ids: ["dataset-1", "dataset-2"],
      requested_outputs: ["cycle_life"],
    });
  });

  it("requires at least one frozen dataset identifier", async () => {
    const onCreate = vi.fn();
    const user = userEvent.setup();
    render(<CreateAgentRunForm onCreate={onCreate} projectId="project-1" />);
    await user.type(screen.getByLabelText("你想让 Agent 完成什么"), "分析电芯寿命风险");
    await user.click(screen.getByRole("button", { name: "创建分析任务" }));
    expect(onCreate).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("至少填写一个已冻结的数据集 ID");
  });
});
