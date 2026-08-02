import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { WorkflowRail } from "@/components/workflow-rail";

describe("WorkflowRail", () => {
  it("does not infer completed business stages from the current position", () => {
    render(<WorkflowRail current="agent" />);

    expect(screen.getByText("九步 Agent").closest("li")).toHaveAttribute(
      "aria-current",
      "step",
    );
    expect(screen.getByText("样例 / 上传").closest("li")).not.toHaveClass(
      "workflow-complete",
    );
    expect(screen.queryByText("已完成")).not.toBeInTheDocument();
  });
});
