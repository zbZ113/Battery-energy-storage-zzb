import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AgentTimeline } from "@/components/agent-timeline";

describe("AgentTimeline", () => {
  it("makes the waiting state explicit", () => {
    render(<AgentTimeline events={[]} connectionState="connecting" runId="run-1" />);
    expect(screen.getByText("正在连接运行事件…")).toBeInTheDocument();
  });

  it("renders event evidence without deriving engineering numbers", () => {
    render(
      <AgentTimeline
        connectionState="open"
        runId="run-1"
        events={[
          {
            event_id: "evt-1",
            run_id: "run-1",
            sequence: 1,
            event_type: "STEP_COMPLETED",
            created_at: "2026-07-16T08:30:00Z",
            payload: { step_id: "quality_check", result_id: "result-verified" },
          },
        ]}
      />,
    );

    expect(screen.getByText("步骤完成")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "result-verified" })).toHaveAttribute(
      "href",
      "/results/result-verified?run_id=run-1",
    );
  });

  it("uses the backend approval event contract", () => {
    render(
      <AgentTimeline
        connectionState="open"
        runId="run-1"
        events={[
          {
            event_id: "evt-approval",
            run_id: "run-1",
            sequence: 2,
            event_type: "APPROVAL_APPROVED",
            created_at: "2026-07-16T08:31:00Z",
            payload: { approval_id: "approval-1" },
          },
        ]}
      />,
    );

    expect(screen.getByText("审批已通过")).toBeInTheDocument();
  });
});
