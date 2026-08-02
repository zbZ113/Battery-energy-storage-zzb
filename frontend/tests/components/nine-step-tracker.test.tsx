import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { NineStepTracker } from "@/components/nine-step-tracker";

describe("NineStepTracker", () => {
  it("announces a server-issued step failure and its safe failure code", () => {
    render(
      <NineStepTracker
        events={[
          {
            event_id: "event-failed",
            run_id: "run-1",
            sequence: 7,
            event_type: "STEP_FAILED",
            created_at: "2026-08-02T04:00:00Z",
            payload: {
              step_id: "rul-interval",
              failure_code: "CALIBRATION_EVIDENCE_MISSING",
              will_retry: false,
            },
          },
        ]}
      />,
    );

    const tracker = screen.getByRole("list", { name: "九步执行状态" });
    expect(tracker).toHaveAttribute("aria-live", "polite");
    const failedStep = within(tracker).getByText("RUL Conformal").closest("li");
    expect(failedStep).toHaveTextContent("失败");
    expect(failedStep).toHaveTextContent("CALIBRATION_EVIDENCE_MISSING");
  });

  it("distinguishes a retryable failure from a terminal failure", () => {
    render(
      <NineStepTracker
        events={[
          {
            event_id: "event-retry",
            run_id: "run-1",
            sequence: 4,
            event_type: "STEP_FAILED",
            created_at: "2026-08-02T04:00:00Z",
            payload: {
              step_id: "soh",
              failure_code: "TRANSIENT_TOOL_ERROR",
              will_retry: true,
            },
          },
        ]}
      />,
    );

    const retrying = screen.getByText("SOH 有限时域").closest("li");
    expect(retrying).toHaveTextContent("等待重试");
    expect(retrying).not.toHaveTextContent("失败TRANSIENT_TOOL_ERROR");
  });
});
