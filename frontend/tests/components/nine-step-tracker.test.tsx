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

  it("shows elapsed time only when server events provide a valid start and end", () => {
    render(
      <NineStepTracker
        events={[
          {
            event_id: "event-started",
            run_id: "run-1",
            sequence: 1,
            event_type: "STEP_STARTED",
            created_at: "2026-08-02T04:00:00.000Z",
            payload: { step_id: "rul-point" },
          },
          {
            event_id: "event-completed",
            run_id: "run-1",
            sequence: 2,
            event_type: "STEP_COMPLETED",
            created_at: "2026-08-02T04:00:01.500Z",
            payload: { step_id: "rul-point", result_id: "result-1" },
          },
          {
            event_id: "event-no-start",
            run_id: "run-1",
            sequence: 3,
            event_type: "STEP_COMPLETED",
            created_at: "2026-08-02T04:00:02.000Z",
            payload: { step_id: "soh", result_id: "result-2" },
          },
        ]}
      />,
    );

    expect(screen.getByText("RUL 精度路线").closest("li")).toHaveTextContent("耗时 1.5 s");
    expect(screen.getByText("SOH 有限时域").closest("li")).not.toHaveTextContent("耗时");
  });

  it("adds server-timed retry attempts without using browser time", () => {
    render(
      <NineStepTracker
        events={[
          {
            event_id: "attempt-1-start",
            run_id: "run-1",
            sequence: 1,
            event_type: "STEP_STARTED",
            created_at: "2026-08-02T04:00:00Z",
            payload: { step_id: "soh-band" },
          },
          {
            event_id: "attempt-1-failed",
            run_id: "run-1",
            sequence: 2,
            event_type: "STEP_FAILED",
            created_at: "2026-08-02T04:00:01Z",
            payload: { step_id: "soh-band", will_retry: true },
          },
          {
            event_id: "attempt-2-start",
            run_id: "run-1",
            sequence: 3,
            event_type: "STEP_STARTED",
            created_at: "2026-08-02T04:00:02Z",
            payload: { step_id: "soh-band" },
          },
          {
            event_id: "attempt-2-completed",
            run_id: "run-1",
            sequence: 4,
            event_type: "STEP_COMPLETED",
            created_at: "2026-08-02T04:00:04Z",
            payload: { step_id: "soh-band", result_id: "result-3" },
          },
        ]}
      />,
    );

    expect(screen.getByText("SOH Conformal").closest("li")).toHaveTextContent("耗时 3.0 s");
  });
});
