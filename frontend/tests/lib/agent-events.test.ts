import { describe, expect, it } from "vitest";

import {
  AGENT_EVENT_NAMES,
  decodeAgentEvent,
  findPendingApproval,
  isTerminalAgentEvent,
} from "@/lib/agent-events";

describe("Agent SSE contract", () => {
  it("decodes a complete event bound to the current run", () => {
    const event = decodeAgentEvent({
      event_id: "event-1",
      run_id: "run-1",
      sequence: 1,
      event_type: "STEP_STARTED",
      created_at: "2026-08-02T04:00:00Z",
      payload: { step_id: "advanced-input" },
    }, "run-1");

    expect(event.run_id).toBe("run-1");
    expect(event.event_type).toBe("STEP_STARTED");
  });

  it("rejects malformed, foreign-run, and unknown events", () => {
    const valid = {
      event_id: "event-1",
      run_id: "run-1",
      sequence: 1,
      event_type: "STEP_STARTED",
      created_at: "2026-08-02T04:00:00Z",
      payload: { step_id: "advanced-input" },
    };

    expect(() => decodeAgentEvent({ ...valid, run_id: "run-2" }, "run-1")).toThrow();
    expect(() => decodeAgentEvent({ ...valid, sequence: 0 }, "run-1")).toThrow();
    expect(() => decodeAgentEvent({ ...valid, event_type: "MADE_UP" }, "run-1")).toThrow();
    expect(() => decodeAgentEvent({ ...valid, payload: [] }, "run-1")).toThrow();
    expect(() => decodeAgentEvent({ ...valid, created_at: "2026-08-02 04:00:00" }, "run-1")).toThrow();
  });

  it("subscribes to the exact public backend event names", () => {
    expect(AGENT_EVENT_NAMES).toEqual([
      "RUN_CREATED",
      "RUN_DISPATCHED",
      "DISPATCH_FAILED",
      "STEP_STARTED",
      "STEP_COMPLETED",
      "STEP_FAILED",
      "RUN_CANCELLED",
      "APPROVAL_REQUESTED",
      "APPROVAL_APPROVED",
      "APPROVAL_REJECTED",
      "APPROVAL_EXPIRED",
      "RUN_COMPLETED",
      "RUN_FAILED",
    ]);
  });

  it("recognizes terminal events so a normal server close is not an error", () => {
    expect(isTerminalAgentEvent("RUN_COMPLETED")).toBe(true);
    expect(isTerminalAgentEvent("RUN_FAILED")).toBe(true);
    expect(isTerminalAgentEvent("RUN_CANCELLED")).toBe(true);
    expect(isTerminalAgentEvent("STEP_COMPLETED")).toBe(false);
  });

  it("finds only an unresolved approval request", () => {
    const request = {
      event_id: "event-1",
      run_id: "run-1",
      sequence: 1,
      event_type: "APPROVAL_REQUESTED",
      created_at: "2026-07-16T09:00:00Z",
      payload: { approval_id: "approval-1", approval_kind: "FORMAL_DECISION" },
    };
    expect(findPendingApproval([request])).toEqual({
      approvalId: "approval-1",
      approvalKind: "FORMAL_DECISION",
    });
    expect(findPendingApproval([
      request,
      {
        ...request,
        event_id: "event-2",
        sequence: 2,
        event_type: "APPROVAL_APPROVED",
      },
    ])).toBeNull();
  });
});
