import type { AgentEvent } from "@/lib/types";

export const AGENT_EVENT_NAMES = [
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
] as const;

export type AgentEventType = (typeof AGENT_EVENT_NAMES)[number];

const TERMINAL_EVENTS: ReadonlySet<string> = new Set([
  "RUN_COMPLETED",
  "RUN_FAILED",
  "RUN_CANCELLED",
]);

export function isTerminalAgentEvent(eventType: string): boolean {
  return TERMINAL_EVENTS.has(eventType);
}

export interface PendingApproval {
  approvalId: string;
  approvalKind: string;
}

export function findPendingApproval(events: AgentEvent[]): PendingApproval | null {
  const resolved = new Set<string>();
  for (const event of events) {
    if (["APPROVAL_APPROVED", "APPROVAL_REJECTED", "APPROVAL_EXPIRED"].includes(event.event_type)) {
      const approvalId = event.payload.approval_id;
      if (typeof approvalId === "string") resolved.add(approvalId);
    }
  }
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.event_type !== "APPROVAL_REQUESTED") continue;
    const approvalId = event.payload.approval_id;
    const approvalKind = event.payload.approval_kind;
    if (
      typeof approvalId === "string"
      && typeof approvalKind === "string"
      && !resolved.has(approvalId)
    ) {
      return { approvalId, approvalKind };
    }
  }
  return null;
}
