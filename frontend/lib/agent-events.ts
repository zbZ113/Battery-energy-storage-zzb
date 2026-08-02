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
const AGENT_EVENT_SET: ReadonlySet<string> = new Set(AGENT_EVENT_NAMES);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isJsonValue(value: unknown): boolean {
  if (value === null || typeof value === "string" || typeof value === "boolean") return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(isJsonValue);
  return isRecord(value) && Object.values(value).every(isJsonValue);
}

function nonblank(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

export function decodeAgentEvent(value: unknown, expectedRunId: string): AgentEvent {
  if (!isRecord(value)) throw new Error("Agent event must be an object");
  const { created_at: createdAt, event_id: eventId, event_type: eventType, payload, run_id: runId, sequence } = value;
  if (!nonblank(eventId) || !nonblank(runId) || runId !== expectedRunId) {
    throw new Error("Agent event identity does not match the current run");
  }
  if (!Number.isInteger(sequence) || (sequence as number) < 1) {
    throw new Error("Agent event sequence must be a positive integer");
  }
  if (!nonblank(eventType) || !AGENT_EVENT_SET.has(eventType)) {
    throw new Error("Agent event type is unsupported");
  }
  if (!nonblank(createdAt) || !/(?:Z|[+-]\d{2}:\d{2})$/.test(createdAt) || Number.isNaN(Date.parse(createdAt))) {
    throw new Error("Agent event timestamp must include a timezone");
  }
  if (!isRecord(payload) || !isJsonValue(payload)) {
    throw new Error("Agent event payload must be a JSON object");
  }
  return {
    created_at: createdAt,
    event_id: eventId,
    event_type: eventType,
    payload,
    run_id: runId,
    sequence: sequence as number,
  };
}

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
