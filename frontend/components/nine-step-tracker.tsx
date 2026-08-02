import { Check, Circle, LoaderCircle, TriangleAlert } from "lucide-react";

import type { AgentEvent } from "@/lib/types";

export const RUNTIME_V7_STEPS = [
  { id: "advanced-input", label: "冻结分析输入" },
  { id: "rul-point", label: "RUL 精度路线" },
  { id: "rul-coverage", label: "RUL 覆盖路线" },
  { id: "soh", label: "SOH 有限时域" },
  { id: "rul-calibration", label: "RUL 校准" },
  { id: "rul-interval", label: "RUL Conformal" },
  { id: "soh-calibration", label: "SOH 校准" },
  { id: "soh-band", label: "SOH Conformal" },
  { id: "report", label: "审计报告" },
] as const;

type StepStatus = "waiting" | "retrying" | "running" | "completed" | "failed";
interface StepState {
  failureCode: string | null;
  status: StepStatus;
}

const STATUS_LABELS: Record<StepStatus, string> = {
  waiting: "等待",
  retrying: "等待重试",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
};

function eventStatus(event: AgentEvent): StepStatus | null {
  const eventType = event.event_type;
  if (eventType === "STEP_STARTED") return "running";
  if (eventType === "STEP_COMPLETED") return "completed";
  if (eventType === "STEP_FAILED") return event.payload.will_retry === true ? "retrying" : "failed";
  return null;
}

function statesFromEvents(events: AgentEvent[]): Map<string, StepState> {
  const supported = new Set(RUNTIME_V7_STEPS.map((step) => step.id));
  const states = new Map<string, StepState>();
  [...events].sort((left, right) => left.sequence - right.sequence).forEach((event) => {
    const stepId = event.payload.step_id;
    const status = eventStatus(event);
    if (typeof stepId === "string" && supported.has(stepId as (typeof RUNTIME_V7_STEPS)[number]["id"]) && status) {
      const failureCode = event.payload.failure_code;
      states.set(stepId, {
        failureCode: (status === "failed" || status === "retrying") && typeof failureCode === "string" && failureCode
          ? failureCode
          : null,
        status,
      });
    }
  });
  return states;
}

function StatusIcon({ status }: { status: StepStatus }) {
  if (status === "completed") return <Check aria-hidden="true" />;
  if (status === "running") return <LoaderCircle aria-hidden="true" className="spin" />;
  if (status === "failed") return <TriangleAlert aria-hidden="true" />;
  return <Circle aria-hidden="true" />;
}

export function NineStepTracker({ events }: { events: AgentEvent[] }) {
  const states = statesFromEvents(events);

  return (
    <ol aria-label="九步执行状态" aria-live="polite" className="nine-step-tracker">
      {RUNTIME_V7_STEPS.map((step, index) => {
        const state = states.get(step.id) ?? { failureCode: null, status: "waiting" as const };
        return (
          <li className={`nine-step-${state.status}`} key={step.id}>
            <span className="nine-step-number">{String(index + 1).padStart(2, "0")}</span>
            <span className="nine-step-name">{step.label}</span>
            <span className="nine-step-status">
              <span className="nine-step-state"><StatusIcon status={state.status} />{STATUS_LABELS[state.status]}</span>
              {state.failureCode ? <code className="nine-step-failure-code">{state.failureCode}</code> : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
