import { AlertTriangle, CheckCircle2, CircleDot, LoaderCircle, ShieldCheck } from "lucide-react";
import Link from "next/link";

import type { AgentEvent } from "@/lib/types";

export type EventConnectionState = "connecting" | "open" | "closed" | "error";

const EVENT_LABELS: Record<string, string> = {
  RUN_CREATED: "任务已创建",
  RUN_DISPATCHED: "任务已进入执行队列",
  DISPATCH_FAILED: "任务派发失败",
  STEP_STARTED: "步骤开始",
  STEP_COMPLETED: "步骤完成",
  STEP_FAILED: "步骤失败",
  APPROVAL_REQUESTED: "等待人工审批",
  APPROVAL_APPROVED: "审批已通过",
  APPROVAL_REJECTED: "审批已拒绝",
  APPROVAL_EXPIRED: "审批已过期",
  RUN_COMPLETED: "任务已完成",
  RUN_FAILED: "任务执行失败",
  RUN_CANCELLED: "任务已取消",
};

function evidenceId(payload: Record<string, unknown>): string | null {
  const result = payload.result_id;
  return typeof result === "string" && result ? result : null;
}

export function AgentTimeline({
  events,
  connectionState,
  runId,
}: {
  events: AgentEvent[];
  connectionState: EventConnectionState;
  runId: string;
}) {
  if (events.length === 0) {
    return (
      <section className="empty-state compact">
        {connectionState === "connecting" ? (
          <LoaderCircle aria-hidden="true" className="spin" />
        ) : (
          <CircleDot aria-hidden="true" />
        )}
        <h2>{connectionState === "connecting" ? "正在连接运行事件…" : "尚无运行事件"}</h2>
        <p>事件由FastAPI签发后会显示在这里，页面不会自行推测运行状态。</p>
      </section>
    );
  }

  return (
    <ol className="timeline">
      {events.map((event) => {
        const resultId = evidenceId(event.payload);
        const failed = event.event_type.includes("FAILED") || event.event_type.includes("REJECTED");
        return (
          <li className="timeline-item" key={event.event_id}>
            <span className={`timeline-marker ${failed ? "danger" : ""}`}>
              {failed ? <AlertTriangle aria-hidden="true" /> : <CheckCircle2 aria-hidden="true" />}
            </span>
            <article>
              <header>
                <strong>{EVENT_LABELS[event.event_type] ?? event.event_type}</strong>
                <time dateTime={event.created_at}>{new Date(event.created_at).toLocaleString("zh-CN")}</time>
              </header>
              {resultId ? (
                <p className="evidence-reference">
                  <ShieldCheck aria-hidden="true" size={15} />
                  证据结果：
                  <Link href={`/results/${encodeURIComponent(resultId)}?run_id=${encodeURIComponent(runId)}`}>
                    <code>{resultId}</code>
                  </Link>
                </p>
              ) : null}
            </article>
          </li>
        );
      })}
    </ol>
  );
}
