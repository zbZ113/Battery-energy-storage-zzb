"use client";

import { use, useEffect, useRef, useState } from "react";
import { Radio, RefreshCw } from "lucide-react";

import { AgentTimeline, type EventConnectionState } from "@/components/agent-timeline";
import { ApprovalPanel } from "@/components/approval-panel";
import { AppShell } from "@/components/app-shell";
import { NineStepTracker, RUNTIME_V7_STEPS } from "@/components/nine-step-tracker";
import { SseConnectionNotice } from "@/components/sse-connection-notice";
import { AGENT_EVENT_NAMES, decodeAgentEvent, findPendingApproval, isTerminalAgentEvent } from "@/lib/agent-events";
import {
  agentEventsUrl,
  approveAgentRun,
  getAgentRun,
  rejectAgentRun,
} from "@/lib/api-client";
import type { AgentEvent, AgentRunRecord } from "@/lib/types";

function isRuntimeV7Run(run: AgentRunRecord): boolean {
  const plan = run.plan;
  if (typeof plan !== "object" || plan === null) return false;
  const steps = (plan as { steps?: unknown }).steps;
  if (!Array.isArray(steps) || steps.length !== RUNTIME_V7_STEPS.length) return false;
  return steps.every((step, index) => (
    typeof step === "object"
    && step !== null
    && (step as { step_id?: unknown }).step_id === RUNTIME_V7_STEPS[index].id
  ));
}

export default function AgentRunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);
  return <AgentRunLiveView runId={runId} />;
}

interface AgentRunLiveViewProps {
  runId: string;
  loadRun?: (runId: string) => Promise<AgentRunRecord>;
  eventsUrl?: (runId: string) => string;
}

export function AgentRunLiveView({
  runId,
  loadRun = getAgentRun,
  eventsUrl = agentEventsUrl,
}: AgentRunLiveViewProps) {
  return <AgentRunLiveSession eventsUrl={eventsUrl} key={runId} loadRun={loadRun} runId={runId} />;
}

function AgentRunLiveSession({
  runId,
  loadRun,
  eventsUrl,
}: Required<AgentRunLiveViewProps>) {
  const [run, setRun] = useState<AgentRunRecord | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connection, setConnection] = useState<EventConnectionState>("connecting");
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const terminalEventReceived = useRef(false);

  useEffect(() => {
    let active = true;
    let refreshSequence = 0;
    const refreshRun = (failureMessage: string) => {
      const sequence = ++refreshSequence;
      void loadRun(runId)
        .then((record) => {
          if (active && sequence === refreshSequence) setRun(record);
        })
        .catch(() => {
          if (active && sequence === refreshSequence) setError(failureMessage);
        });
    };
    refreshRun("无法读取该运行，请检查运行 ID 和项目权限。");
    terminalEventReceived.current = false;
    const source = new EventSource(eventsUrl(runId), { withCredentials: true });
    source.onopen = () => setConnection("open");
    source.onerror = () => {
      source.close();
      setConnection(terminalEventReceived.current ? "closed" : "error");
    };
    const receive = (message: MessageEvent<string>) => {
      try {
        const event = decodeAgentEvent(JSON.parse(message.data), runId);
        setEvents((current) => current.some((item) => item.event_id === event.event_id) ? current : [...current, event]);
        if (isTerminalAgentEvent(event.event_type)) {
          terminalEventReceived.current = true;
          source.close();
          setConnection("closed");
          refreshRun("运行已结束，但最终状态刷新失败。请重新打开该页面。");
        }
      } catch { setError("收到无法识别的运行事件，已停止解释该事件。"); }
    };
    for (const name of AGENT_EVENT_NAMES) source.addEventListener(name, receive as EventListener);
    return () => {
      active = false;
      source.close();
    };
  }, [connectionAttempt, eventsUrl, loadRun, runId]);

  const pendingApproval = run?.status === "AWAITING_APPROVAL"
    ? findPendingApproval(events)
    : null;

  return (
    <AppShell workflowStage="agent">
      <header className="page-header"><div><p className="eyebrow">AGENT RUN</p><h1>多智能体运行时间线</h1><p>运行 ID：<code>{runId}</code></p></div><span className={`status-pill status-${run?.status?.toLowerCase() ?? "loading"}`}><Radio />{run?.status ?? "读取中"}</span></header>
      {!run && !error ? <div className="loading-state"><RefreshCw className="spin" />正在读取运行状态…</div> : null}
      {error ? <p className="alert alert-error" role="alert">{error}</p> : null}
      {pendingApproval ? (
        <ApprovalPanel
          approvalId={pendingApproval.approvalId}
          approvalKind={pendingApproval.approvalKind}
          onApprove={async (approvalId, reason) => {
            setRun(await approveAgentRun(runId, approvalId, reason));
          }}
          onReject={async (approvalId, reason) => {
            setRun(await rejectAgentRun(runId, approvalId, reason));
          }}
        />
      ) : null}
      <div className="run-workspace">
        <section className="workspace-panel run-audit-panel">
          <div className="panel-heading"><Radio /><div><h2>实时审计事件</h2><p>连接状态：{connection}</p></div></div>
          <SseConnectionNotice
            onReconnect={() => {
              setConnection("connecting");
              setConnectionAttempt((attempt) => attempt + 1);
            }}
            state={connection}
          />
          <AgentTimeline events={events} connectionState={connection} runId={runId} />
        </section>
        <aside className="workspace-panel run-status-panel">
          <div className="panel-heading"><Radio aria-hidden="true" /><div><h2>九步实时进度</h2><p>只映射 Runtime V7 固定步骤身份。</p></div></div>
          {!run ? <p className="muted" role="status">正在核验运行计划…</p> : null}
          {run && !isRuntimeV7Run(run) ? <p className="muted" role="status">当前运行不是 Runtime V7 九步计划</p> : null}
          {run && isRuntimeV7Run(run) && events.length === 0 && connection !== "open" ? (
            <p className="muted" role="status">九步进度尚未加载；页面不会推测步骤状态。</p>
          ) : null}
          {run && isRuntimeV7Run(run) && (events.length > 0 || connection === "open") ? (
            <NineStepTracker events={events} />
          ) : null}
        </aside>
      </div>
    </AppShell>
  );
}
