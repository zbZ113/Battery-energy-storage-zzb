import { act, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentRunLiveView } from "@/app/agent/runs/[runId]/page";
import { RUNTIME_V7_STEPS } from "@/components/nine-step-tracker";
import type { AgentRunRecord } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), refresh: vi.fn() }),
}));

vi.mock("@/lib/advanced-analysis-contract", () => ({
  isFixedAdvancedRun: (value: AgentRunRecord) => {
    const stepIds = [
      "advanced-input", "rul-point", "rul-coverage", "soh", "rul-calibration",
      "rul-interval", "soh-calibration", "soh-band", "report",
    ];
    const intent = value.intent as { dataset_ids?: unknown; requested_outputs?: unknown } | undefined;
    const plan = value.plan as { steps?: { step_id?: unknown }[] } | undefined;
    const rawRequestedOutputs = intent?.requested_outputs;
    const requestedOutputs = Array.isArray(rawRequestedOutputs) ? rawRequestedOutputs : [];
    const datasetIds = intent?.dataset_ids;
    return requestedOutputs[0] === "advanced_single_cell_analysis"
      && Array.isArray(datasetIds)
      && datasetIds.length === 1
      && Array.isArray(plan?.steps)
      && plan.steps.length === stepIds.length
      && plan.steps.every((step, index) => step.step_id === stepIds[index]);
  },
}));

class FakeEventSource {
  static instance: FakeEventSource | null = null;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  listeners = new Map<string, (event: MessageEvent<string>) => void>();
  closed = false;

  constructor(url: string, init: EventSourceInit) {
    void url;
    void init;
    FakeEventSource.instance = this;
  }
  addEventListener(name: string, callback: EventListener) {
    this.listeners.set(name, callback as (event: MessageEvent<string>) => void);
  }
  close() { this.closed = true; }
  emit(name: string, data: object) {
    this.listeners.get(name)?.(new MessageEvent("message", { data: JSON.stringify(data) }));
  }
}

function run(
  status: string,
  stepIds: readonly string[] = RUNTIME_V7_STEPS.map((step) => step.id),
): AgentRunRecord {
  return {
    run_id: "run-1",
    project_id: "project-1",
    created_by_user_id: "user-1",
    status,
    planning_mode: "FIXED_FALLBACK",
    intent: {
      intent_id: "intent-1",
      project_id: "project-1",
      goal: "运行可信分析",
      dataset_ids: ["record-batch-1"],
      requested_outputs: ["advanced_single_cell_analysis"],
      created_at: "2026-07-16T09:00:00Z",
    },
    plan: {
      plan_version: "runtime-v7",
      intent_id: "intent-1",
      steps: stepIds.map((stepId) => ({ step_id: stepId })),
      planning_mode: "FIXED_FALLBACK",
      plan_hash: "a".repeat(64),
      created_at: "2026-07-16T09:00:00Z",
    },
    dispatch_status: "DISPATCHED",
    created_at: "2026-07-16T09:00:00Z",
    updated_at: "2026-07-16T09:00:00Z",
    completed_at: null,
  };
}

describe("AgentRunLiveView", () => {
  afterEach(() => {
    FakeEventSource.instance = null;
    vi.unstubAllGlobals();
  });

  it("refreshes the authoritative run after a terminal SSE event", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const loadRun = vi.fn()
      .mockResolvedValueOnce(run("RUNNING"))
      .mockResolvedValueOnce(run("COMPLETED"));
    render(
      <AgentRunLiveView
        eventsUrl={() => "http://localhost/events"}
        loadRun={loadRun}
        runId="run-1"
      />,
    );
    await waitFor(() => expect(loadRun).toHaveBeenCalledOnce());

    await act(async () => {
      FakeEventSource.instance?.emit("RUN_COMPLETED", {
        event_id: "event-terminal",
        run_id: "run-1",
        sequence: 3,
        event_type: "RUN_COMPLETED",
        created_at: "2026-07-16T09:01:00Z",
        payload: {},
      });
    });

    await waitFor(() => expect(loadRun).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("COMPLETED")).toBeInTheDocument();
    expect(FakeEventSource.instance?.closed).toBe(true);
  });

  it("does not let a slow initial read overwrite a newer terminal refresh", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    let resolveInitial!: (value: AgentRunRecord) => void;
    const initial = new Promise<AgentRunRecord>((resolve) => { resolveInitial = resolve; });
    const loadRun = vi.fn()
      .mockReturnValueOnce(initial)
      .mockResolvedValueOnce(run("COMPLETED"));
    render(
      <AgentRunLiveView
        eventsUrl={() => "http://localhost/events"}
        loadRun={loadRun}
        runId="run-1"
      />,
    );
    await waitFor(() => expect(loadRun).toHaveBeenCalledOnce());
    await act(async () => {
      FakeEventSource.instance?.emit("RUN_COMPLETED", {
        event_id: "event-terminal-race",
        run_id: "run-1",
        sequence: 3,
        event_type: "RUN_COMPLETED",
        created_at: "2026-07-16T09:01:00Z",
        payload: {},
      });
    });
    await waitFor(() => expect(screen.getByText("COMPLETED")).toBeInTheDocument());

    await act(async () => { resolveInitial(run("RUNNING")); });
    expect(screen.getByText("COMPLETED")).toBeInTheDocument();
    expect(screen.queryByText("RUNNING")).not.toBeInTheDocument();
  });

  it("maps only the fixed Runtime V7 step identities into the nine-step tracker", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    render(
      <AgentRunLiveView
        eventsUrl={() => "http://localhost/events"}
        loadRun={vi.fn().mockResolvedValue(run("RUNNING"))}
        runId="run-1"
      />,
    );
    await waitFor(() => expect(FakeEventSource.instance).not.toBeNull());

    await act(async () => {
      FakeEventSource.instance?.emit("STEP_COMPLETED", {
        event_id: "event-rul-point",
        run_id: "run-1",
        sequence: 2,
        event_type: "STEP_COMPLETED",
        created_at: "2026-07-16T09:00:30Z",
        payload: { step_id: "rul-point", result_id: "result-rul" },
      });
      FakeEventSource.instance?.emit("STEP_COMPLETED", {
        event_id: "event-unknown",
        run_id: "run-1",
        sequence: 3,
        event_type: "STEP_COMPLETED",
        created_at: "2026-07-16T09:00:40Z",
        payload: { step_id: "unrecognized-step", result_id: "result-unknown" },
      });
    });

    const tracker = screen.getByRole("list", { name: "九步执行状态" });
    expect(within(tracker).getAllByRole("listitem")).toHaveLength(9);
    expect(within(tracker).getByText("RUL 精度路线").closest("li")).toHaveTextContent("已完成");
    expect(within(tracker).getByText("RUL Conformal").closest("li")).toHaveTextContent("等待");
    expect(within(tracker).queryByText("unrecognized-step")).not.toBeInTheDocument();
  });

  it("does not label a different backend plan as Runtime V7", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    render(
      <AgentRunLiveView
        eventsUrl={() => "http://localhost/events"}
        loadRun={vi.fn().mockResolvedValue(run("RUNNING", ["features"]))}
        runId="run-1"
      />,
    );

    expect(await screen.findByText("当前运行不是 Runtime V7 九步计划")).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "九步执行状态" })).not.toBeInTheDocument();
  });

  it("offers a result entry for a completed fixed advanced run", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    render(
      <AgentRunLiveView
        eventsUrl={() => "http://localhost/events"}
        loadRun={vi.fn().mockResolvedValue(run("COMPLETED"))}
        runId="run-1"
      />,
    );

    expect(await screen.findByRole("link", { name: "查看本次分析结果" })).toHaveAttribute(
      "href",
      "/projects/project-1/cells/record-batch-1?run_id=run-1",
    );
  });

  it.each(["FAILED", "CANCELLED"])(
    "offers the partial result catalog for a %s fixed advanced run",
    async (status) => {
      vi.stubGlobal("EventSource", FakeEventSource);
      render(
        <AgentRunLiveView
          eventsUrl={() => "http://localhost/events"}
          loadRun={vi.fn().mockResolvedValue(run(status))}
          runId="run-1"
        />,
      );

      expect(await screen.findByRole("link", { name: "查看结果目录" })).toHaveAttribute(
        "href",
        "/projects/project-1/cells/record-batch-1?run_id=run-1",
      );
    },
  );

  it("clears the previous run evidence when the route changes", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const loadRun = vi.fn(async (runId: string) => ({ ...run("RUNNING"), run_id: runId }));
    const { rerender } = render(
      <AgentRunLiveView eventsUrl={(runId) => `http://localhost/${runId}`} loadRun={loadRun} runId="run-a" />,
    );
    await waitFor(() => expect(FakeEventSource.instance).not.toBeNull());
    await act(async () => {
      FakeEventSource.instance?.emit("STEP_COMPLETED", {
        event_id: "event-run-a",
        run_id: "run-a",
        sequence: 1,
        event_type: "STEP_COMPLETED",
        created_at: "2026-07-16T09:00:30Z",
        payload: { step_id: "rul-point", result_id: "result-run-a" },
      });
    });
    expect(screen.getByText("result-run-a")).toBeInTheDocument();

    rerender(
      <AgentRunLiveView eventsUrl={(runId) => `http://localhost/${runId}`} loadRun={loadRun} runId="run-b" />,
    );

    await waitFor(() => expect(screen.queryByText("result-run-a")).not.toBeInTheDocument());
    expect(screen.getByText("run-b")).toBeInTheDocument();
  });
});
