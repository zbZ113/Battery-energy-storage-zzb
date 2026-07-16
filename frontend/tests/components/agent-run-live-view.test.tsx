import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentRunLiveView } from "@/app/agent/runs/[runId]/page";
import type { AgentRunRecord } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), refresh: vi.fn() }),
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

function run(status: string): AgentRunRecord {
  return {
    run_id: "run-1",
    project_id: "project-1",
    status,
    created_at: "2026-07-16T09:00:00Z",
    updated_at: "2026-07-16T09:00:00Z",
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
});
