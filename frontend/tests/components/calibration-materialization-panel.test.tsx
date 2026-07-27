import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CalibrationMaterializationPanel,
  type CalibrationMaterializationPanelProps,
} from "@/components/calibration-materialization-panel";
import {
  decodeActiveModelRouteList,
  decodeAdvancedCalibrationMaterialization,
  decodeAdvancedCalibrationMaterializationList,
  decodeCreateAdvancedCalibrationMaterializationResponse,
  type ActiveModelRouteSummary,
  type AdvancedCalibrationMaterialization,
  type SupportedCutoffCycle,
} from "@/components/calibration-materialization-contract";

const SHA = {
  artifact: "1".repeat(64),
  ledger: "2".repeat(64),
  normalization: "3".repeat(64),
  sample: "4".repeat(64),
  source: "5".repeat(64),
};

function route(
  cutoffCycle: SupportedCutoffCycle = 100,
  task: "RUL" | "SOH" = "RUL",
): ActiveModelRouteSummary {
  return {
    artifact_id: `artifact-${task.toLowerCase()}-${cutoffCycle}`,
    artifact_kind: task === "RUL" ? "cyclepatch_direct" : "current_hybrid",
    artifact_manifest_sha256: SHA.artifact,
    cutoff_cycle: cutoffCycle,
    data_version: "matr-three-batch-v1",
    decision_event_id: `decision-${task.toLowerCase()}-${cutoffCycle}`,
    feature_version: "advanced-feature-v1",
    ledger_head_sha256: SHA.ledger,
    ledger_sequence_number: cutoffCycle,
    model_version: "advanced-model-v1",
    normalization_statistics_sha256: SHA.normalization,
    route_role: task === "RUL" ? "COVERAGE" : "MEAN_ACCURACY",
    split_version: "matr-cell-split-v1",
    task,
  };
}

function materialization(
  status: AdvancedCalibrationMaterialization["status"],
  activeRoute = route(),
  overrides: Partial<AdvancedCalibrationMaterialization> = {},
): AdvancedCalibrationMaterialization {
  const terminal = ["READY", "FAILED", "STALE"].includes(status);
  return {
    artifact_id: activeRoute.artifact_id,
    artifact_manifest_sha256: activeRoute.artifact_manifest_sha256,
    completed_at: terminal ? "2026-07-27T08:02:00Z" : null,
    created_at: "2026-07-27T08:00:00Z",
    cutoff_cycle: activeRoute.cutoff_cycle,
    data_version: activeRoute.data_version,
    decision_event_id: activeRoute.decision_event_id,
    failure_code: status === "FAILED" ? "CALIBRATION_INPUT_INVALID" : null,
    feature_version: activeRoute.feature_version,
    ledger_head_sha256: activeRoute.ledger_head_sha256,
    ledger_sequence_number: activeRoute.ledger_sequence_number,
    materialization_id: `materialization-${status.toLowerCase()}`,
    normalization_statistics_sha256:
      activeRoute.normalization_statistics_sha256,
    project_id: "project-1",
    route_role: activeRoute.route_role,
    sample_count: status === "READY" || status === "STALE" ? 18 : 0,
    sample_manifest_sha256:
      status === "READY" || status === "STALE" ? SHA.sample : null,
    source_identity_sha256: SHA.source,
    source_registration_id: "matr-three-batch-final-v1",
    split_version: activeRoute.split_version,
    started_at: status === "PENDING" ? null : "2026-07-27T08:01:00Z",
    status,
    task: activeRoute.task,
    ...overrides,
  };
}

function panelProps(
  overrides: Partial<CalibrationMaterializationPanelProps> = {},
): CalibrationMaterializationPanelProps {
  return {
    activeRoutes: [route()],
    createMaterialization: vi.fn(),
    initialMaterializations: [],
    loadMaterialization: vi.fn(),
    principalRole: "ADMIN",
    ...overrides,
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("advanced calibration materialization decoders", () => {
  it("decodes only the path-free server contracts", () => {
    const activeRoute = route();
    const ready = materialization("READY", activeRoute);

    expect(decodeActiveModelRouteList([activeRoute])).toEqual([activeRoute]);
    expect(decodeAdvancedCalibrationMaterialization(ready)).toEqual(ready);
    expect(decodeAdvancedCalibrationMaterializationList([ready])).toEqual([
      ready,
    ]);
    expect(
      decodeCreateAdvancedCalibrationMaterializationResponse({
        dispatch: {
          materialization_id: ready.materialization_id,
          task_id: "queue-task-1",
        },
        materialization: ready,
      }),
    ).toEqual({
      dispatch: {
        materialization_id: ready.materialization_id,
        task_id: "queue-task-1",
      },
      materialization: ready,
    });
  });

  it.each([
    ["status", { status: "ALMOST_READY" }],
    ["task", { task: "CAPACITY" }],
    ["route", { route_role: "BEST_SCORE" }],
    ["SHA", { sample_manifest_sha256: "not-a-sha" }],
    ["timestamp", { completed_at: "2026-07-27 08:02:00" }],
  ])("rejects a malformed %s instead of weakening readiness", (_name, patch) => {
    expect(() =>
      decodeAdvancedCalibrationMaterialization({
        ...materialization("READY"),
        ...patch,
      }),
    ).toThrow("invalid advanced calibration");
  });

  it("rejects active routes and materializations outside frozen cutoffs", () => {
    expect(() =>
      decodeActiveModelRouteList([{ ...route(), cutoff_cycle: 30 }]),
    ).toThrow("invalid advanced calibration");
    expect(() =>
      decodeAdvancedCalibrationMaterialization({
        ...materialization("READY"),
        cutoff_cycle: 200,
      }),
    ).toThrow("invalid advanced calibration");
  });

  it("rejects extra business values, arrays and sample result identities", () => {
    expect(() =>
      decodeAdvancedCalibrationMaterialization({
        ...materialization("READY"),
        observed_soh: [1, 0.99],
        predicted_soh: [0.99, 0.98],
        sample_result_ids: ["result-secret-1"],
      }),
    ).toThrow("invalid advanced calibration");
  });

  it("accepts audited sample identity retained after READY becomes STALE", () => {
    const stale = materialization("STALE", route(), {
      failure_code: "ACTIVE_ROUTE_CHANGED",
    });

    expect(decodeAdvancedCalibrationMaterialization(stale)).toEqual(stale);
  });
});

describe("CalibrationMaterializationPanel", () => {
  it("renders PENDING, RUNNING, READY, FAILED and STALE as distinct states", () => {
    const routes = [
      route(20),
      route(50),
      route(100),
      route(150),
      route(100, "SOH"),
    ];
    const statuses: AdvancedCalibrationMaterialization["status"][] = [
      "PENDING",
      "RUNNING",
      "READY",
      "FAILED",
      "STALE",
    ];

    render(
      <CalibrationMaterializationPanel
        {...panelProps({
          activeRoutes: routes,
          initialMaterializations: routes.map((activeRoute, index) =>
            materialization(statuses[index], activeRoute),
          ),
          principalRole: "MEMBER",
        })}
      />,
    );

    expect(screen.getByText("等待队列")).toHaveAttribute(
      "data-materialization-status",
      "PENDING",
    );
    expect(screen.getByText("正在生成")).toHaveAttribute(
      "data-materialization-status",
      "RUNNING",
    );
    expect(screen.getByText("证据就绪")).toHaveAttribute(
      "data-materialization-status",
      "READY",
    );
    expect(screen.getByText("生成失败")).toHaveAttribute(
      "data-materialization-status",
      "FAILED",
    );
    expect(screen.getByText("证据过期")).toHaveAttribute(
      "data-materialization-status",
      "STALE",
    );
  });

  it("lets ADMIN trigger a server-listed route using identity-only fields", async () => {
    const activeRoute = route(100);
    const pending = materialization("PENDING", activeRoute);
    const createMaterialization = vi.fn().mockResolvedValue({
      dispatch: {
        materialization_id: pending.materialization_id,
        task_id: "queue-task-1",
      },
      materialization: pending,
    });

    render(
      <CalibrationMaterializationPanel
        {...panelProps({
          activeRoutes: [activeRoute],
          createMaterialization,
        })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "生成校准证据" }));

    await act(async () => undefined);
    expect(createMaterialization).toHaveBeenCalledWith(
      {
        cutoff_cycle: 100,
        route_role: "COVERAGE",
        task: "RUL",
      },
      expect.any(String),
    );
    expect(JSON.stringify(createMaterialization.mock.calls)).not.toMatch(
      /observed|predicted|sample_result/i,
    );
    expect(await screen.findByText("等待队列")).toBeInTheDocument();
  });

  it("gives MEMBER a read-only view with no mutation control", () => {
    render(
      <CalibrationMaterializationPanel
        {...panelProps({ principalRole: "MEMBER" })}
      />,
    );

    expect(
      screen.queryByRole("button", { name: "生成校准证据" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("仅 ADMIN 可生成校准证据")).toBeInTheDocument();
  });

  it("keeps active routes with incompatible calibration semantics read-only", () => {
    const cutoff20Coverage = {
      ...route(20),
      route_role: "COVERAGE" as const,
    };
    const pointAccuracy = {
      ...route(100),
      route_role: "POINT_ACCURACY" as const,
    };
    const sohDefault = {
      ...route(100, "SOH"),
      route_role: "DEFAULT" as const,
    };
    const unsupportedCutoff = {
      ...route(100),
      cutoff_cycle: 30,
    } as unknown as ActiveModelRouteSummary;

    render(
      <CalibrationMaterializationPanel
        {...panelProps({
          activeRoutes: [
            cutoff20Coverage,
            pointAccuracy,
            sohDefault,
            unsupportedCutoff,
          ],
        })}
      />,
    );

    expect(
      screen.queryByRole("button", { name: "生成校准证据" }),
    ).not.toBeInTheDocument();
    expect(screen.getAllByText("不支持校准物化")).toHaveLength(4);
  });

  it("shows the complete frozen readiness identity and timestamps", () => {
    render(
      <CalibrationMaterializationPanel
        {...panelProps({
          initialMaterializations: [materialization("READY")],
          principalRole: "MEMBER",
        })}
      />,
    );

    expect(screen.getByText("matr-three-batch-v1")).toBeInTheDocument();
    expect(screen.getByText("matr-cell-split-v1")).toBeInTheDocument();
    expect(screen.getByText("advanced-feature-v1")).toBeInTheDocument();
    expect(screen.getByText("advanced-model-v1")).toBeInTheDocument();
    expect(screen.getByText("artifact-rul-100")).toBeInTheDocument();
    expect(screen.getByText("matr-three-batch-final-v1")).toBeInTheDocument();
    expect(screen.getByText("2026-07-27T08:00:00Z")).toBeInTheDocument();
    expect(screen.getByText("2026-07-27T08:02:00Z")).toBeInTheDocument();
    expect(screen.getByText(SHA.normalization)).toBeInTheDocument();
  });

  it("shows obsolete evidence as STALE when the same route coordinate was replaced", () => {
    const activeRoute = route(100);
    const previousRoute = {
      ...activeRoute,
      artifact_id: "artifact-rul-previous",
      decision_event_id: "decision-rul-previous",
    };
    const stale = materialization("STALE", previousRoute, {
      failure_code: "ACTIVE_ROUTE_CHANGED",
    });

    render(
      <CalibrationMaterializationPanel
        {...panelProps({
          activeRoutes: [activeRoute],
          initialMaterializations: [stale],
        })}
      />,
    );

    expect(screen.getByText("证据过期")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "生成校准证据" }),
    ).toBeInTheDocument();
  });

  it("never exposes arrays or per-sample result identities and marks SHA selectable", () => {
    const unsafeRecord = {
      ...materialization("READY"),
      observed_soh: [1, 0.99],
      predicted_soh: [0.98, 0.97],
      sample_result_ids: ["sample-result-secret"],
    } as AdvancedCalibrationMaterialization;

    const { container } = render(
      <CalibrationMaterializationPanel
        {...panelProps({ initialMaterializations: [unsafeRecord] })}
      />,
    );

    expect(container).not.toHaveTextContent("sample-result-secret");
    expect(container).not.toHaveTextContent("0.99");
    expect(container).not.toHaveTextContent("0.98");
    expect(container.querySelectorAll(".calibration-sha")).not.toHaveLength(0);
    for (const hash of container.querySelectorAll(".calibration-sha")) {
      expect(hash).toHaveClass("calibration-sha--selectable");
    }
  });

  it("polls a nonterminal record and stops after the server returns READY", async () => {
    vi.useFakeTimers();
    const activeRoute = route();
    const running = materialization("RUNNING", activeRoute);
    const ready = materialization("READY", activeRoute, {
      materialization_id: running.materialization_id,
    });
    const loadMaterialization = vi.fn().mockResolvedValue(ready);

    render(
      <CalibrationMaterializationPanel
        {...panelProps({
          activeRoutes: [activeRoute],
          initialMaterializations: [running],
          loadMaterialization,
          pollIntervalMs: 25,
        })}
      />,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(25);
    });
    expect(loadMaterialization).toHaveBeenCalledTimes(1);
    expect(screen.getByText("证据就绪")).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });
    expect(loadMaterialization).toHaveBeenCalledTimes(1);
  });

  it("bounds polling and cancels pending work on unmount", async () => {
    vi.useFakeTimers();
    const running = materialization("RUNNING");
    const loadMaterialization = vi.fn().mockResolvedValue(running);
    const { unmount } = render(
      <CalibrationMaterializationPanel
        {...panelProps({
          initialMaterializations: [running],
          loadMaterialization,
          maxPollAttempts: 2,
          pollIntervalMs: 25,
        })}
      />,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(25);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(25);
    });
    expect(loadMaterialization).toHaveBeenCalledTimes(2);
    unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });
    expect(loadMaterialization).toHaveBeenCalledTimes(2);
  });

  it("reports API failure without inventing a READY state", async () => {
    const createMaterialization = vi
      .fn()
      .mockRejectedValue(new Error("network unavailable"));
    render(
      <CalibrationMaterializationPanel
        {...panelProps({ createMaterialization })}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "生成校准证据" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "校准证据请求失败",
    );
    expect(screen.queryByText("证据就绪")).not.toBeInTheDocument();
  });

  it("reuses the same idempotency key after an uncertain network failure", async () => {
    const randomUUID = vi.fn().mockReturnValue(
      "84b2dadf-e8b5-4868-9ca6-088b129582ab",
    );
    vi.stubGlobal("crypto", { randomUUID });
    const createMaterialization = vi
      .fn()
      .mockRejectedValue(new Error("response lost"));
    render(
      <CalibrationMaterializationPanel
        {...panelProps({ createMaterialization })}
      />,
    );

    const action = screen.getByRole("button", {
      name: "生成校准证据",
    });
    fireEvent.click(action);
    await screen.findByRole("alert");
    fireEvent.click(action);
    await waitFor(() => expect(createMaterialization).toHaveBeenCalledTimes(2));

    expect(createMaterialization.mock.calls[0][1]).toBe(
      "advanced-calibration-84b2dadf-e8b5-4868-9ca6-088b129582ab",
    );
    expect(createMaterialization.mock.calls[1][1]).toBe(
      "advanced-calibration-84b2dadf-e8b5-4868-9ca6-088b129582ab",
    );
    expect(randomUUID).toHaveBeenCalledTimes(1);
  });
});
