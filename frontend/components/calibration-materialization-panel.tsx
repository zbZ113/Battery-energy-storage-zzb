"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type {
  ActiveModelRouteSummary,
  AdvancedCalibrationMaterialization,
  CreateAdvancedCalibrationMaterializationRequest,
  CreateAdvancedCalibrationMaterializationResponse,
} from "@/components/calibration-materialization-contract";
import { isSupportedCutoffCycle } from "@/components/calibration-materialization-contract";

export interface CalibrationMaterializationPanelProps {
  activeRoutes: ActiveModelRouteSummary[];
  initialMaterializations: AdvancedCalibrationMaterialization[];
  principalRole: "ADMIN" | "MEMBER";
  createMaterialization: (
    request: CreateAdvancedCalibrationMaterializationRequest,
    idempotencyKey: string,
  ) => Promise<CreateAdvancedCalibrationMaterializationResponse>;
  loadMaterialization: (
    materializationId: string,
  ) => Promise<AdvancedCalibrationMaterialization>;
  pollIntervalMs?: number;
  maxPollAttempts?: number;
}

const STATUS_PRESENTATION: Record<
  AdvancedCalibrationMaterialization["status"],
  { label: string; explanation: string }
> = {
  FAILED: {
    explanation: "服务端未生成可用证据，可由 ADMIN 对当前活动路由重新请求。",
    label: "生成失败",
  },
  PENDING: {
    explanation: "身份请求已登记，正在等待受控 Worker 领取。",
    label: "等待队列",
  },
  READY: {
    explanation: "服务端已验证并登记校准样本清单。",
    label: "证据就绪",
  },
  RUNNING: {
    explanation: "受控 Worker 正在生成并核验校准证据。",
    label: "正在生成",
  },
  STALE: {
    explanation: "活动模型路由已变化，该证据不可再供新任务使用。",
    label: "证据过期",
  },
};

const TERMINAL_STATUSES = new Set<
  AdvancedCalibrationMaterialization["status"]
>(["READY", "FAILED", "STALE"]);

function routeKey(route: ActiveModelRouteSummary): string {
  return [
    route.task,
    route.cutoff_cycle,
    route.route_role,
    route.artifact_id,
    route.decision_event_id,
  ].join(":");
}

function routeCoordinateKey(
  value: Pick<
    ActiveModelRouteSummary,
    "task" | "cutoff_cycle" | "route_role"
  >,
): string {
  return [value.task, value.cutoff_cycle, value.route_role].join(":");
}

function supportsCalibrationMaterialization(
  route: ActiveModelRouteSummary,
): boolean {
  if (!isSupportedCutoffCycle(route.cutoff_cycle)) return false;
  if (route.task === "RUL") {
    return route.cutoff_cycle === 20
      ? route.route_role === "DEFAULT"
      : route.route_role === "COVERAGE";
  }
  return (
    route.route_role === "MEAN_ACCURACY"
    || route.route_role === "TAIL_EFFICIENCY"
  );
}

function upsertMaterialization(
  current: AdvancedCalibrationMaterialization[],
  next: AdvancedCalibrationMaterialization,
): AdvancedCalibrationMaterialization[] {
  const index = current.findIndex(
    (item) => item.materialization_id === next.materialization_id,
  );
  if (index === -1) return [next, ...current];
  const updated = [...current];
  updated[index] = next;
  return updated;
}

function ShaEvidence({
  label,
  value,
}: {
  label: string;
  value: string | null;
}) {
  if (value === null) return null;
  return (
    <div className="calibration-evidence-line">
      <span>{label}</span>
      <code
        className="calibration-sha calibration-sha--selectable"
        title={`${label}；可选择复制并用于制品核验`}
      >
        {value}
      </code>
    </div>
  );
}

function ReadinessValue({
  label,
  value,
}: {
  label: string;
  value: string | null;
}) {
  if (value === null) return null;
  return (
    <div className="calibration-evidence-line">
      <span>{label}</span>
      <code>{value}</code>
    </div>
  );
}

function StatusBadge({
  status,
}: {
  status: AdvancedCalibrationMaterialization["status"];
}) {
  const presentation = STATUS_PRESENTATION[status];
  return (
    <span
      className={`calibration-status calibration-status--${status.toLowerCase()}`}
      data-materialization-status={status}
      title={presentation.explanation}
    >
      {presentation.label}
    </span>
  );
}

export function CalibrationMaterializationPanel({
  activeRoutes,
  createMaterialization,
  initialMaterializations,
  loadMaterialization,
  maxPollAttempts = 30,
  pollIntervalMs = 2_000,
  principalRole,
}: CalibrationMaterializationPanelProps) {
  const [materializations, setMaterializations] = useState(
    initialMaterializations,
  );
  const [requestError, setRequestError] = useState<string | null>(null);
  const [creatingRouteKey, setCreatingRouteKey] = useState<string | null>(null);
  const pollAttempts = useRef(new Map<string, number>());
  const idempotencyKeys = useRef(new Map<string, string>());

  useEffect(() => {
    if (
      !Number.isFinite(pollIntervalMs)
      || pollIntervalMs <= 0
      || !Number.isInteger(maxPollAttempts)
      || maxPollAttempts <= 0
    ) {
      return;
    }
    const targets = materializations.filter((item) => {
      const attempts = pollAttempts.current.get(item.materialization_id) ?? 0;
      return !TERMINAL_STATUSES.has(item.status) && attempts < maxPollAttempts;
    });
    if (targets.length === 0) return;

    let cancelled = false;
    const timer = window.setTimeout(async () => {
      for (const target of targets) {
        if (cancelled) return;
        const attempts =
          pollAttempts.current.get(target.materialization_id) ?? 0;
        if (attempts >= maxPollAttempts) continue;
        pollAttempts.current.set(target.materialization_id, attempts + 1);
        try {
          const refreshed = await loadMaterialization(
            target.materialization_id,
          );
          if (!cancelled) {
            setMaterializations((current) =>
              upsertMaterialization(current, refreshed),
            );
          }
        } catch {
          if (!cancelled) {
            setRequestError(
              "校准证据状态读取失败；当前页面不会将其视为 READY。",
            );
            setMaterializations((current) => [...current]);
          }
        }
      }
    }, pollIntervalMs);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    loadMaterialization,
    materializations,
    maxPollAttempts,
    pollIntervalMs,
  ]);

  const materializationByRoute = useMemo(() => {
    const result = new Map<string, AdvancedCalibrationMaterialization>();
    const newestFirst = [...materializations].sort((left, right) =>
      right.created_at.localeCompare(left.created_at),
    );
    for (const item of newestFirst) {
      const key = routeCoordinateKey(item);
      if (!result.has(key)) result.set(key, item);
    }
    return result;
  }, [materializations]);

  async function startMaterialization(
    activeRoute: ActiveModelRouteSummary,
  ): Promise<void> {
    const key = routeKey(activeRoute);
    setCreatingRouteKey(key);
    setRequestError(null);
    const request: CreateAdvancedCalibrationMaterializationRequest = {
      cutoff_cycle: activeRoute.cutoff_cycle,
      route_role: activeRoute.route_role,
      task: activeRoute.task,
    };
    const idempotencyKey =
      idempotencyKeys.current.get(key)
      ?? `advanced-calibration-${crypto.randomUUID()}`;
    idempotencyKeys.current.set(key, idempotencyKey);
    try {
      const response = await createMaterialization(request, idempotencyKey);
      idempotencyKeys.current.delete(key);
      pollAttempts.current.set(response.materialization.materialization_id, 0);
      setMaterializations((current) =>
        upsertMaterialization(current, response.materialization),
      );
    } catch {
      setRequestError(
        "校准证据请求失败；服务端尚未签发 READY 校准证据。",
      );
    } finally {
      setCreatingRouteKey(null);
    }
  }

  return (
    <section
      aria-labelledby="calibration-materialization-title"
      className="calibration-materialization-panel"
    >
      <div className="calibration-materialization-heading">
        <div>
          <h2 id="calibration-materialization-title">Advanced 校准证据</h2>
          <p>
            路由身份由服务端冻结；页面不接收或展示逐样本观测值、预测值与结果 ID。
          </p>
        </div>
        {principalRole === "MEMBER" ? (
          <p className="calibration-readonly-notice">
            仅 ADMIN 可生成校准证据
          </p>
        ) : null}
      </div>

      {requestError ? (
        <p className="calibration-materialization-error" role="alert">
          {requestError}
        </p>
      ) : null}

      {activeRoutes.length === 0 ? (
        <p className="calibration-materialization-empty">
          当前项目没有可用于校准物化的活动模型路由。
        </p>
      ) : (
        <div className="calibration-materialization-table-wrap">
          <table className="calibration-materialization-table">
            <thead>
              <tr>
                <th>任务 / cutoff</th>
                <th>活动路由</th>
                <th>物化状态</th>
                <th>校准证据</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {activeRoutes.map((activeRoute) => {
                const key = routeKey(activeRoute);
                const current = materializationByRoute.get(
                  routeCoordinateKey(activeRoute),
                );
                const supportsMaterialization =
                  supportsCalibrationMaterialization(activeRoute);
                const canCreate =
                  supportsMaterialization
                  && (
                    current === undefined
                    || current.status === "FAILED"
                    || current.status === "STALE"
                  );
                return (
                  <tr key={key}>
                    <td data-label="任务 / cutoff">
                      <strong>{activeRoute.task}</strong>
                      <span>{activeRoute.cutoff_cycle} cycles</span>
                    </td>
                    <td data-label="活动路由">
                      <strong>{activeRoute.route_role}</strong>
                      <span>{activeRoute.artifact_kind}</span>
                      <ReadinessValue
                        label="Model version"
                        value={activeRoute.model_version}
                      />
                      <ReadinessValue
                        label="Artifact ID"
                        value={activeRoute.artifact_id}
                      />
                      <ShaEvidence
                        label="Artifact SHA-256"
                        value={activeRoute.artifact_manifest_sha256}
                      />
                    </td>
                    <td data-label="物化状态">
                      {current ? (
                        <>
                          <StatusBadge status={current.status} />
                          {current.failure_code ? (
                            <span className="calibration-failure-code">
                              {current.failure_code}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <span className="calibration-status calibration-status--missing">
                          尚未生成
                        </span>
                      )}
                    </td>
                    <td data-label="校准证据">
                      {current ? (
                        <>
                          <span>样本数：{current.sample_count}</span>
                          <ReadinessValue
                            label="Data version"
                            value={current.data_version}
                          />
                          <ReadinessValue
                            label="Split version"
                            value={current.split_version}
                          />
                          <ReadinessValue
                            label="Feature version"
                            value={current.feature_version}
                          />
                          <ReadinessValue
                            label="Source registration"
                            value={current.source_registration_id}
                          />
                          <ReadinessValue
                            label="Created at"
                            value={current.created_at}
                          />
                          <ReadinessValue
                            label="Started at"
                            value={current.started_at}
                          />
                          <ReadinessValue
                            label="Completed at"
                            value={current.completed_at}
                          />
                          <ShaEvidence
                            label="Normalizer SHA-256"
                            value={current.normalization_statistics_sha256}
                          />
                          <ShaEvidence
                            label="Source SHA-256"
                            value={current.source_identity_sha256}
                          />
                          <ShaEvidence
                            label="Sample manifest SHA-256"
                            value={current.sample_manifest_sha256}
                          />
                          <ShaEvidence
                            label="Ledger SHA-256"
                            value={current.ledger_head_sha256}
                          />
                        </>
                      ) : (
                        <span>
                          {supportsMaterialization
                            ? "等待 ADMIN 发起身份请求"
                            : "该活动路由不参与 Split Conformal 校准"}
                        </span>
                      )}
                    </td>
                    <td data-label="操作">
                      {principalRole === "ADMIN" && canCreate ? (
                        <button
                          className="calibration-materialization-action"
                          disabled={creatingRouteKey === key}
                          onClick={() => void startMaterialization(activeRoute)}
                          title="仅发送 task、cutoff_cycle 与 route_role；证据来源由服务端解析"
                          type="button"
                        >
                          {creatingRouteKey === key
                            ? "正在提交"
                            : "生成校准证据"}
                        </button>
                      ) : (
                        <span className="calibration-materialization-no-action">
                          {!supportsMaterialization
                            ? "不支持校准物化"
                            : current?.status === "READY"
                              ? "无需操作"
                              : "只读"}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
