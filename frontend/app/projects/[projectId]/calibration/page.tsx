"use client";

import Link from "next/link";
import { use, useCallback, useEffect, useState } from "react";
import { ArrowLeft, RefreshCw, ShieldCheck } from "lucide-react";

import {
  CalibrationMaterializationPanel,
} from "@/components/calibration-materialization-panel";
import type {
  ActiveModelRouteSummary,
  AdvancedCalibrationMaterialization,
  CreateAdvancedCalibrationMaterializationRequest,
} from "@/components/calibration-materialization-contract";
import { AppShell } from "@/components/app-shell";
import {
  createAdvancedCalibrationMaterialization,
  getAdvancedCalibrationMaterialization,
  getCurrentPrincipal,
  listActiveModelRoutes,
  listAdvancedCalibrationMaterializations,
} from "@/lib/api-client";
import type { UserRole } from "@/lib/types";

interface CalibrationPageState {
  activeRoutes: ActiveModelRouteSummary[];
  materializations: AdvancedCalibrationMaterialization[];
  principalRole: Extract<UserRole, "ADMIN" | "MEMBER">;
}

export default function CalibrationPage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = use(params);
  const [state, setState] = useState<CalibrationPageState | null>(null);
  const [loadError, setLoadError] = useState(false);

  const loadPage = useCallback(async () => {
    setLoadError(false);
    try {
      setState(await fetchCalibrationPageState(projectId));
    } catch {
      setState(null);
      setLoadError(true);
    }
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;
    void fetchCalibrationPageState(projectId)
      .then((next) => {
        if (!cancelled) setState(next);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  return (
    <AppShell>
      <header className="page-header calibration-page-header">
        <div>
          <p className="eyebrow">
            <ShieldCheck aria-hidden="true" size={15} />
            TRUSTED CALIBRATION
          </p>
          <h1>校准证据管理</h1>
          <p>
            为当前 active route 准备服务端验证的 RUL / SOH 校准样本，并审计冻结版本与
            SHA-256。
          </p>
        </div>
        <Link
          className="button calibration-back-link"
          href={`/projects/${encodeURIComponent(projectId)}`}
        >
          <ArrowLeft aria-hidden="true" size={16} />
          返回项目
        </Link>
      </header>

      {!state && !loadError ? (
        <div className="loading-state">
          <RefreshCw className="spin" aria-hidden="true" />
          正在核验 active route 与校准证据…
        </div>
      ) : null}

      {loadError ? (
        <div className="alert alert-error" role="alert">
          <p>校准证据暂时无法读取；页面不会把未知状态显示为 READY。</p>
          <button
            className="button button-quiet"
            onClick={() => void loadPage()}
            type="button"
          >
            重新核验
          </button>
        </div>
      ) : null}

      {state ? (
        <CalibrationMaterializationPanel
          activeRoutes={state.activeRoutes}
          createMaterialization={(
            request: CreateAdvancedCalibrationMaterializationRequest,
            idempotencyKey: string,
          ) =>
            createAdvancedCalibrationMaterialization(
              projectId,
              request,
              idempotencyKey,
            )
          }
          initialMaterializations={state.materializations}
          loadMaterialization={(materializationId: string) =>
            getAdvancedCalibrationMaterialization(
              projectId,
              materializationId,
            )
          }
          principalRole={state.principalRole}
        />
      ) : null}
    </AppShell>
  );
}

async function fetchCalibrationPageState(
  projectId: string,
): Promise<CalibrationPageState> {
  const [principal, activeRoutes, materializations] = await Promise.all([
    getCurrentPrincipal(),
    listActiveModelRoutes(projectId),
    listAdvancedCalibrationMaterializations(projectId),
  ]);
  if (principal.role !== "ADMIN" && principal.role !== "MEMBER") {
    throw new Error("calibration management requires an operator role");
  }
  return {
    activeRoutes,
    materializations,
    principalRole: principal.role,
  };
}
