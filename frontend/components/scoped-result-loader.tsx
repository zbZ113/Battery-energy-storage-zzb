"use client";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";

import { EvidencePanel } from "@/components/evidence-panel";
import { getAgentRunResult } from "@/lib/api-client";
import type { ToolResult } from "@/lib/types";

type ResultLoader = (runId: string, resultId: string) => Promise<ToolResult>;

export function ScopedResultLoader({
  resultId,
  runId,
  loadResult = getAgentRunResult,
}: {
  resultId: string;
  runId: string | null;
  loadResult?: ResultLoader;
}) {
  if (runId === null) {
    return (
      <p className="alert alert-error" role="alert">
        缺少运行 ID，无法核验你是否有权查看这个结果。请从 Agent 运行时间线进入。
      </p>
    );
  }
  return <AuthorizedResultLoader loadResult={loadResult} resultId={resultId} runId={runId} />;
}

function AuthorizedResultLoader({
  resultId,
  runId,
  loadResult,
}: {
  resultId: string;
  runId: string;
  loadResult: ResultLoader;
}) {
  const [result, setResult] = useState<ToolResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    loadResult(runId, resultId)
      .then((value) => { if (active) setResult(value); })
      .catch(() => { if (active) setError("无法读取该结果，请检查运行 ID、结果 ID 和项目权限。"); });
    return () => { active = false; };
  }, [loadResult, resultId, runId]);

  if (error) return <p className="alert alert-error" role="alert">{error}</p>;
  if (result === null) {
    return <div className="loading-state"><RefreshCw className="spin" />正在核验结果…</div>;
  }
  return <EvidencePanel result={result} />;
}
