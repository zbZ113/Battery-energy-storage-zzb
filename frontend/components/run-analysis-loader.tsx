"use client";

import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";

import { RunEvidenceCatalog } from "@/components/run-evidence-catalog";
import { SingleCellAnalysis, type SingleCellResultIds } from "@/components/single-cell-analysis";
import {
  decodeAgentRun,
  decodeAgentRunResults,
  resultIdsFromRunResults,
} from "@/lib/advanced-analysis-contract";
import { getAgentRun, listAgentRunResults } from "@/lib/api-client";
import type { AgentRunRecord, AgentRunResultRecord } from "@/lib/types";

type RunLoader = (runId: string) => Promise<AgentRunRecord>;
type ResultCatalogLoader = (runId: string) => Promise<AgentRunResultRecord[]>;
interface LoadedRunAnalysis {
  resultIds: SingleCellResultIds;
  results: AgentRunResultRecord[];
  runCompleted: boolean;
  runTerminal: boolean;
}

interface RunAnalysisLoaderProps {
  loadResults?: ResultCatalogLoader;
  loadRun?: RunLoader;
  projectId: string;
  recordBatchId: string;
  runId: string;
}

export function RunAnalysisLoader(props: RunAnalysisLoaderProps) {
  const requestSignature = JSON.stringify([
    props.projectId,
    props.recordBatchId,
    props.runId,
  ]);
  return <RunAnalysisSession key={requestSignature} {...props} />;
}

function RunAnalysisSession({
  loadResults = listAgentRunResults,
  loadRun = getAgentRun,
  projectId,
  recordBatchId,
  runId,
}: RunAnalysisLoaderProps) {
  const [loaded, setLoaded] = useState<LoadedRunAnalysis | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    void Promise.all([loadRun(runId), loadResults(runId)])
      .then(([rawRun, rawResults]) => {
        const run = decodeAgentRun(rawRun, projectId, recordBatchId);
        const results = decodeAgentRunResults(rawResults, run);
        const resultIds = resultIdsFromRunResults(results);
        const runCompleted = run.status === "COMPLETED";
        const runTerminal = ["COMPLETED", "FAILED", "CANCELLED"].includes(run.status);
        if (active) setLoaded({ resultIds, results, runCompleted, runTerminal });
      })
      .catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, [loadResults, loadRun, projectId, recordBatchId, runId]);

  if (error) {
    return (
      <p className="alert alert-error" role="alert">
        结果目录无法通过身份核验，已停止展示 RUL、SOH、Conformal 和报告。
      </p>
    );
  }
  if (!loaded) {
    return <div className="loading-state" role="status"><LoaderCircle className="spin" />正在核验本次运行结果目录…</div>;
  }
  return (
    <>
      <SingleCellAnalysis
        projectId={projectId}
        recordBatchId={recordBatchId}
        resultIds={loaded.resultIds}
        toolResultIds={loaded.results.map((item) => item.result.result_id)}
        runCompleted={loaded.runCompleted}
        runId={runId}
        runTerminal={loaded.runTerminal}
        showEvidence={false}
      />
      <RunEvidenceCatalog results={loaded.results} />
    </>
  );
}
