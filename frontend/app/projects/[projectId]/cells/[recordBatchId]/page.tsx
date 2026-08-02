"use client";

import { use } from "react";
import { Microscope } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { RunAnalysisLoader } from "@/components/run-analysis-loader";

type SearchParams = {
  run_id?: string | string[];
};

function runId(value: string | string[] | undefined): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export default function SingleCellPage({
  params,
  searchParams,
}: {
  params: Promise<{ projectId: string; recordBatchId: string }>;
  searchParams: Promise<SearchParams>;
}) {
  const { projectId, recordBatchId } = use(params);
  const query = use(searchParams);
  const selectedRunId = runId(query.run_id);

  return (
    <AppShell workflowStage="results">
      <header className="page-header">
        <div>
          <p className="eyebrow">VERIFIED SINGLE-CELL ANALYSIS</p>
          <h1>单电芯诊断</h1>
          <p>项目 <code>{projectId}</code> · record batch <code>{recordBatchId}</code></p>
        </div>
        <Microscope aria-hidden="true" className="header-icon" />
      </header>
      {selectedRunId ? (
        <RunAnalysisLoader
          projectId={projectId}
          recordBatchId={recordBatchId}
          runId={selectedRunId}
        />
      ) : (
        <p className="alert alert-error" role="alert">缺少本次分析的 run_id，无法发现可信结果目录。</p>
      )}
    </AppShell>
  );
}
