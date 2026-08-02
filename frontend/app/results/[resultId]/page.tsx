"use client";

import { use } from "react";
import { FileSearch } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { ScopedResultLoader } from "@/components/scoped-result-loader";

export default function ResultPage({
  params,
  searchParams,
}: {
  params: Promise<{ resultId: string }>;
  searchParams: Promise<{ run_id?: string | string[] }>;
}) {
  const { resultId } = use(params);
  const rawRunId = use(searchParams).run_id;
  const runId = typeof rawRunId === "string" && rawRunId.trim() ? rawRunId.trim() : null;

  return (
    <AppShell workflowStage="results">
      <header className="page-header"><div><p className="eyebrow">VERIFIED RESULT</p><h1>结果与证据</h1><p>结果 ID：<code>{resultId}</code></p></div><FileSearch aria-hidden="true" className="header-icon" /></header>
      <ScopedResultLoader resultId={resultId} runId={runId} />
    </AppShell>
  );
}
