"use client";

import { use } from "react";
import { Microscope } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import {
  SingleCellAnalysis,
  type SingleCellResultIds,
} from "@/components/single-cell-analysis";

type SearchParams = {
  report_result_id?: string | string[];
  rul_conformal_result_id?: string | string[];
  rul_result_id?: string | string[];
  soh_conformal_result_id?: string | string[];
  soh_result_id?: string | string[];
};

function resultId(value: string | string[] | undefined): string | null {
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
  const resultIds: SingleCellResultIds = {
    reportResultId: resultId(query.report_result_id),
    rulConformalResultId: resultId(query.rul_conformal_result_id),
    rulResultId: resultId(query.rul_result_id),
    sohConformalResultId: resultId(query.soh_conformal_result_id),
    sohResultId: resultId(query.soh_result_id),
  };

  return (
    <AppShell>
      <header className="page-header">
        <div>
          <p className="eyebrow">VERIFIED SINGLE-CELL ANALYSIS</p>
          <h1>单电芯诊断</h1>
          <p>项目 <code>{projectId}</code> · record batch <code>{recordBatchId}</code></p>
        </div>
        <Microscope aria-hidden="true" className="header-icon" />
      </header>
      <SingleCellAnalysis
        projectId={projectId}
        recordBatchId={recordBatchId}
        resultIds={resultIds}
      />
    </AppShell>
  );
}
