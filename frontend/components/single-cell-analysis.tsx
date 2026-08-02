"use client";

import { useEffect, useState } from "react";
import {
  BatteryCharging,
  ChartNoAxesCombined,
  CircleAlert,
  Download,
  FileText,
  LoaderCircle,
  Route,
  ShieldCheck,
} from "lucide-react";

import { EvidencePanel } from "@/components/evidence-panel";
import {
  RESULT_LABELS,
  isSafeToolResult,
  parseAnalysis,
  type LoadedResults,
  type ParsedAnalysis,
  type ProjectResultLoader,
  type ResultKey,
  type SingleCellResultIds,
} from "@/components/single-cell-analysis-contract";
import { SohTrajectoryChart } from "@/components/soh-trajectory-chart";
import { getProjectResult } from "@/lib/api-client";
import type { ToolResult } from "@/lib/types";

export type { SingleCellResultIds } from "@/components/single-cell-analysis-contract";

function EvidenceStack({ results }: { results: LoadedResults }) {
  const entries = (Object.entries(results) as [ResultKey, ToolResult][])
    .filter(([, value]) => value !== undefined && isSafeToolResult(value));
  if (!entries.length) return null;
  return (
    <section className="evidence-stack" aria-labelledby="single-cell-evidence-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow"><ShieldCheck aria-hidden="true" size={16} />AUDIT EVIDENCE</p>
          <h2 id="single-cell-evidence-heading">逐项 ToolResult 证据</h2>
        </div>
        <p>每项结果保留服务端原值、来源链、哈希和警告。</p>
      </div>
      {entries.map(([key, result]) => (
        <details className="evidence-disclosure" key={key}>
          <summary>{RESULT_LABELS[key]} · <code>{result.result_id}</code></summary>
          <div aria-label={`${RESULT_LABELS[key]} ToolResult`}>
            <EvidencePanel result={result} />
          </div>
        </details>
      ))}
    </section>
  );
}

function WaitingState({
  missingKeys,
  runTerminal,
}: {
  missingKeys: ResultKey[];
  runTerminal: boolean;
}) {
  return (
    <section className="result-waiting" aria-live="polite" role="status">
      {runTerminal ? <CircleAlert aria-hidden="true" /> : <LoaderCircle aria-hidden="true" />}
      <div>
        <h2>{runTerminal ? "运行已结束，部分结果未签发" : "等待服务端签发分析结果"}</h2>
        <p>{runTerminal ? "未签发" : "尚缺"} {missingKeys.length} 项 ToolResult</p>
        <ul>
          {missingKeys.map((key) => <li key={key}>{RESULT_LABELS[key]}</li>)}
        </ul>
      </div>
    </section>
  );
}

function ExportStatus() {
  return (
    <section aria-label="下载状态" className="export-status" role="status">
      <Download aria-hidden="true" />
      <div>
        <h2>下载</h2>
        <p>待服务端提供导出；页面不会在浏览器中拼装业务报告。</p>
      </div>
    </section>
  );
}

function Warnings({ results }: { results: LoadedResults }) {
  const warnings = (Object.entries(results) as [ResultKey, ToolResult][])
    .flatMap(([key, result]) => result.warnings.map((warning) => ({
      key,
      resultId: result.result_id,
      warning,
    })));
  if (!warnings.length) return null;
  return (
    <section className="panel analysis-warning-panel">
      <div className="panel-heading">
        <CircleAlert aria-hidden="true" />
        <div><h2>服务端限制与警告</h2><p>按 ToolResult 逐条展示，不隐藏、不改写。</p></div>
      </div>
      <ul>
        {warnings.map((item, index) => (
          <li key={`${item.key}:${item.resultId}:${index}`}>
            <strong>{RESULT_LABELS[item.key]}：</strong>{item.warning}
          </li>
        ))}
      </ul>
    </section>
  );
}

function SingleCellAnalysisLoader({
  loadResult,
  projectId,
  recordBatchId,
  resultIds,
  runTerminal,
  showEvidence,
}: {
  loadResult: ProjectResultLoader;
  projectId: string;
  recordBatchId: string;
  resultIds: SingleCellResultIds;
  runTerminal: boolean;
  showEvidence: boolean;
}) {
  const [results, setResults] = useState<LoadedResults>({});
  const [loadErrors, setLoadErrors] = useState<ResultKey[]>([]);
  const {
    reportResultId,
    rulConformalResultId,
    rulResultId,
    sohConformalResultId,
    sohResultId,
  } = resultIds;
  const requestedResults = (Object.entries(resultIds) as [ResultKey, string | null][])
    .filter((entry): entry is [ResultKey, string] => entry[1] !== null);
  const [loading, setLoading] = useState(requestedResults.length > 0);
  const missingKeys = (Object.entries(resultIds) as [ResultKey, string | null][])
    .filter(([, value]) => value === null)
    .map(([key]) => key);

  useEffect(() => {
    let active = true;
    const requested = (
      [
        ["reportResultId", reportResultId],
        ["rulConformalResultId", rulConformalResultId],
        ["rulResultId", rulResultId],
        ["sohConformalResultId", sohConformalResultId],
        ["sohResultId", sohResultId],
      ] satisfies [ResultKey, string | null][]
    ).filter((entry): entry is [ResultKey, string] => entry[1] !== null);
    if (!requested.length) {
      return () => { active = false; };
    }
    Promise.allSettled(
      requested.map(async ([key, resultId]) => ({
        key,
        result: await loadResult(projectId, resultId),
      })),
    ).then((settled) => {
      if (!active) return;
      const nextResults: LoadedResults = {};
      const nextErrors: ResultKey[] = [];
      settled.forEach((item, index) => {
        const key = requested[index][0];
        if (item.status === "fulfilled") {
          nextResults[item.value.key] = item.value.result;
        } else {
          nextErrors.push(key);
        }
      });
      setResults(nextResults);
      setLoadErrors(nextErrors);
      setLoading(false);
    });
    return () => { active = false; };
  }, [
    loadResult,
    projectId,
    reportResultId,
    rulConformalResultId,
    rulResultId,
    sohConformalResultId,
    sohResultId,
  ]);

  let parsed: ParsedAnalysis | null = null;
  let validationError: string | null = null;
  if (!loading && Object.keys(results).length) {
    try {
      parsed = parseAnalysis(results, resultIds, recordBatchId);
    } catch {
      validationError = "身份或证据链不一致，页面已停止展示业务数值。";
    }
  }

  if (!requestedResults.length) {
    return (
      <div className="single-cell-analysis">
        <WaitingState missingKeys={missingKeys} runTerminal={runTerminal} />
        <ExportStatus />
      </div>
    );
  }

  return (
    <div className="single-cell-analysis">
      {missingKeys.length ? <WaitingState missingKeys={missingKeys} runTerminal={runTerminal} /> : null}
      {loading ? (
        <div className="loading-state" role="status">
          <LoaderCircle className="spin" aria-hidden="true" />正在核验项目 ToolResult…
        </div>
      ) : null}
      {loadErrors.length ? (
        <p className="alert alert-error" role="alert">
          {loadErrors.map((key) => RESULT_LABELS[key]).join("、")}读取失败；其余已签发证据仍保留展示。
        </p>
      ) : null}
      {validationError ? (
        <p className="alert alert-error" role="alert">
          分析结果暂不可用：{validationError}
        </p>
      ) : null}

      {parsed && (parsed.rul || parsed.rulConformal) ? (
        <section className="analysis-grid" aria-labelledby="rul-analysis-heading">
          <article className="panel analysis-callout">
            <div className="panel-heading">
              <BatteryCharging aria-hidden="true" />
              <div>
                <h2 id="rul-analysis-heading">MATR 官方 cycle life（非统一 EOL80）</h2>
                <p>目标定义和所有数值均来自项目 ToolResult。</p>
              </div>
            </div>
            <dl className="metric-list">
              {parsed.rul ? (
                <>
                  <div><dt>预测 cycle</dt><dd>{parsed.rul.predictedCycle}</dd></div>
                  <div><dt>服务端签发 remaining cycles</dt><dd>{parsed.rul.derivedRemainingCycles}</dd></div>
                  <div><dt>模型路由</dt><dd>{parsed.rul.routeRole}</dd></div>
                  <div><dt>模型制品</dt><dd>{parsed.rul.artifactKind}</dd></div>
                </>
              ) : null}
            </dl>
          </article>
          {parsed.rulConformal ? (
            <article className="panel">
              <div className="panel-heading">
                <Route aria-hidden="true" />
                <div><h2>route-specific Split Conformal</h2><p>prediction interval，不是参数 confidence interval。</p></div>
              </div>
              <dl className="metric-list">
                <div><dt>coverage_target</dt><dd>{parsed.rulConformal.coverageTarget}</dd></div>
                <div><dt>coverage route interval center</dt><dd>{parsed.rulConformal.pointPredictionCycle}</dd></div>
                <div><dt>coverage route remaining center</dt><dd>{parsed.rulConformal.derivedRulCycle}</dd></div>
                <div><dt>cycle interval</dt><dd>[{parsed.rulConformal.lowerCycle}, {parsed.rulConformal.upperCycle}]</dd></div>
                <div><dt>remaining-cycle interval</dt><dd>[{parsed.rulConformal.lowerRulCycle}, {parsed.rulConformal.upperRulCycle}]</dd></div>
                <div><dt>区间路由</dt><dd>{parsed.rulConformal.routeRole}</dd></div>
              </dl>
            </article>
          ) : null}
        </section>
      ) : null}

      {parsed && (parsed.soh || parsed.sohConformal) ? (
        <section className="panel trajectory-shell" aria-labelledby="soh-analysis-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow"><ChartNoAxesCombined aria-hidden="true" size={16} />FINITE HORIZON</p>
              <h2 id="soh-analysis-heading">SOH trajectory</h2>
            </div>
            <div className="trajectory-status">
              {parsed.soh ? <span className="status-pill">{parsed.soh.routeRole}</span> : null}
              {parsed.sohConformal ? <span>finite_horizon_only = true</span> : null}
              {parsed.sohConformal ? <span>{parsed.sohConformal.coverageScope}</span> : null}
            </div>
          </div>
          {parsed.soh && parsed.sohConformal ? (
            <SohTrajectoryChart
              coverageTarget={parsed.sohConformal.coverageTarget}
              lowerSoh={parsed.sohConformal.lowerSoh}
              predictedSoh={parsed.soh.predictedSoh}
              predictionCycles={parsed.soh.predictionCycles}
              upperSoh={parsed.sohConformal.upperSoh}
            />
          ) : (
            <p className="muted">需要 SOH point ToolResult 与匹配的 simultaneous band 才能绘图。</p>
          )}
        </section>
      ) : null}

      {parsed?.rul ? (
        <section className="panel provenance-summary">
          <div className="panel-heading">
            <ShieldCheck aria-hidden="true" />
            <div><h2>冻结模型与数据来源</h2><p>展示 active route 解析后随 ToolResult 签发的版本和 SHA。</p></div>
          </div>
          <dl className="provenance-grid">
            <div><dt>route</dt><dd>{parsed.rul.routeRole}</dd></div>
            <div><dt>artifact</dt><dd>{parsed.rul.artifactId}</dd></div>
            <div><dt>model</dt><dd>{results.rulResultId?.model_version}</dd></div>
            <div><dt>data</dt><dd>{results.rulResultId?.data_version}</dd></div>
            <div><dt>feature</dt><dd>{results.rulResultId?.feature_version}</dd></div>
            <div><dt>split</dt><dd>{parsed.rul.splitVersion}</dd></div>
            <div><dt>artifact SHA-256</dt><dd><code>{parsed.rul.artifactManifestSha256}</code></dd></div>
            <div><dt>normalizer SHA-256</dt><dd><code>{parsed.rul.normalizationStatisticsSha256}</code></dd></div>
          </dl>
        </section>
      ) : null}

      {parsed?.report ? (
        <section className="panel report-panel" aria-labelledby="audited-report-heading">
          <div className="panel-heading">
            <FileText aria-hidden="true" />
            <div><h2 id="audited-report-heading">服务端审计报告 Markdown</h2><p>原文显示并由 React 转义，不在浏览器重写结论。</p></div>
          </div>
          <pre className="report-markdown">{parsed.report.markdown}</pre>
        </section>
      ) : null}

      {validationError ? null : <Warnings results={results} />}
      {showEvidence ? <EvidenceStack results={results} /> : null}
      <ExportStatus />
    </div>
  );
}

export function SingleCellAnalysis({
  loadResult = getProjectResult,
  projectId,
  recordBatchId,
  resultIds,
  runTerminal = false,
  showEvidence = true,
}: {
  loadResult?: ProjectResultLoader;
  projectId: string;
  recordBatchId: string;
  resultIds: SingleCellResultIds;
  runTerminal?: boolean;
  showEvidence?: boolean;
}) {
  const requestSignature = JSON.stringify([
    projectId,
    recordBatchId,
    resultIds.reportResultId,
    resultIds.rulConformalResultId,
    resultIds.rulResultId,
    resultIds.sohConformalResultId,
    resultIds.sohResultId,
    runTerminal,
    showEvidence,
  ]);
  return (
    <SingleCellAnalysisLoader
      key={requestSignature}
      loadResult={loadResult}
      projectId={projectId}
      recordBatchId={recordBatchId}
      resultIds={resultIds}
      runTerminal={runTerminal}
      showEvidence={showEvidence}
    />
  );
}
