"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  ChartNoAxesCombined,
  Download,
  FileCheck2,
  FileUp,
  History,
  RefreshCw,
} from "lucide-react";

import { AdvancedAnalysisLauncher } from "@/components/advanced-analysis-launcher";
import { AnalysisWorkspace } from "@/components/analysis-workspace";
import { AppShell } from "@/components/app-shell";
import { NineStepTracker } from "@/components/nine-step-tracker";
import {
  getAnalysisInputs,
  getProject,
  listProjectAgentRuns,
} from "@/lib/api-client";
import { decodeAnalysisInputs, isFixedAdvancedRun } from "@/lib/advanced-analysis-contract";
import type { AgentRunRecord, AnalysisInputsRecord, ProjectRecord } from "@/lib/types";

export default function ProjectPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = use(params);
  return <ProjectWorkspace projectId={projectId} />;
}

export function ProjectWorkspace({ projectId }: { projectId: string }) {
  return <ProjectWorkspaceSession key={projectId} projectId={projectId} />;
}

function ProjectWorkspaceSession({ projectId }: { projectId: string }) {
  const router = useRouter();
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [inputs, setInputs] = useState<AnalysisInputsRecord | null>(null);
  const [runs, setRuns] = useState<AgentRunRecord[]>([]);
  const [projectError, setProjectError] = useState(false);
  const [inputsError, setInputsError] = useState(false);
  const [runsError, setRunsError] = useState(false);
  useEffect(() => {
    let active = true;
    void getProject(projectId)
      .then((record) => { if (active) setProject(record); })
      .catch(() => { if (active) setProjectError(true); });
    void getAnalysisInputs(projectId)
      .then((record) => decodeAnalysisInputs(record, projectId))
      .then((record) => { if (active) setInputs(record); })
      .catch(() => { if (active) setInputsError(true); });
    void listProjectAgentRuns(projectId)
      .then((records) => records.filter((record) => (
        isFixedAdvancedRun(record) && record.project_id === projectId
      )))
      .then((records) => { if (active) setRuns(records); })
      .catch(() => { if (active) setRunsError(true); });
    return () => { active = false; };
  }, [projectId]);

  const firstDataset = inputs?.datasets[0] ?? null;
  const firstBatch = inputs?.batches[0] ?? null;
  const sampleLabel = inputs
    ? firstDataset?.name ?? "暂无可用冻结数据"
    : "正在读取冻结目录";

  return (
    <AppShell workflowStage="source">
      {project ? <header className="page-header workspace-header"><div><p className="eyebrow">ANALYSIS WORKSPACE</p><h1>{project.name}</h1><p>从可信数据输入开始，在同一工作台观察结果与运行状态。</p></div><Link className="text-link" href="/projects">返回项目目录</Link></header> : null}
      {!project && !projectError ? <div className="loading-state"><RefreshCw className="spin" />正在读取项目…</div> : null}
      {projectError ? <p className="alert alert-error" role="alert">项目不存在，或当前账号没有查看权限。</p> : null}
      {project ? (
        <>
          <section aria-label="分析输入" className="workspace-inputs">
            <div className="input-group">
              <span className="workspace-label">数据来源</span>
              <div className="source-options">
                <button
                  className={`source-option ${firstDataset ? "source-option-active" : "source-option-disabled"}`}
                  disabled={!firstDataset}
                  type="button"
                >
                  <strong>内置样例</strong><span>{sampleLabel}</span>
                </button>
                <button className="source-option source-option-disabled" disabled type="button"><FileUp aria-hidden="true" /><strong>上传数据</strong><span>上传能力尚未接入</span></button>
              </div>
            </div>
            <div className="input-group scope-summary">
              <span className="workspace-label">分析范围</span>
              <dl>
                <div><dt>冻结数据</dt><dd>{firstDataset ? `${firstDataset.name} · ${firstDataset.data_version}` : "正在读取"}</dd></div>
                <div><dt>电芯 / cutoff</dt><dd>{firstBatch ? `${firstBatch.cell_id} / ${firstBatch.cutoff_cycle} cycles` : "选择数据后可用"}</dd></div>
              </dl>
            </div>
          </section>
          {inputsError ? <p className="alert alert-error" role="alert">冻结分析输入读取失败，已停止创建运行。</p> : null}
          <AnalysisWorkspace
            analysis={(
              <>
                <section className="workspace-panel result-surface">
                  <div aria-label="结果视图" className="result-tabs"><strong>SOH 轨迹</strong><span>RUL</span><span>Conformal</span><span>审计报告</span></div>
                  <div className="trusted-empty" role="status">
                    <ChartNoAxesCombined aria-hidden="true" />
                    <div><strong>等待 ToolResult</strong><p>启动九步 Agent 并由服务端签发结果后，此处才绘制图表或显示业务数值。</p></div>
                  </div>
                </section>
                <section className="workspace-panel recent-runs">
                  <div className="panel-heading"><History aria-hidden="true" /><div><h2>最近运行</h2><p>从项目目录继续查看服务端已记录的分析。</p></div></div>
                  {runsError ? <p className="alert alert-error" role="alert">最近运行读取失败。</p> : null}
                  {!runsError && runs.length === 0 ? <p className="muted">还没有运行记录。</p> : null}
                  <ul className="recent-run-list">
                    {runs.map((run) => (
                      <li key={run.run_id}>
                        <Link href={`/agent/runs/${encodeURIComponent(run.run_id)}`}>
                          <span>{run.run_id}</span><strong>{run.status}</strong>
                        </Link>
                      </li>
                    ))}
                  </ul>
                  <Link className="text-link" href={`/projects/${encodeURIComponent(projectId)}/calibration`}>进入校准工作台</Link>
                </section>
              </>
            )}
            operations={(
              <>
                <section className="workspace-panel run-launcher">
                  <div className="panel-heading"><Activity aria-hidden="true" /><div><h2>新运行</h2><p>每次启动创建一个独立、可审计的运行。</p></div></div>
                  {inputs ? (
                    <AdvancedAnalysisLauncher
                      inputs={inputs}
                      onCreated={(run) => router.push(`/agent/runs/${encodeURIComponent(run.run_id)}`)}
                      projectId={projectId}
                    />
                  ) : null}
                  {!inputs && !inputsError ? <p className="muted" role="status">正在读取冻结分析输入…</p> : null}
                </section>
                <section className="workspace-panel run-preview">
                  <div className="panel-heading"><Activity aria-hidden="true" /><div><h2>九步实时进度</h2><p>启动后由 run record 与 SSE 事件逐步更新。</p></div></div>
                  <NineStepTracker events={[]} />
                </section>
                <section className="workspace-panel evidence-download">
                  <div><FileCheck2 aria-hidden="true" /><strong>证据抽屉</strong><p>等待 ToolResult 身份、来源和警告。</p></div>
                  <div><Download aria-hidden="true" /><strong>下载</strong><p>待服务端提供导出。</p></div>
                </section>
              </>
            )}
          />
        </>
      ) : null}
    </AppShell>
  );
}
