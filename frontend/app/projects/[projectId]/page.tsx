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
  RefreshCw,
  ShieldCheck,
} from "lucide-react";

import { AnalysisWorkspace } from "@/components/analysis-workspace";
import { AppShell } from "@/components/app-shell";
import { CreateAgentRunForm } from "@/components/create-agent-run-form";
import { NineStepTracker } from "@/components/nine-step-tracker";
import { ResourceLocator } from "@/components/resource-locator";
import { createAgentRun, getProject } from "@/lib/api-client";
import type { ProjectRecord } from "@/lib/types";

export default function ProjectPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = use(params);
  const router = useRouter();
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => { getProject(projectId).then(setProject).catch(() => setError(true)); }, [projectId]);

  return (
    <AppShell workflowStage="source">
      {project ? <header className="page-header workspace-header"><div><p className="eyebrow">ANALYSIS WORKSPACE</p><h1>{project.name}</h1><p>从可信数据输入开始，在同一工作台观察结果与运行状态。</p></div><Link className="text-link" href="/projects">返回项目目录</Link></header> : null}
      {!project && !error ? <div className="loading-state"><RefreshCw className="spin" />正在读取项目…</div> : null}
      {error ? <p className="alert alert-error" role="alert">项目不存在，或当前账号没有查看权限。</p> : null}
      {project ? (
        <>
          <section aria-label="分析输入" className="workspace-inputs">
            <div className="input-group">
              <span className="workspace-label">数据来源</span>
              <div className="source-options">
                <button className="source-option source-option-disabled" disabled type="button"><strong>内置样例</strong><span>等待项目数据目录</span></button>
                <button className="source-option source-option-disabled" disabled type="button"><FileUp aria-hidden="true" /><strong>上传数据</strong><span>上传能力尚未接入</span></button>
              </div>
            </div>
            <div className="input-group scope-summary">
              <span className="workspace-label">分析范围</span>
              <dl><div><dt>冻结数据</dt><dd>等待目录 API</dd></div><div><dt>电芯 / cutoff</dt><dd>选择数据后可用</dd></div></dl>
            </div>
          </section>
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
                <section className="workspace-panel compatibility-panel">
                  <div className="panel-heading"><ShieldCheck aria-hidden="true" /><div><h2>高级定位与校准证据</h2><p>目录 API 接入前，保留现有受项目权限约束的定位入口。</p></div></div>
                  <div className="compatibility-grid">
                    <div><h3>运行 ID</h3><ResourceLocator kind="run" /></div>
                    <div><h3>结果 ID</h3><ResourceLocator kind="result" /></div>
                  </div>
                  <Link className="text-link" href={`/projects/${encodeURIComponent(projectId)}/calibration`}>进入校准工作台</Link>
                </section>
              </>
            )}
            operations={(
              <>
                <section className="workspace-panel run-launcher">
                  <div className="panel-heading"><Activity aria-hidden="true" /><div><h2>新运行</h2><p>每次启动创建一个独立、可审计的运行。</p></div></div>
                  <CreateAgentRunForm
                    onCreate={async (request) => {
                      const run = await createAgentRun(request, `agent-run-${crypto.randomUUID()}`);
                      router.push(`/agent/runs/${encodeURIComponent(run.run_id)}`);
                    }}
                    projectId={projectId}
                  />
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
