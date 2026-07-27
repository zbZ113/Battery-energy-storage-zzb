"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Activity, FileCheck2, RefreshCw, ShieldCheck } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { CreateAgentRunForm } from "@/components/create-agent-run-form";
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
    <AppShell>
      {project ? <header className="page-header"><div><p className="eyebrow">PROJECT WORKSPACE</p><h1>{project.name}</h1><p>选择一个已存在的运行或签发结果进行复核。</p></div></header> : null}
      {!project && !error ? <div className="loading-state"><RefreshCw className="spin" />正在读取项目…</div> : null}
      {error ? <p className="alert alert-error" role="alert">项目不存在，或当前账号没有查看权限。</p> : null}
      {project ? <section className="panel create-run-panel">
        <div className="panel-heading"><Activity /><div><h2>新建多智能体分析</h2><p>输入目标并绑定已冻结数据，系统会生成受约束计划。</p></div></div>
        <CreateAgentRunForm
          onCreate={async (request) => {
            const run = await createAgentRun(request, `agent-run-${crypto.randomUUID()}`);
            router.push(`/agent/runs/${encodeURIComponent(run.run_id)}`);
          }}
          projectId={projectId}
        />
      </section> : null}
      {project ? <div className="action-grid project-secondary-actions">
        <Link className="panel action-card calibration-entry" href={`/projects/${encodeURIComponent(projectId)}/calibration`}>
          <ShieldCheck aria-hidden="true" />
          <h2>校准证据管理</h2>
          <p>查看 active route 的 RUL / SOH 校准物化状态、冻结版本与 SHA-256 证据。</p>
          <span className="text-link">进入校准工作台</span>
        </Link>
        <section className="panel action-card"><Activity /><h2>Agent 运行时间线</h2><p>输入运行 ID，查看后端通过 SSE 签发的计划与步骤事件。</p><ResourceLocator kind="run" /></section>
        <section className="panel action-card"><FileCheck2 /><h2>结果与证据</h2><p>输入结果 ID，查看 ToolResult 原值、来源链及警告。</p><ResourceLocator kind="result" /></section>
      </div> : null}
    </AppShell>
  );
}
