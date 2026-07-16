"use client";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { ProjectOverview } from "@/components/project-overview";
import { ApiError, listProjects } from "@/lib/api-client";
import type { ProjectRecord } from "@/lib/types";

function projectLoadMessage(reason: unknown) {
  return reason instanceof ApiError && reason.status === 401
    ? "登录状态已失效，请重新登录。"
    : "项目列表暂时无法加载，请稍后重试。";
}

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const records = await listProjects();
      setProjects(records);
      setError(null);
    } catch (reason) {
      setError(projectLoadMessage(reason));
    }
  }
  useEffect(() => {
    let active = true;
    listProjects()
      .then((records) => { if (active) setProjects(records); })
      .catch((reason: unknown) => { if (active) setError(projectLoadMessage(reason)); });
    return () => { active = false; };
  }, []);

  return (
    <AppShell>
      <header className="page-header"><div><p className="eyebrow">PROJECTS</p><h1>项目总览</h1><p>进入获授权的电芯研究项目，查看运行任务和可信证据。</p></div></header>
      {projects === null && error === null ? <div className="loading-state"><RefreshCw className="spin" />正在加载项目…</div> : null}
      {error ? <section className="alert alert-error" role="alert"><p>{error}</p><button className="button button-quiet" onClick={() => void load()}>重新加载</button></section> : null}
      {projects !== null ? <ProjectOverview projects={projects} /> : null}
    </AppShell>
  );
}
