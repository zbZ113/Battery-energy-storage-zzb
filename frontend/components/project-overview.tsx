import Link from "next/link";
import { ArrowUpRight, BatteryCharging, FolderKanban } from "lucide-react";

import type { ProjectRecord } from "@/lib/types";

export function ProjectOverview({ projects }: { projects: ProjectRecord[] }) {
  if (projects.length === 0) {
    return (
      <section className="empty-state">
        <span className="empty-icon"><FolderKanban aria-hidden="true" /></span>
        <h2>还没有可查看的项目</h2>
        <p>联系管理员创建项目，或请管理员为你分配已有项目权限。</p>
      </section>
    );
  }

  return (
    <section className="card-grid" aria-label="项目列表">
      {projects.map((project) => (
        <Link className="project-card" href={`/projects/${project.project_id}`} key={project.project_id}>
          <span className="project-icon"><BatteryCharging aria-hidden="true" /></span>
          <span className="project-card-copy">
            <strong>{project.name}</strong>
            <small>创建于 {new Date(project.created_at).toLocaleDateString("zh-CN")}</small>
          </span>
          <ArrowUpRight aria-hidden="true" className="project-arrow" />
        </Link>
      ))}
    </section>
  );
}
