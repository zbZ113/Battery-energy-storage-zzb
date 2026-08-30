"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Activity, BatteryCharging, BookOpenCheck, Database, LogOut, Menu, ShieldCheck, X } from "lucide-react";

import { WorkflowRail, type WorkflowStage } from "@/components/workflow-rail";
import { logout } from "@/lib/api-client";

export function AppShell({
  children,
  onLogout,
  workflowStage,
}: {
  children: React.ReactNode;
  onLogout?: () => Promise<void>;
  workflowStage?: WorkflowStage;
}) {
  const router = useRouter();
  const [logoutPending, setLogoutPending] = useState(false);
  const [logoutError, setLogoutError] = useState(false);
  const [navigationOpen, setNavigationOpen] = useState(false);

  async function performLogout() {
    setLogoutPending(true);
    setLogoutError(false);
    try {
      await (onLogout ?? logout)();
      if (onLogout === undefined) {
        router.replace("/login");
        router.refresh();
      }
    } catch {
      setLogoutError(true);
      setLogoutPending(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <Link className="brand" href="/projects">
          <span className="brand-mark"><BatteryCharging aria-hidden="true" /></span>
          <span><strong>Hiro</strong><small>可信电芯智能诊断</small></span>
        </Link>
        <button
          aria-controls="main-navigation"
          aria-expanded={navigationOpen}
          aria-label={navigationOpen ? "收起主导航" : "展开主导航"}
          className="icon-button navigation-toggle"
          onClick={() => setNavigationOpen((open) => !open)}
          title={navigationOpen ? "收起主导航" : "展开主导航"}
          type="button"
        >
          {navigationOpen ? <X aria-hidden="true" size={18} /> : <Menu aria-hidden="true" size={18} />}
        </button>
        <nav
          aria-label="主导航"
          className={`main-nav${navigationOpen ? " main-nav-open" : ""}`}
          id="main-navigation"
        >
          <Link aria-current="page" href="/projects"><Activity aria-hidden="true" />分析工作台</Link>
          <span aria-disabled="true" className="nav-disabled"><Database aria-hidden="true" />数据中心</span>
          <span aria-disabled="true" className="nav-disabled"><Activity aria-hidden="true" />运行记录</span>
          <span aria-disabled="true" className="nav-disabled"><BookOpenCheck aria-hidden="true" />报告中心</span>
        </nav>
        <div className="header-actions">
          <span className="trust-status"><ShieldCheck aria-hidden="true" />可信边界已开启</span>
          {logoutError ? <span className="header-error" role="alert">退出失败，请重试</span> : null}
          <button
            aria-label={logoutPending ? "正在退出" : "退出登录"}
            className="icon-button header-logout"
            disabled={logoutPending}
            onClick={() => void performLogout()}
            title={logoutPending ? "正在退出" : "退出登录"}
            type="button"
          >
            <LogOut aria-hidden="true" size={17} />
          </button>
        </div>
      </header>
      {workflowStage ? <WorkflowRail current={workflowStage} /> : null}
      <main className="main-content">
        {children}
      </main>
    </div>
  );
}
