"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Activity, BatteryCharging, BookOpenCheck, FolderKanban, LogOut, ShieldCheck } from "lucide-react";

import { logout } from "@/lib/api-client";

export function AppShell({
  children,
  onLogout,
}: {
  children: React.ReactNode;
  onLogout?: () => Promise<void>;
}) {
  const router = useRouter();
  const [logoutPending, setLogoutPending] = useState(false);
  const [logoutError, setLogoutError] = useState(false);

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
      <aside className="sidebar">
        <Link className="brand" href="/projects">
          <span className="brand-mark"><BatteryCharging aria-hidden="true" /></span>
          <span><strong>泉芯智寿</strong><small>可信电芯智能诊断</small></span>
        </Link>
        <nav aria-label="主导航">
          <Link href="/projects"><FolderKanban aria-hidden="true" />项目总览</Link>
          <span className="nav-disabled"><Activity aria-hidden="true" />运行任务</span>
          <span className="nav-disabled"><BookOpenCheck aria-hidden="true" />知识证据</span>
        </nav>
        <div className="trust-note"><ShieldCheck aria-hidden="true" /><p><strong>数值可信边界</strong><br />页面只展示后端工具签发结果。</p></div>
        {logoutError ? <p className="sidebar-error" role="alert">退出失败，请检查网络后重试。</p> : null}
        <button className="sidebar-logout" disabled={logoutPending} onClick={() => void performLogout()} type="button">
          <LogOut aria-hidden="true" size={16} />{logoutPending ? "正在退出…" : "退出登录"}
        </button>
      </aside>
      <main className="main-content">{children}</main>
    </div>
  );
}
