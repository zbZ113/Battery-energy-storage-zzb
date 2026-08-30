"use client";

import { useRouter } from "next/navigation";
import { BatteryCharging, ShieldCheck } from "lucide-react";

import { LoginForm } from "@/components/login-form";
import { login } from "@/lib/api-client";

export default function LoginPage() {
  const router = useRouter();
  return (
    <main className="login-page">
      <section className="login-story">
        <div className="story-brand"><BatteryCharging aria-hidden="true" /><span>Hiro</span></div>
        <div>
          <p className="eyebrow">HIRO BATTERY INTELLIGENCE</p>
          <h1>让每一次寿命判断，<br />都有证据可循。</h1>
          <p className="story-lead">面向储能电芯的可信多智能体诊断平台。Agent负责协同，工程数值始终由受审计工具签发。</p>
        </div>
        <div className="story-trust"><ShieldCheck aria-hidden="true" /><span>会话凭证保存在安全的 HttpOnly Cookie 中，前端无法读取。</span></div>
      </section>
      <section className="login-panel">
        <div className="auth-card">
          <p className="eyebrow">受控访问</p>
          <h2>欢迎回来</h2>
          <p className="muted">使用管理员分配的账号进入诊断工作区。</p>
          <LoginForm onLogin={async (username, password) => {
            const principal = await login(username, password);
            router.push(principal.must_change_password ? "/login/change-password" : "/projects");
          }} />
        </div>
      </section>
    </main>
  );
}
