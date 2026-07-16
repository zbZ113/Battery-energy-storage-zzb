"use client";

import { useRouter } from "next/navigation";
import { KeyRound, ShieldCheck } from "lucide-react";

import { ChangePasswordForm } from "@/components/change-password-form";
import { changePassword } from "@/lib/api-client";

export default function ChangePasswordPage() {
  const router = useRouter();

  return (
    <main className="login-page">
      <section className="login-story">
        <div className="story-brand"><KeyRound aria-hidden="true" /><span>首次登录保护</span></div>
        <div>
          <p className="eyebrow">CREDENTIAL UPDATE</p>
          <h1>先换掉临时密码，<br />再进入诊断工作区。</h1>
          <p className="story-lead">管理员分配的临时密码只用于第一次登录。新密码不会保存在浏览器页面中。</p>
        </div>
        <div className="story-trust"><ShieldCheck aria-hidden="true" /><span>密码由后端安全校验和哈希；前端只通过加密连接提交。</span></div>
      </section>
      <section className="login-panel">
        <div className="auth-card">
          <p className="eyebrow">必须完成</p>
          <h2>设置你的新密码</h2>
          <p className="muted">输入当前临时密码，并连续输入两次相同的新密码。</p>
          <ChangePasswordForm
            onChangePassword={async (currentPassword, newPassword) => {
              await changePassword(currentPassword, newPassword);
              router.replace("/projects");
            }}
          />
        </div>
      </section>
    </main>
  );
}
