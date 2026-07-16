"use client";

import { FormEvent, useState } from "react";
import { ArrowRight, LockKeyhole, UserRound } from "lucide-react";

interface LoginFormProps {
  onLogin: (username: string, password: string) => Promise<void>;
}

export function LoginForm({ onLogin }: LoginFormProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setError(null);
    try {
      await onLogin(username.trim(), password);
    } catch {
      setError("登录失败，请检查账号和密码后重试。");
      setPending(false);
    }
  }

  return (
    <form className="auth-form" onSubmit={submit}>
      <label className="field">
        <span>账号</span>
        <span className="field-control">
          <UserRound aria-hidden="true" size={18} />
          <input
            autoComplete="username"
            onChange={(event) => setUsername(event.target.value)}
            required
            value={username}
          />
        </span>
      </label>
      <label className="field">
        <span>密码</span>
        <span className="field-control">
          <LockKeyhole aria-hidden="true" size={18} />
          <input
            autoComplete="current-password"
            onChange={(event) => setPassword(event.target.value)}
            required
            type="password"
            value={password}
          />
        </span>
      </label>
      {error ? <p className="alert alert-error" role="alert">{error}</p> : null}
      <button className="button button-primary button-wide" disabled={pending} type="submit">
        {pending ? "正在验证…" : "安全登录"}
        {!pending ? <ArrowRight aria-hidden="true" size={18} /> : null}
      </button>
      <p className="form-note">账号由管理员创建，系统不开放自助注册。</p>
    </form>
  );
}
