"use client";

import { FormEvent, useState } from "react";
import { ArrowRight, LockKeyhole } from "lucide-react";

interface ChangePasswordFormProps {
  onChangePassword: (currentPassword: string, newPassword: string) => Promise<void>;
}

export function ChangePasswordForm({ onChangePassword }: ChangePasswordFormProps) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (newPassword !== confirmation) {
      setError("两次输入的新密码不一致，请重新确认。");
      return;
    }
    setPending(true);
    try {
      await onChangePassword(currentPassword, newPassword);
    } catch {
      setError("密码修改失败。请确认临时密码正确，并换一个更安全的新密码。");
      setPending(false);
    }
  }

  return (
    <form className="auth-form" onSubmit={submit}>
      <PasswordField
        autoComplete="current-password"
        label="当前临时密码"
        onChange={setCurrentPassword}
        value={currentPassword}
      />
      <PasswordField
        autoComplete="new-password"
        label="新密码"
        onChange={setNewPassword}
        value={newPassword}
      />
      <PasswordField
        autoComplete="new-password"
        label="再次输入新密码"
        onChange={setConfirmation}
        value={confirmation}
      />
      {error ? <p className="alert alert-error" role="alert">{error}</p> : null}
      <button className="button button-primary button-wide" disabled={pending} type="submit">
        {pending ? "正在保存…" : "保存新密码"}
        {!pending ? <ArrowRight aria-hidden="true" size={18} /> : null}
      </button>
      <p className="form-note">修改成功后，临时会话会被替换为新的安全会话。</p>
    </form>
  );
}

function PasswordField({
  autoComplete,
  label,
  onChange,
  value,
}: {
  autoComplete: string;
  label: string;
  onChange: (value: string) => void;
  value: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <span className="field-control">
        <LockKeyhole aria-hidden="true" size={18} />
        <input
          autoComplete={autoComplete}
          minLength={8}
          onChange={(event) => onChange(event.target.value)}
          required
          type="password"
          value={value}
        />
      </span>
    </label>
  );
}
