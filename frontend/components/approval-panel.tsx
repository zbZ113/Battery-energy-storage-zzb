"use client";

import { useState } from "react";
import { ShieldAlert, ThumbsDown, ThumbsUp } from "lucide-react";

export function ApprovalPanel({
  approvalId,
  approvalKind,
  onApprove,
  onReject,
}: {
  approvalId: string;
  approvalKind: string;
  onApprove: (approvalId: string, reason: string | null) => Promise<void>;
  onReject: (approvalId: string, reason: string | null) => Promise<void>;
}) {
  const [reason, setReason] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const normalizedReason = reason.trim() || null;

  async function act(action: "approve" | "reject") {
    setPending(true);
    setError(null);
    try {
      await (action === "approve"
        ? onApprove(approvalId, normalizedReason)
        : onReject(approvalId, normalizedReason));
    } catch {
      setError("审批操作未成功，任务状态可能已经变化。请刷新后重试。");
      setPending(false);
    }
  }

  return (
    <section className="panel approval-panel">
      <div className="panel-heading"><ShieldAlert aria-hidden="true" /><div><h2>等待你的明确审批</h2><p>类型：{approvalKind}</p></div></div>
      <p>这一步会产生正式决策或外部影响，系统不会自动替你同意。</p>
      <label className="field">
        <span>审批理由（可选）</span>
        <textarea maxLength={2000} onChange={(event) => setReason(event.target.value)} value={reason} />
      </label>
      {error ? <p className="alert alert-error" role="alert">{error}</p> : null}
      <div className="approval-actions">
        <button className="button button-primary" disabled={pending} onClick={() => void act("approve")} type="button"><ThumbsUp aria-hidden="true" size={16} />批准并继续</button>
        <button className="button button-quiet" disabled={pending} onClick={() => void act("reject")} type="button"><ThumbsDown aria-hidden="true" size={16} />拒绝并停止</button>
      </div>
    </section>
  );
}
