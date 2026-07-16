"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { useState } from "react";

export function ResourceLocator({ kind }: { kind: "run" | "result" }) {
  const [identifier, setIdentifier] = useState("");
  const [runIdentifier, setRunIdentifier] = useState("");
  const normalized = identifier.trim();
  const normalizedRun = runIdentifier.trim();
  const isRun = kind === "run";
  const label = isRun ? "运行 ID" : "结果 ID";
  const href = isRun
    ? `/agent/runs/${encodeURIComponent(normalized)}`
    : `/results/${encodeURIComponent(normalized)}?run_id=${encodeURIComponent(normalizedRun)}`;
  const ready = normalized.length > 0 && (isRun || normalizedRun.length > 0);

  return (
    <div className="resource-locator">
      <div className="resource-locator-fields">
        {!isRun ? (
          <label className="field">
            <span>运行 ID</span>
            <span className="field-control">
              <input
                onChange={(event) => setRunIdentifier(event.target.value)}
                placeholder="粘贴运行 ID"
                value={runIdentifier}
              />
            </span>
          </label>
        ) : null}
        <label className="field">
          <span>{label}</span>
          <span className="field-control">
            <input
              onChange={(event) => setIdentifier(event.target.value)}
              placeholder={`粘贴${label}`}
              value={identifier}
            />
          </span>
        </label>
      </div>
      {ready ? (
        <Link className="button button-primary" href={href}>
          {isRun ? "查看运行" : "查看结果"}<ArrowRight aria-hidden="true" size={17} />
        </Link>
      ) : (
        <span className="button button-disabled" aria-disabled="true">
          请先填写{isRun ? label : "运行 ID 和结果 ID"}
        </span>
      )}
    </div>
  );
}
