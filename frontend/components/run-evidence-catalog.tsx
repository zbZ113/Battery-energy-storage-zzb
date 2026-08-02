import { ShieldCheck } from "lucide-react";

import { EvidencePanel } from "@/components/evidence-panel";
import { RUNTIME_V7_STEPS } from "@/components/nine-step-tracker";
import type { AgentRunResultRecord } from "@/lib/types";

const STEP_LABELS = new Map<string, string>(
  RUNTIME_V7_STEPS.map((step) => [step.id, step.label]),
);

export function RunEvidenceCatalog({ results }: { results: AgentRunResultRecord[] }) {
  if (!results.length) return null;
  return (
    <section className="evidence-stack" aria-labelledby="run-evidence-catalog-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow"><ShieldCheck aria-hidden="true" size={16} />RUNTIME V7 EVIDENCE</p>
          <h2 id="run-evidence-catalog-heading">九步 ToolResult 证据目录</h2>
        </div>
        <p>仅列出结果目录 API 已签发并通过九步身份核验的 ToolResult。</p>
      </div>
      {results.map((entry) => (
        <details className="evidence-disclosure" key={entry.result.result_id}>
          <summary>
            {String(entry.ordinal).padStart(2, "0")} · {STEP_LABELS.get(entry.step_id) ?? entry.step_id} · <code>{entry.result.result_id}</code>
          </summary>
          <div aria-label={`${entry.step_id} ToolResult`}>
            <EvidencePanel result={entry.result} />
          </div>
        </details>
      ))}
    </section>
  );
}
