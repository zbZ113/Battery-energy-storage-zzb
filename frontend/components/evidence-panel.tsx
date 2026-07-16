import { Braces, FileCheck2, ShieldAlert, ShieldCheck } from "lucide-react";

import type { ToolResult } from "@/lib/types";

function JsonValue({ value }: { value: unknown }) {
  return <pre className="json-value">{JSON.stringify(value, null, 2)}</pre>;
}

export function EvidencePanel({ result }: { result: ToolResult | null }) {
  if (result === null) {
    return (
      <section className="empty-state">
        <FileCheck2 aria-hidden="true" />
        <h2>尚未签发结果</h2>
        <p>只有后端工具完成计算并签发 ToolResult 后，结果和证据才会显示。</p>
      </section>
    );
  }

  return (
    <div className="evidence-layout">
      <section className="panel evidence-summary">
        <div className="eyebrow"><ShieldCheck aria-hidden="true" size={16} />已签发工具结果</div>
        <dl className="metadata-list">
          <div><dt>结果 ID</dt><dd><code>{result.result_id}</code></dd></div>
          <div><dt>工具</dt><dd>{result.tool_name}</dd></div>
          <div><dt>工具版本</dt><dd>{result.tool_version}</dd></div>
          <div><dt>输入哈希</dt><dd><code>{result.input_hash}</code></dd></div>
        </dl>
      </section>
      <section className="panel">
        <div className="panel-heading"><Braces aria-hidden="true" /><div><h2>服务端签发值</h2><p>原样展示，不在浏览器中重新计算。</p></div></div>
        <JsonValue value={result.values} />
      </section>
      <section className="panel">
        <div className="panel-heading"><FileCheck2 aria-hidden="true" /><div><h2>来源链</h2><p>用于复核数据、模型与工具来源。</p></div></div>
        {result.provenance.length ? (
          <ul className="provenance-list">
            {result.provenance.map((item) => (
              <li key={`${item.source_id}:${item.sha256}`}>
                <div>
                  <strong>{item.source_id}</strong>
                  <span className="status-pill">{item.source_kind}</span>
                </div>
                <p>{item.description}</p>
                <code>{item.uri}</code>
                <small>SHA-256：{item.sha256}</small>
              </li>
            ))}
          </ul>
        ) : <p className="muted">该结果没有可展示的来源记录。</p>}
      </section>
      {result.warnings.length ? (
        <section className="panel warning-panel">
          <div className="panel-heading"><ShieldAlert aria-hidden="true" /><div><h2>限制与警告</h2><p>这些信息不能被页面隐藏。</p></div></div>
          <ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </section>
      ) : null}
    </div>
  );
}
