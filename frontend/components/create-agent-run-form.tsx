"use client";

import { FormEvent, useState } from "react";
import { Bot, Database, Play } from "lucide-react";

import type { AgentRunCreateRequest } from "@/lib/types";

export function CreateAgentRunForm({
  projectId,
  onCreate,
}: {
  projectId: string;
  onCreate: (request: AgentRunCreateRequest) => Promise<void>;
}) {
  const [goal, setGoal] = useState("");
  const [datasetText, setDatasetText] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const datasetIds = Array.from(
      new Set(datasetText.split(/[\n,]/).map((item) => item.trim()).filter(Boolean)),
    );
    if (datasetIds.length === 0) {
      setError("至少填写一个已冻结的数据集 ID。");
      return;
    }
    setPending(true);
    try {
      await onCreate({
        project_id: projectId,
        user_goal: goal.trim(),
        dataset_ids: datasetIds,
        requested_outputs: ["cycle_life"],
      });
    } catch {
      setError("任务创建失败。请确认数据集已冻结，并且你有该项目的操作权限。");
      setPending(false);
    }
  }

  return (
    <form className="agent-run-form" onSubmit={submit}>
      <label className="field">
        <span><Bot aria-hidden="true" size={16} />你想让 Agent 完成什么</span>
        <textarea
          maxLength={4000}
          minLength={2}
          onChange={(event) => setGoal(event.target.value)}
          placeholder="例如：分析这批电芯的循环寿命风险，并给出带证据的复检建议"
          required
          value={goal}
        />
      </label>
      <label className="field">
        <span><Database aria-hidden="true" size={16} />冻结数据集 ID</span>
        <textarea
          onChange={(event) => setDatasetText(event.target.value)}
          placeholder="每行一个，也可以用逗号分隔"
          value={datasetText}
        />
      </label>
      <p className="form-note form-note-left">默认安全输出：循环寿命（cycle_life）。工程数值仍只由后端工具签发。</p>
      {error ? <p className="alert alert-error" role="alert">{error}</p> : null}
      <button className="button button-primary" disabled={pending} type="submit">
        <Play aria-hidden="true" size={17} />{pending ? "正在启动…" : "启动全新九步 Agent"}
      </button>
    </form>
  );
}
