"use client";

import { FormEvent, useMemo, useRef, useState } from "react";
import { BatteryMedium, Database, Play } from "lucide-react";

import { createAdvancedAnalysis } from "@/lib/api-client";
import { decodeAgentRun } from "@/lib/advanced-analysis-contract";
import type {
  AgentRunRecord,
  AnalysisInputsRecord,
  CreateAdvancedAnalysisRequest,
} from "@/lib/types";

type CreateAnalysis = (
  projectId: string,
  payload: CreateAdvancedAnalysisRequest,
  idempotencyKey: string,
) => Promise<AgentRunRecord>;

function nextKey(): string {
  return `advanced-analysis-${crypto.randomUUID()}`;
}

export function AdvancedAnalysisLauncher({
  createAnalysis = createAdvancedAnalysis,
  inputs,
  onCreated,
  projectId,
}: {
  createAnalysis?: CreateAnalysis;
  inputs: AnalysisInputsRecord;
  onCreated: (run: AgentRunRecord) => void;
  projectId: string;
}) {
  const usableInputs = inputs.project_id === projectId;
  const datasets = useMemo(
    () => usableInputs
      ? inputs.datasets.filter((dataset) => (
        dataset.project_id === projectId
        && dataset.status === "FROZEN"
        && inputs.batches.some((batch) => batch.dataset_id === dataset.dataset_id)
      ))
      : [],
    [inputs, projectId, usableInputs],
  );
  const [datasetId, setDatasetId] = useState(datasets[0]?.dataset_id ?? "");
  const datasetBatches = inputs.batches.filter((batch) => (
    batch.project_id === projectId && batch.dataset_id === datasetId
  ));
  const cellIds = Array.from(new Set(datasetBatches.map((batch) => batch.cell_id)));
  const [cellId, setCellId] = useState(cellIds[0] ?? "");
  const cellBatches = datasetBatches.filter((batch) => batch.cell_id === cellId);
  const [recordBatchId, setRecordBatchId] = useState(cellBatches[0]?.record_batch_id ?? "");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const idempotencyKey = useRef<string | null>(null);

  function resetRequestIdentity() {
    idempotencyKey.current = null;
    setError(null);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const batch = cellBatches.find((item) => item.record_batch_id === recordBatchId);
    if (!batch || pending) return;
    const key = idempotencyKey.current ?? nextKey();
    idempotencyKey.current = key;
    setPending(true);
    setError(null);
    try {
      const rawRun = await createAnalysis(
        projectId,
        {
          record_batch_id: batch.record_batch_id,
          cell_id: batch.cell_id,
          cutoff_cycle: batch.cutoff_cycle,
        },
        key,
      );
      const run = decodeAgentRun(rawRun, projectId, batch.record_batch_id);
      idempotencyKey.current = null;
      onCreated(run);
    } catch {
      setError("启动失败。请检查网络后重试；重试会复用同一次请求身份。");
    } finally {
      setPending(false);
    }
  }

  if (!usableInputs) {
    return <p className="alert alert-error" role="alert">分析输入不属于当前项目，已停止创建运行。</p>;
  }
  if (!datasets.length) {
    return <p className="muted" role="status">当前项目还没有可分析的冻结数据。</p>;
  }

  return (
    <form className="advanced-analysis-form" onSubmit={submit}>
      <label className="field">
        <span><Database aria-hidden="true" size={16} />数据集</span>
        <select
          aria-label="数据集"
          disabled={pending}
          onChange={(event) => {
            const nextDatasetId = event.target.value;
            const nextBatches = inputs.batches.filter((batch) => (
              batch.project_id === projectId && batch.dataset_id === nextDatasetId
            ));
            const nextCellId = nextBatches[0]?.cell_id ?? "";
            setDatasetId(nextDatasetId);
            setCellId(nextCellId);
            setRecordBatchId(
              nextBatches.find((batch) => batch.cell_id === nextCellId)?.record_batch_id ?? "",
            );
            resetRequestIdentity();
          }}
          value={datasetId}
        >
          {datasets.map((dataset) => (
            <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.name}</option>
          ))}
        </select>
      </label>
      <label className="field">
        <span><BatteryMedium aria-hidden="true" size={16} />电芯</span>
        <select
          aria-label="电芯"
          disabled={pending}
          onChange={(event) => {
            const nextCellId = event.target.value;
            setCellId(nextCellId);
            setRecordBatchId(
              datasetBatches.find((batch) => batch.cell_id === nextCellId)?.record_batch_id ?? "",
            );
            resetRequestIdentity();
          }}
          value={cellId}
        >
          {cellIds.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </label>
      <label className="field">
        <span>cutoff</span>
        <select
          aria-label="cutoff"
          disabled={pending}
          onChange={(event) => {
            setRecordBatchId(event.target.value);
            resetRequestIdentity();
          }}
          value={recordBatchId}
        >
          {cellBatches.map((batch) => (
            <option key={batch.record_batch_id} value={batch.record_batch_id}>
              {batch.cutoff_cycle} cycles
            </option>
          ))}
        </select>
      </label>
      <p className="form-note form-note-left">
        电芯与 cutoff 均来自服务端冻结目录；每次主动启动创建一个新的固定九步运行。
      </p>
      {error ? <p className="alert alert-error" role="alert">{error}</p> : null}
      <button className="button button-primary" disabled={pending || !recordBatchId} type="submit">
        <Play aria-hidden="true" size={17} />{pending ? "正在启动…" : "启动全新九步 Agent"}
      </button>
    </form>
  );
}
