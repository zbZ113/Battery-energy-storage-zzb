"""Deterministic machine-readable and human-readable MATR experiment summaries."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

_REPORT_METRICS = (
    "mae",
    "rmse",
    "mape",
    "r2",
    "picp",
    "mpiw_cycle",
    "monotonic_violation_rate",
    "best_epoch",
    "last_epoch",
    "best_iteration",
    "training_time_seconds",
    "peak_gpu_memory_bytes",
)


def write_matr_experiment_reports(
    output_root: Path,
    aggregate: dict[str, Any],
) -> None:
    """Write flattened CSV and a target-aware Markdown report from aggregate data."""

    root = output_root.resolve(strict=True)
    summaries = aggregate.get("summaries")
    failures = aggregate.get("failed_metric_rows")
    if not isinstance(summaries, list) or not isinstance(failures, list):
        raise ValueError("aggregate report requires summary and failure lists")
    rows = [_flatten_summary(item) for item in summaries]
    _write_csv_atomic(root / "aggregate_metrics.csv", rows)
    _write_text_atomic(root / "experiment_summary.md", _render_markdown(aggregate, rows))


def _flatten_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
        raise ValueError("aggregate summary row is invalid")
    row: dict[str, Any] = {
        "model": summary.get("model"),
        "cutoff_cycle": summary.get("cutoff_cycle"),
        "target": summary.get("target"),
        "run_count": summary.get("run_count"),
        "seeds": ",".join(str(seed) for seed in summary.get("seeds", [])),
        "complete_seed_matrix": summary.get("complete_seed_matrix"),
    }
    metrics = summary["metrics"]
    for name in _REPORT_METRICS:
        value = metrics.get(name)
        if isinstance(value, dict):
            row[f"{name}_count"] = value.get("count")
            row[f"{name}_mean"] = value.get("mean")
            row[f"{name}_std"] = value.get("std")
    return row


def _render_markdown(aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    formal = aggregate.get("formal_performance_claim") is True
    status = (
        "正式五种子矩阵完整, 可在保留适用边界的前提下引用。"
        if formal
        else "本次运行不完整或属于 Smoke, 不得作为正式性能结论。"
    )
    lines = [
        "# 泉芯智寿 MATR 训练与评价汇总",
        "",
        f"- 运行模式: `{aggregate.get('mode')}`",
        f"- 运行记录数: {aggregate.get('run_count')}",
        f"- 预期种子数: {aggregate.get('expected_seed_count')}",
        f"- 结论状态: {status}",
        f"- 失败或不完整记录: {len(aggregate.get('failed_metric_rows', []))}",
        "",
        "## 目标语义",
        "",
        "标量寿命任务使用 **MATR 官方 cycle-life**; 该目标不得解释为统一 EOL80。",
        "Hybrid 使用循环 500 以内真实 QDischarge 构造的 SOH 轨迹, 不生成外推训练标签。",
        "",
        "## 汇总指标",
        "",
        "| 模型 | 截断循环 | 目标 | 种子完整 | MAE (均值 ± 标准差) | PICP | MPIW |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {cutoff} | {target} | {complete} | {mae} | {picp} | {mpiw} |".format(
                model=row.get("model", ""),
                cutoff=row.get("cutoff_cycle", ""),
                target=row.get("target", ""),
                complete="是" if row.get("complete_seed_matrix") is True else "否",
                mae=_mean_std(row, "mae"),
                picp=_number(row.get("picp_mean")),
                mpiw=_number(row.get("mpiw_cycle_mean")),
            )
        )
    if not rows:
        lines.append("| — | — | — | 否 | — | — | — |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "结果来自公开 MATR 实验室电芯, 不代表海辰或济南企业工业电芯验证。",
            "Conformal 覆盖率必须结合校准电芯数量与区间宽度解读; 跨域时需重新校准。",
            "",
        ]
    )
    return "\n".join(lines)


def _mean_std(row: dict[str, Any], name: str) -> str:
    mean = row.get(f"{name}_mean")
    std = row.get(f"{name}_std")
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        return f"{float(mean):.4f} ± {float(std):.4f}"
    return "—"


def _number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "—"


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "cutoff_cycle",
        "target",
        "run_count",
        "seeds",
        "complete_seed_matrix",
        *(f"{name}_{suffix}" for name in _REPORT_METRICS for suffix in ("count", "mean", "std")),
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
