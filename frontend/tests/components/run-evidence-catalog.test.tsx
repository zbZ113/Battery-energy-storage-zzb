import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RunEvidenceCatalog } from "@/components/run-evidence-catalog";

describe("RunEvidenceCatalog", () => {
  it("lists only ToolResults signed by the run result catalog", () => {
    render(
      <RunEvidenceCatalog
        results={[{
          step_id: "advanced-input",
          ordinal: 1,
          result: {
            result_id: "00000000-0000-4000-8000-000000000010",
            tool_name: "extract_early_cycle_features",
            tool_version: "advanced-input-v1",
            model_version: null,
            data_version: "data-v1",
            feature_version: "feature-v1",
            input_hash: "a".repeat(64),
            values: { artifact: { record_batch_id: "batch-1" } },
            uncertainty: null,
            warnings: [],
            provenance: [],
            created_at: "2026-08-02T04:00:00Z",
          },
        }]}
      />,
    );

    expect(screen.getByRole("heading", { name: "九步 ToolResult 证据目录" })).toBeInTheDocument();
    expect(screen.getByText(/01 · 冻结分析输入/)).toBeInTheDocument();
    expect(screen.getAllByText("00000000-0000-4000-8000-000000000010")).not.toHaveLength(0);
    expect(screen.queryByText("RUL 精度路线")).not.toBeInTheDocument();
  });
});
