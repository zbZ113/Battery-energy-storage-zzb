import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EvidencePanel } from "@/components/evidence-panel";

describe("EvidencePanel", () => {
  it("shows an explicit no-data state", () => {
    render(<EvidencePanel result={null} />);
    expect(screen.getByText("尚未签发结果")).toBeInTheDocument();
  });

  it("shows provenance and server-issued values as read-only evidence", () => {
    render(
      <EvidencePanel
        result={{
          result_id: "result-001",
          tool_name: "predict_cycle_life",
          tool_version: "1.0.0",
          input_hash: "sha256:abc",
          values: { status: "RECHECK" },
          uncertainty: null,
          provenance: [
            {
              source_id: "dataset:hust-reviewed",
              source_kind: "OBSERVED",
              uri: "minio://datasets/hust-reviewed.parquet",
              sha256: "a".repeat(64),
              description: "经过审核的HUST安全转换数据",
              created_at: "2026-07-16T10:00:00Z",
            },
          ],
          warnings: ["目标域仍需复检"],
          created_at: "2026-07-16T10:00:00Z",
        }}
      />,
    );

    expect(screen.getByText("result-001")).toBeInTheDocument();
    expect(screen.getByText("dataset:hust-reviewed")).toBeInTheDocument();
    expect(screen.getByText("经过审核的HUST安全转换数据")).toBeInTheDocument();
    expect(screen.getByText("OBSERVED")).toBeInTheDocument();
    expect(screen.getByText("目标域仍需复检")).toBeInTheDocument();
  });
});
