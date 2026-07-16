import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ScopedResultLoader } from "@/components/scoped-result-loader";

describe("ScopedResultLoader", () => {
  it("refuses to load a result without an Agent run security context", () => {
    const loadResult = vi.fn();
    render(
      <ScopedResultLoader
        loadResult={loadResult}
        resultId="result-1"
        runId={null}
      />,
    );

    expect(loadResult).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "缺少运行 ID，无法核验你是否有权查看这个结果",
    );
  });
});
