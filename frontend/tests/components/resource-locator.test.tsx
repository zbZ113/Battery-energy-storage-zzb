import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ResourceLocator } from "@/components/resource-locator";

describe("ResourceLocator", () => {
  it("does not invent an identifier and navigates only after user input", async () => {
    const user = userEvent.setup();
    render(<ResourceLocator kind="run" />);

    expect(screen.queryByRole("link", { name: "查看运行" })).not.toBeInTheDocument();
    await user.type(screen.getByLabelText("运行 ID"), "run reviewed/1");

    expect(screen.getByRole("link", { name: "查看运行" })).toHaveAttribute(
      "href",
      "/agent/runs/run%20reviewed%2F1",
    );
  });

  it("requires both run and result identifiers for project-scoped result access", async () => {
    const user = userEvent.setup();
    render(<ResourceLocator kind="result" />);

    await user.type(screen.getByLabelText("结果 ID"), "result/1");
    expect(screen.queryByRole("link", { name: "查看结果" })).not.toBeInTheDocument();

    await user.type(screen.getByLabelText("运行 ID"), "run/1");
    expect(screen.getByRole("link", { name: "查看结果" })).toHaveAttribute(
      "href",
      "/results/result%2F1?run_id=run%2F1",
    );
  });
});
