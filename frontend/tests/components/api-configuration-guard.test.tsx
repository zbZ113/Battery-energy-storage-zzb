import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ApiConfigurationGuard } from "@/components/api-configuration-guard";

describe("ApiConfigurationGuard", () => {
  it("shows a visible deployment warning instead of silently using an unsafe API", () => {
    render(
      <ApiConfigurationGuard
        validate={() => { throw new Error("生产环境必须配置 NEXT_PUBLIC_API_BASE_URL"); }}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "生产环境必须配置 NEXT_PUBLIC_API_BASE_URL",
    );
  });
});
