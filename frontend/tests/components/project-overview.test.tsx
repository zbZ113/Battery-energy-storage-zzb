import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ProjectOverview } from "@/components/project-overview";

describe("ProjectOverview", () => {
  it("renders a clear empty state without invented project metrics", () => {
    render(<ProjectOverview projects={[]} />);

    expect(screen.getByText("还没有可查看的项目")).toBeInTheDocument();
    expect(screen.getByText(/联系管理员创建项目/)).toBeInTheDocument();
  });

  it("links every visible project to its authenticated detail page", () => {
    render(
      <ProjectOverview
        projects={[
          {
            project_id: "proj-safe",
            name: "HUST 早期寿命研究",
            owner_user_id: "user-1",
            created_at: "2026-07-16T08:00:00Z",
          },
        ]}
      />,
    );

    expect(screen.getByRole("link", { name: /HUST 早期寿命研究/ })).toHaveAttribute(
      "href",
      "/projects/proj-safe",
    );
  });
});
