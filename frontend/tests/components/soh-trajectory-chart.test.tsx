import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SohTrajectoryChart } from "@/components/soh-trajectory-chart";

describe("SohTrajectoryChart", () => {
  it("maps the server-issued finite trajectory and simultaneous band into an accessible SVG", () => {
    render(
      <SohTrajectoryChart
        coverageTarget={0.9}
        lowerSoh={[0.94, 0.93, 0.92]}
        predictedSoh={[0.99, 0.98, 0.97]}
        predictionCycles={[21, 22, 23]}
        upperSoh={[1.04, 1.03, 1.02]}
      />,
    );

    expect(
      screen.getByRole("img", {
        name: /SOH 有限 horizon 预测轨迹与 simultaneous Split Conformal band/,
      }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("soh-band")).toHaveAttribute("points");
    expect(screen.getByTestId("soh-prediction-line")).toHaveAttribute("points");
    expect(screen.getByText("cycle 21–23")).toBeInTheDocument();
    expect(screen.getByText("coverage_target = 0.9")).toBeInTheDocument();
  });

  it("fails closed when the server-issued axes do not align", () => {
    render(
      <SohTrajectoryChart
        coverageTarget={0.9}
        lowerSoh={[0.94]}
        predictedSoh={[0.99, 0.98]}
        predictionCycles={[21, 22]}
        upperSoh={[1.04, 1.03]}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("SOH 轨迹暂不可用");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
