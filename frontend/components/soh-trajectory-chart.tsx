"use client";

import { useId } from "react";

type SohTrajectoryChartProps = {
  coverageTarget: number;
  lowerSoh: number[];
  predictedSoh: number[];
  predictionCycles: number[];
  upperSoh: number[];
};

const WIDTH = 760;
const HEIGHT = 360;
const PADDING = { bottom: 50, left: 62, right: 24, top: 24 };

function isFiniteSeries(values: number[]): boolean {
  return values.length > 0 && values.every(Number.isFinite);
}

function isValidTrajectory({
  coverageTarget,
  lowerSoh,
  predictedSoh,
  predictionCycles,
  upperSoh,
}: SohTrajectoryChartProps): boolean {
  const lengths = new Set([
    lowerSoh.length,
    predictedSoh.length,
    predictionCycles.length,
    upperSoh.length,
  ]);
  return (
    Number.isFinite(coverageTarget)
    && coverageTarget > 0
    && coverageTarget < 1
    && lengths.size === 1
    && isFiniteSeries(lowerSoh)
    && isFiniteSeries(predictedSoh)
    && isFiniteSeries(predictionCycles)
    && isFiniteSeries(upperSoh)
    && predictionCycles.every(
      (cycle, index) => Number.isInteger(cycle) && (index === 0 || cycle > predictionCycles[index - 1]),
    )
    && predictedSoh.every(
      (point, index) => lowerSoh[index] <= point && point <= upperSoh[index],
    )
  );
}

function coordinateMapper(values: number[], outputStart: number, outputEnd: number) {
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = maximum - minimum;
  return (value: number): number => {
    if (span === 0) return (outputStart + outputEnd) / 2;
    return outputStart + ((value - minimum) / span) * (outputEnd - outputStart);
  };
}

function points(
  cycles: number[],
  values: number[],
  x: (value: number) => number,
  y: (value: number) => number,
): string {
  return cycles.map((cycle, index) => `${x(cycle)},${y(values[index])}`).join(" ");
}

export function SohTrajectoryChart(props: SohTrajectoryChartProps) {
  const titleId = useId();
  const descriptionId = useId();
  if (!isValidTrajectory(props)) {
    return (
      <p className="alert alert-error" role="alert">
        SOH 轨迹暂不可用：服务端签发的 cycle、预测值或 band 未对齐。
      </p>
    );
  }

  const {
    coverageTarget,
    lowerSoh,
    predictedSoh,
    predictionCycles,
    upperSoh,
  } = props;
  const x = coordinateMapper(
    predictionCycles,
    PADDING.left,
    WIDTH - PADDING.right,
  );
  const yValues = [...lowerSoh, ...predictedSoh, ...upperSoh];
  const y = coordinateMapper(
    yValues,
    HEIGHT - PADDING.bottom,
    PADDING.top,
  );
  const lowerPoints = predictionCycles.map(
    (cycle, index) => `${x(cycle)},${y(lowerSoh[index])}`,
  );
  const upperPoints = predictionCycles
    .map((cycle, index) => `${x(cycle)},${y(upperSoh[index])}`)
    .reverse();
  const firstCycle = predictionCycles[0];
  const lastCycle = predictionCycles[predictionCycles.length - 1];

  return (
    <figure className="trajectory-chart">
      <svg
        aria-labelledby={`${titleId} ${descriptionId}`}
        role="img"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      >
        <title id={titleId}>
          SOH 有限 horizon 预测轨迹与 simultaneous Split Conformal band
        </title>
        <desc id={descriptionId}>
          图形仅将服务端签发的 prediction_cycles、predicted_soh、lower_soh
          和 upper_soh 映射为坐标，不在浏览器重新计算区间。
        </desc>
        <line
          className="chart-axis"
          x1={PADDING.left}
          x2={WIDTH - PADDING.right}
          y1={HEIGHT - PADDING.bottom}
          y2={HEIGHT - PADDING.bottom}
        />
        <line
          className="chart-axis"
          x1={PADDING.left}
          x2={PADDING.left}
          y1={PADDING.top}
          y2={HEIGHT - PADDING.bottom}
        />
        <polygon
          className="soh-band"
          data-testid="soh-band"
          points={[...lowerPoints, ...upperPoints].join(" ")}
        />
        <polyline
          className="soh-band-edge"
          points={points(predictionCycles, lowerSoh, x, y)}
        />
        <polyline
          className="soh-band-edge"
          points={points(predictionCycles, upperSoh, x, y)}
        />
        <polyline
          className="soh-prediction-line"
          data-testid="soh-prediction-line"
          points={points(predictionCycles, predictedSoh, x, y)}
        />
        <text className="chart-axis-label" x={WIDTH / 2} y={HEIGHT - 10}>
          cycle
        </text>
        <text
          className="chart-axis-label"
          transform={`rotate(-90 18 ${HEIGHT / 2})`}
          x={18}
          y={HEIGHT / 2}
        >
          SOH
        </text>
      </svg>
      <figcaption className="chart-caption">
        <span>cycle {firstCycle}–{lastCycle}</span>
        <span>coverage_target = {coverageTarget}</span>
      </figcaption>
      <div className="chart-legend" aria-label="图例">
        <span><i className="legend-line" />服务端预测 SOH</span>
        <span><i className="legend-band" />simultaneous finite band</span>
      </div>
    </figure>
  );
}
