export function AnalysisWorkspace({
  analysis,
  operations,
}: {
  analysis: React.ReactNode;
  operations: React.ReactNode;
}) {
  return (
    <div className="analysis-workspace">
      <section aria-label="分析与结果" className="analysis-pane">
        {analysis}
      </section>
      <aside aria-label="运行与进度" className="operations-pane" role="region">
        {operations}
      </aside>
    </div>
  );
}
