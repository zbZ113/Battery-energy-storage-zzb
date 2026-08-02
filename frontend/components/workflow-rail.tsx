export const WORKFLOW_STAGES = [
  { id: "project", label: "登录 / 项目" },
  { id: "source", label: "样例 / 上传" },
  { id: "scope", label: "电芯 / cutoff" },
  { id: "agent", label: "九步 Agent" },
  { id: "results", label: "结果 / 报告" },
  { id: "download", label: "下载" },
] as const;

export type WorkflowStage = (typeof WORKFLOW_STAGES)[number]["id"];

export function WorkflowRail({ current }: { current: WorkflowStage }) {
  return (
    <nav aria-label="完整流程" className="workflow-rail">
      <span className="workflow-label">完整流程</span>
      <ol>
        {WORKFLOW_STAGES.map((stage, index) => (
          <li
            aria-current={stage.id === current ? "step" : undefined}
            key={stage.id}
          >
            <span aria-hidden="true" className="workflow-number">{index + 1}</span>
            <span>{stage.label}</span>
          </li>
        ))}
      </ol>
    </nav>
  );
}
