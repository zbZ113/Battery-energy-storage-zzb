# 泉芯智寿方案 B 前端实施计划

> 执行约束：用户禁止使用任何 `superpowers:*` 技能，且未授权 commit/push。本计划由根 Agent 在当前工作区逐项执行，每个行为变更严格执行红灯、最小实现、绿灯和回归验证。

**目标：** 将现有门户实现为方案 B 的双栏分析工作台，并为尚未接入的上传、目录和导出能力提供真实、明确的不可用状态。

**架构：** 保留 Next.js App Router、现有 API client、ToolResult 校验和 SSE 数据流。新增纯展示的流程条、工作台骨架和九步跟踪组件；现有业务组件继续负责数据加载与安全校验，CSS 统一实现桌面双栏和移动单栏。

**技术栈：** Next.js 16、React 19、TypeScript、Lucide React、Vitest、Testing Library、原生 CSS。

---

## 文件职责

- `frontend/app/globals.css`：方案 B 令牌、全局布局、响应式和状态样式。
- `frontend/components/app-shell.tsx`：顶部导航、可信边界、账号菜单和内容外壳。
- `frontend/components/workflow-rail.tsx`：六段完整产品流程，只接收显式当前阶段。
- `frontend/components/analysis-workspace.tsx`：无数据请求的双栏布局容器。
- `frontend/components/nine-step-tracker.tsx`：固定九步名称和 run/event 状态映射。
- `frontend/components/create-agent-run-form.tsx`：唯一主操作文案与现有创建契约。
- `frontend/app/projects/[projectId]/page.tsx`：项目工作台组合与真实兼容入口。
- `frontend/app/agent/runs/[runId]/page.tsx`：SSE 运行页组合。
- `frontend/components/single-cell-analysis.tsx`：结果标签、报告和证据布局。
- `frontend/app/login/*.tsx`、`frontend/app/projects/page.tsx`、校准页和结果页：统一页面框架。
- `frontend/tests/components/*.test.tsx`：行为、可访问性和可信等待态回归测试。
- `design-qa.md`：浏览器尺寸、页面、控制台、溢出和视觉基准核验结果。

### Task 1：顶部 AppShell 与六段流程

**测试：** `frontend/tests/components/app-shell.test.tsx`、新建 `frontend/tests/components/workflow-rail.test.tsx`

1. 在 AppShell 测试中断言存在顶部主导航、可信边界文案、退出按钮和主内容；断言旧侧栏类名不再作为页面结构。
2. 运行目标测试，确认因顶部结构和流程组件尚不存在而失败。
3. 新建 `WorkflowRail`，限定阶段联合类型为 `project | source | scope | agent | results | download`，渲染六个有序步骤并用 `aria-current="step"` 标记当前阶段。
4. 修改 `AppShell` 为顶部布局，保留现有登出行为和错误处理。
5. 在 `globals.css` 写入批准色值、可见焦点、顶部栏、流程条和响应式样式。
6. 重跑目标测试并确认通过。

运行：

```powershell
frontend\node_modules\.bin\vitest.cmd run tests/components/app-shell.test.tsx tests/components/workflow-rail.test.tsx
```

### Task 2：项目双栏工作台与唯一主操作

**测试：** `frontend/tests/components/create-agent-run-form.test.tsx`、新建 `frontend/tests/components/analysis-workspace.test.tsx`

1. 添加测试，断言主操作文案精确为“启动全新九步 Agent”，提交仍只发送现有 `AgentRunCreateRequest`，没有浏览器生成结果。
2. 添加工作台测试，断言左右两个具名区域存在；上传能力显示“尚未接入”且没有文件输入；结果区无 ToolResult 时显示等待文案。
3. 运行目标测试，确认因新文案和组件缺失而失败。
4. 新建 `AnalysisWorkspace` 作为纯布局组件。
5. 修改项目页：加入流程条、数据来源摘要、现有创建表单、兼容 UUID 定位入口、结果等待区和下载等待区。
6. 修改创建表单按钮和辅助说明，保持现有 API payload 与错误处理不变。
7. 添加 CSS 双栏比例、标签栏、运行侧栏和不可用状态。
8. 重跑目标测试并确认通过。

运行：

```powershell
frontend\node_modules\.bin\vitest.cmd run tests/components/create-agent-run-form.test.tsx tests/components/analysis-workspace.test.tsx
```

### Task 3：固定九步运行与 SSE 状态

**测试：** 新建 `frontend/tests/components/nine-step-tracker.test.tsx`，扩展 `frontend/tests/components/agent-run-live-view.test.tsx`

1. 添加测试，断言九步名称和顺序固定，未知事件不被标记为完成，失败步骤显示失败文字，等待步骤不显示结果值。
2. 添加运行页测试，断言九步跟踪和原始事件明细同时存在，SSE 断线提示仍可重连。
3. 运行目标测试并确认因组件缺失而失败。
4. 新建 `NineStepTracker`，从事件中的显式 ordinal、step id 或受支持的 step name 更新状态；无法可靠映射的事件只留在事件明细中。
5. 修改运行页为双栏：左侧原始审计事件，右侧九步实时状态和审批区。
6. 重跑目标测试并确认通过。

运行：

```powershell
frontend\node_modules\.bin\vitest.cmd run tests/components/nine-step-tracker.test.tsx tests/components/agent-run-live-view.test.tsx
```

### Task 4：结果、报告与证据布局

**测试：** `frontend/tests/components/single-cell-analysis.test.tsx`、`frontend/tests/components/evidence-panel.test.tsx`、`frontend/tests/components/scoped-result-loader.test.tsx`

1. 添加测试，断言无结果时显示等待 ToolResult；有结果时只有契约解析后的值进入 SOH、RUL、Conformal 和报告区域；下载区在无导出接口时显示“待服务端提供”。
2. 运行目标测试并确认新布局断言失败。
3. 在不改变 `parseAnalysis` 的前提下重组 `SingleCellAnalysis`：结果标签、图表区、报告区、警告区和证据 dock。
4. 更新单电芯页和结果页流程阶段，保留项目/运行作用域校验。
5. 重跑目标测试并确认通过。

运行：

```powershell
frontend\node_modules\.bin\vitest.cmd run tests/components/single-cell-analysis.test.tsx tests/components/evidence-panel.test.tsx tests/components/scoped-result-loader.test.tsx
```

### Task 5：其余页面和响应式统一

**测试：** `frontend/tests/components/login-form.test.tsx`、`frontend/tests/components/change-password-form.test.tsx`、`frontend/tests/components/project-overview.test.tsx`、`frontend/tests/components/calibration-materialization-panel.test.tsx`

1. 添加或调整结构断言，覆盖紧凑认证页、项目状态、校准表格和键盘可达名称。
2. 运行目标测试，确认只因新结构要求失败。
3. 更新登录、改密、项目总览、校准和高级定位样式；移除夸张渐变、过大标题和超过 8px 的区块圆角。
4. 添加 `prefers-reduced-motion`、平板和移动断点，保证表格受控滚动且页面不产生横向溢出。
5. 重跑目标测试并确认通过。

### Task 6：完整门禁和 Design QA

1. 运行前端完整 Vitest、ESLint、TypeScript 和生产构建。
2. 启动未占用端口的 Next.js 开发服务器，不安装或更新依赖。
3. 使用真实浏览器检查登录、项目、工作台、运行、结果和校准路由的可达状态。
4. 在 `1440 × 1024`、`1024 × 768` 和 `390 × 844` 检查 `scrollWidth`、文字裁切、控件重叠、控制台错误和截图。
5. 核对方案 B 基准图、唯一主操作、等待 ToolResult 和不可用状态。
6. 写入根目录 `design-qa.md`；只有全部检查有最新证据时，最后一行写 `final result: passed`。
7. 按仓库规则运行 Python `pytest`、Ruff、mypy 和 compileall，确认前端改动没有破坏整体门禁。

完整前端命令：

```powershell
frontend\node_modules\.bin\vitest.cmd run
frontend\node_modules\.bin\eslint.cmd . --max-warnings=0
frontend\node_modules\.bin\tsc.cmd --noEmit
frontend\node_modules\.bin\next.cmd build
```

安全说明：任何测试失败、浏览器溢出、控制台错误、缺少后端证据或未接入能力都必须按实际状态记录，不允许为了得到 `passed` 隐藏问题。
