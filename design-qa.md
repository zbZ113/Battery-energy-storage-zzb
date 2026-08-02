# 方案 B 前端 Design QA

- 日期：2026-08-02
- 视觉真相：`docs/superpowers/specs/2026-08-02-quanxin-frontend-option-b.png`
- 生产预览：`http://127.0.0.1:3011`
- 构建配置：`NEXT_PUBLIC_API_BASE_URL=https://qa.quanxin.invalid`
- 浏览器数据边界：Playwright 只拦截项目元数据、run 状态和 SSE 状态事件；未注入 SOH、RUL、Conformal、阈值或报告业务数值，未使用或保存登录凭据。

## 自动门禁

| 门禁 | 结果 |
| --- | --- |
| Vitest | 21 files，81 passed |
| ESLint | exit 0，0 warnings |
| TypeScript | exit 0 |
| Next.js production build | exit 0，`/icon.svg` 与全部路由成功生成 |

构建只提示 `baseline-browser-mapping` 数据较旧；未安装、升级或批准任何依赖脚本。

## 视口证据

| 页面与视口 | 核验结果 |
| --- | --- |
| 登录 `1440 × 1024` | 双栏认证布局，无横向溢出 |
| 登录 `1024 × 768` | 双栏保持可读，无横向溢出 |
| 登录 `390 × 844` | 表单先于品牌说明；`scrollWidth === clientWidth` |
| 项目工作台 `1440 × 1024` | 左右栏同起点，约 62/38；唯一主操作为“启动全新九步 Agent” |
| 项目工作台 `1024 × 768` | 运行与进度区排在结果区之前；无页面横向溢出 |
| 项目工作台 `390 × 844` | 单栏；运行区先于结果区；流程条在自身区域可横向滚动；菜单可展开和收起 |
| Agent run `1440 × 1024` | 原始审计事件与九步状态双栏同屏 |
| Agent run `390 × 844` | 九步状态先于审计事件；无页面横向溢出 |
| 单电芯结果等待页 `1440 × 1024` | 显示“等待服务端签发分析结果”和“待服务端提供导出”；未出现业务指标 |

生产页面最终控制台：`0 errors / 0 warnings`。缺失 favicon 的早期缺陷已通过 `frontend/app/icon.svg` 修复。

## 可信状态

- 流程条只标记显式当前阶段，不再根据位置推断之前阶段已完成。
- 内置样例与上传入口在目录/上传 API 未接入时均为禁用控件。
- 项目空结果区只显示“等待 ToolResult”，不显示示例曲线或数值。
- 九步 Tracker 只在后端 plan step IDs 精确匹配 Runtime V7 时显示。
- SSE 事件校验 `run_id`、正整数 sequence、已知事件名、带时区时间和 JSON 对象 payload。
- 可重试失败显示“等待重试”；终态失败展示服务端 `failure_code`。
- 切换 run 会卸载旧运行会话，旧事件不会污染新 run。
- 导出接口未接入时无下载链接，不在浏览器拼装业务报告。

## 响应式与可访问性

- 顶部导航在 `<= 1024px` 使用带 `aria-expanded` 和 `aria-controls` 的菜单按钮。
- 流程条、图表和校准表格使用局部受控滚动，不制造整页横向溢出。
- 手机端认证表单排在品牌说明之前。
- 状态同时使用图标、文字和颜色；九步状态位于 `aria-live="polite"` 区域。
- `prefers-reduced-motion: reduce` 下卡片 transform 为 `none`，过渡时长为 `0.01ms`。
- 焦点环和辅助文字颜色已提高对比度。

## 截图

- `output/playwright/option-b-prod-workspace-1440.png`
- `output/playwright/option-b-prod-workspace-390.png`
- `output/playwright/option-b-prod-login-390.png`
- `output/playwright/option-b-workspace-1024.png`
- `output/playwright/option-b-run-1440.png`
- `output/playwright/option-b-run-390.png`
- `output/playwright/option-b-results-waiting-1440.png`

## 保留限制

- 项目数据目录、电芯/cutoff 目录、正式上传、运行/结果目录和导出接口仍属于后续 Task 3、5、6。
- 本轮浏览器 QA 是前端布局与可信空状态验收，不替代真实登录、真实 ToolResult、数据库、Worker 或 ECS 纵向验收。
- 公网镜像与当前 HEAD 的来源一致性仍未核验。

final result: passed
