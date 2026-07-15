# 泉芯智寿完整多智能体产品实施总计划

## 一、目标与已确定决策

**目标：** 建成一套面向储能电芯寿命诊断的完整多智能体产品，同时服务AI先锋海辰储能赛题和“华为杯”第八届中国研究生人工智能创新大赛。

赛事报名截止时间为2026年8月25日，作品提交截止时间为2026年9月1日。[大赛官方平台](https://cpipc.acge.org.cn/cw/contestNews/list/2c9088a5696cbf370169a3f8101510bd/1)、[高校最新通知](https://gs.shmtu.edu.cn/2026/0713/c8114a295182/page.htm)。

已锁定的决策：

- 采用两次发布：比赛版先交付，比赛后继续完整工业版。
- 单人开发，按纵向可运行切片推进，不并行铺开不可验收模块。
- 总体架构为模块化单体，不使用微服务或Kubernetes。
- 正式部署使用阿里云中国大陆服务器，预算每月300元以内。
- 使用外部OpenAI兼容LLM API，API不可用时退回无LLM固定工作流。
- Agent可在白名单内自主规划、重试和补证据；外部写入和正式决策需要审批。
- PostgreSQL管理状态，Redis管理任务，MinIO管理文件，JSONL保留不可变审计副本。
- Next.js作为正式门户，Streamlit作为科研工作台，飞书作为协同入口。
- 主演示是单电芯寿命诊断，批次风险与复检作为第二场景。
- 正式寿命以循环为单位；年份只允许由工具根据运行策略进行场景换算。
- HUST优先，原始pickle必须隔离转换；Naumann用于工况与主动试验。
- 安全转换后的数据和配置上传实验室A100训练，再下载安全模型制品。
- 代码提交前保持私有；提交后可公开源码，但暂不授予开源许可证。
- 正式项目长期保存，临时任务30天清理，原始聊天短期保存。
- 阿里云OSS保存加密异地备份。
- 用户界面以中文为主，保留英文论文题名、术语和引用。

---

## 二、最终系统架构

```text
Next.js / Streamlit / 飞书 / MCP
                │
                ▼
     FastAPI + 身份认证 + SSE事件流
                │
                ▼
 Supervisor / LLM Gateway / LangGraph
       │        │        │        │
       ▼        ▼        ▼        ▼
  数据Agent  寿命Agent  物理Agent  试验Agent
                │
                ▼
  现有确定性编排器 + Tool Registry + 白名单
                │
                ▼
 数据质检 / 模型 / Conformal / GP / PyBaMM
                │
                ▼
 ToolResult + Audit Ledger + 报告数值防火墙
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
 PostgreSQL   Redis     MinIO
 + pgvector  + Celery   + OSS备份
```

### 运行原则

- LLM只理解任务、生成计划、选择工具和解释结果。
- LLM不得生成、猜测或改写SOH、RUL、区间、阈值判断和业务指标。
- LangGraph负责对话状态、Agent路由、重规划和人工审批。
- 现有确定性编排器继续作为最终安全执行边界。
- Agent之间共享结构化状态、结果ID和证据，不共享隐藏推理过程。
- 工具失败后允许自主重规划，但最多两次。
- 单个计划最多12个步骤；瞬时工具故障最多重试一次。
- 规划连续失败、预算耗尽或LLM不可用时，自动切换固定模板工作流。
- 无法获得有效证据时输出`RECHECK`或明确不可用，不填充占位数字。

---

## 三、公共接口与数据模型

### 1. 新增核心类型

所有类型进入`quanxin_life.core`或复用现有公共契约，不另造重复DTO。

- `LlmProviderConfig`：API地址、模型名、超时、预算和能力检测结果。
- `AgentIntent`：用户目标、项目、数据对象、需要的输出。
- `AgentPlan`：计划版本、步骤、依赖、审批要求和计划哈希。
- `AgentPlanStep`：角色、工具、输入引用、失败策略。
- `AgentRunState`：运行状态、已完成步骤、待审批步骤、结果ID。
- `ApprovalRequest`：审批种类、来源计划、影响范围、过期时间。
- `KnowledgeDocumentManifest`：来源、许可证、SHA-256、版本和审核状态。
- `ScenarioLifetimeRequest`：循环寿命结果ID和明确运行策略。
- `ScenarioLifetimeResult`：场景换算结果及限制说明。
- `FeishuBinding`：项目、群聊、多维表格和用户映射。
- `UserRole`：`ADMIN`、`MEMBER`、`JUDGE`。

### 2. API

保留现有工具、批次和寿命工作流接口，新增：

```text
POST /v1/auth/login
POST /v1/auth/logout
POST /v1/auth/change-password
GET  /v1/auth/me

POST /v1/projects
GET  /v1/projects
GET  /v1/projects/{project_id}

POST /v1/datasets
GET  /v1/datasets/{dataset_id}
POST /v1/datasets/{dataset_id}/freeze

POST /v1/agent/runs
GET  /v1/agent/runs/{run_id}
GET  /v1/agent/runs/{run_id}/events
POST /v1/agent/runs/{run_id}/approve
POST /v1/agent/runs/{run_id}/reject
POST /v1/agent/runs/{run_id}/cancel

GET  /v1/results/{result_id}
GET  /v1/reports/{report_id}
POST /v1/reports/{report_id}/export

POST /v1/knowledge/documents
GET  /v1/knowledge/documents
POST /v1/knowledge/search

POST /v1/admin/users
POST /v1/admin/model-artifacts
POST /v1/admin/decision-policies
POST /v1/admin/calibration-cohorts

POST /v1/integrations/feishu/events
POST /v1/integrations/feishu/actions
```

Agent事件使用SSE推送，不通过频繁轮询。所有创建接口支持幂等键，防止浏览器刷新或飞书重复回调产生重复任务。

### 3. 数据库

主要表：

- 用户与权限：`users`、`sessions`、`user_project_roles`
- 项目与数据：`projects`、`datasets`、`dataset_files`、`cell_splits`
- 任务：`agent_runs`、`agent_steps`、`agent_events`
- 工具证据：`tool_results`、`provenance_records`
- 模型：`model_artifacts`、`model_manifests`
- 校准与策略：`calibration_cohorts`、`decision_policies`
- 审批：`approval_requests`、`approval_actions`
- 报告：`reports`、`report_exports`
- 知识库：`knowledge_documents`、`knowledge_chunks`
- 飞书：`feishu_bindings`、`feishu_event_receipts`

数据库只保存结构化记录和对象地址；CSV、Parquet、PDF、模型和报告文件进入MinIO。

---

## 四、分阶段实施

## 阶段0：AI先锋报名，未来3天

### 交付

- 保存报名字段、字数要求、附件要求和知识产权条款。
- 调研海辰储能公开业务、LFP储能寿命预测难点和公开数据条件。
- 完成Part 1前置洞察和Part 2整体方案。
- 制作3–6页附件：
  1. 业务问题与研究缺口；
  2. 多智能体系统架构；
  3. HUST与Naumann数据路线；
  4. 物理—数据融合、Conformal和在线校正；
  5. Demo与验证指标；
  6. 团队能力、风险和实施进度。
- 所有未来效果写为待验证目标，不填写虚构准确率、寿命或业务收益。
- 提交前执行事实核验、数值来源检查和夸大表述扫描。

### 用户提供

- 报名表最新截图或导出内容；
- 个人/团队简介、专业、项目经历；
- 可公开展示的GitHub或项目链接；
- 实际报名截止时间和提交入口。

---

## 阶段1：工程、云端与持久化底座，7月18日—22日

### 技术栈

- Python 3.11；
- FastAPI、Pydantic、SQLAlchemy 2、Alembic；
- Celery 5.6和Redis；
- PostgreSQL 17与pgvector；
- MinIO；
- Node.js 24 LTS、Next.js 16、TypeScript、pnpm；
- Streamlit与Plotly；
- Caddy反向代理；
- Docker Compose；
- GitHub Actions。

Node.js 24当前属于LTS；Next.js官方要求Node.js不低于20.9并支持标准Node服务器部署。[Node.js版本政策](https://nodejs.org/en/about/previous-releases)、[Next.js安装要求](https://nextjs.org/docs/app/getting-started/installation)。Celery 5.6是当前稳定系列；PostgreSQL 17仍处于官方支持期。[Celery文档](https://docs.celeryq.dev/en/stable/getting-started/)、[PostgreSQL版本政策](https://www.postgresql.org/support/versioning/)。

### 云服务器

- 阿里云深圳地域优先；
- Ubuntu 24.04 LTS；
- 4核、16GB内存；
- 目标总磁盘不低于160GB；
- 若轻量应用服务器无法在预算内满足磁盘要求，则改用同厂商ECS加ESSD；
- 固定公网IPv4；
- SSH仅允许管理员IP；
- 公网只开放22、80、443；
- PostgreSQL、Redis、MinIO管理口不对公网开放。

阿里云轻量应用服务器面向单机Web应用、开发测试和一站式域名管理，并支持大陆地域与Docker部署。[阿里云产品说明](https://help.aliyun.com/zh/simple-application-server/product-overview/product-function-node-swas-1)、[实例规格](https://help.aliyun.com/zh/simple-application-server/product-overview/instance-families/)。

### Compose服务

```text
caddy
frontend
api
worker
scheduler
streamlit
postgres
redis
minio
prometheus
grafana
```

### 安全与身份

- 管理员创建账号，禁止公开注册。
- 密码使用Argon2id哈希。
- 临时密码首次登录必须修改。
- 使用服务端不透明Session和`HttpOnly + Secure + SameSite` Cookie。
- 管理员、成员、评委三类权限。
- 评委只能运行内置样例、查看结果和下载报告。
- 所有密钥只通过服务器环境或Docker Secret注入。
- 增加密钥扫描、依赖漏洞扫描和容器镜像扫描。

### 验收

- 全部容器可启动并通过健康检查。
- Alembic可从空数据库升级和回滚一版。
- 登录、退出、首次改密码和权限隔离测试通过。
- 数据库、Redis和MinIO均不可从公网直接访问。
- CI运行pytest、Ruff、mypy、compileall、前端lint/typecheck/test/build。
- 本阶段独立提交；Codex不执行`git push`，由用户推送。

---

## 阶段2：HUST安全转换与真实数据治理，7月23日—29日

### HUST隔离转换

- 核验77个文件的来源、论文、大小和SHA-256。
- 原始pickle不进入主应用、正式服务器或A100训练环境。
- 创建一次性隔离CPU实例：
  - 不配置任何项目密钥；
  - 原始目录只读；
  - 输出目录单独挂载；
  - 关闭不必要网络；
  - 非root运行；
  - 设置CPU、内存和文件大小限制。
- 按单文件流式反序列化与转换，禁止一次加载全部数据。
- 转换输出：
  - Parquet循环记录；
  - JSON电芯元数据；
  - 文件级转换日志；
  - 原始与目标SHA-256；
  - 字段、单位和缺失情况报告。
- 转换产物重新由主项目的安全解析器验证。
- 销毁一次性实例和原始副本。

### 数据协议

- 按完整`cell_id`划分训练、验证、校准、测试。
- 分层依据为寿命分位数和放电协议。
- 预测截断点使用20、50、100、150循环。
- 训练特征不得访问截断点后的记录。
- 生成并冻结：
  - 数据集清单；
  - 数据卡；
  - 划分文件；
  - 质量报告；
  - 许可证记录；
  - 数据版本和适配器版本。

### Naumann

- 用已审核Excel/MAT布局转换真实工况记录。
- 保存温度、平均SOC、DOD、倍率、容量和阻抗信息。
- 不补造设备成本、时间成本或缺失工况。
- 缺少真实成本时，主动试验模块降级为“不含成本的候选排序”。

### 验收

- 同一电芯跨集合数量必须为零。
- 原始pickle无法被主项目加载。
- 每个Parquet文件均可追溯到原始文件哈希。
- 数据质量问题全部生成机器可读原因码。
- 数据转换在干净环境可重放。
- 正式数据卡中不把HUST描述为海辰真实工业数据。

---

## 阶段3：真实模型、区间与主动试验，7月30日—8月7日

### A100训练流程

- 本地生成安全Parquet、划分清单、配置和训练代码包。
- 上传实验室A100前运行预检脚本，报告Python、CUDA、GPU、内存和依赖。
- 不上传原始pickle、API密钥、飞书凭证或服务器密钥。
- 训练输出：
  - 模型安全制品；
  - 完整配置；
  - 指标JSON/CSV；
  - 训练日志；
  - 图表；
  - 模型卡；
  - 环境清单；
  - SHA-256。
- XGBoost使用JSON/UBJ。
- CPMLP和Hybrid优先使用safetensors；架构和特征配置单独保存为JSON。
- 禁止以pickle、joblib、`.pt`或`.pth`作为正式发布制品。

### 实验矩阵

- Dummy；
- Variance；
- XGBoost；
- CPMLP；
- Hybrid；
- Hybrid消融：
  - 无趋势项；
  - 无残差项；
  - 无拐点；
  - 无工况编码。
- 20/50/100/150循环截断对比。
- 固定电芯划分下运行多个随机种子。
- SOH、EOL80、RUL、排序和校准指标均由评测工具生成。
- 右删失样本不得伪造EOL标签。

### 不确定性与在线更新

- Split Conformal作为基线；
- Normalized Conformal作为主方法；
- 报告80%、90%、95%目标覆盖设置；
- 输出PICP、MPIW及分组覆盖结果；
- 运行20→50→100→150循环在线更新回放；
- 全局模型冻结，只更新个体轻量参数。

### Naumann与PyBaMM

- 运行随机、最大方差和EIVR主动试验回放。
- 记录每一步候选、采集策略、误差变化和来源。
- 安装锁定版PyBaMM，真实执行SPMe短周期核验。
- PyBaMM结果只标记为`SIMULATED/PHYSICS_REFERENCE`，不得进入长期寿命标签。

### 场景年份

新增受约束换算工具：

```text
循环寿命ToolResult
+ 每日等效循环
+ 运行策略版本
→ 场景年份ToolResult
```

输出必须明确标记为情景换算，不是15–25年真实验证结果。

---

## 阶段4：LLM Gateway与自主多Agent，8月8日—15日

### LLM Gateway

- 支持OpenAI兼容API。
- 启动时检测：
  - JSON Schema；
  - 原生工具调用；
  - 流式输出；
  - Embedding接口。
- 支持原生工具调用与严格JSON计划两种模式。
- 配置主模型、经济模型和无LLM模板模式。
- 每月预算100元以内。
- 实现单任务步骤、Token、重试和费用上限。
- 费用达到阈值时切换经济模型或固定流程。
- 原始数组、密钥和受限文件不发送给LLM。

### LangGraph

- Supervisor负责意图、任务拆解、角色分派和重规划。
- 四个专业Agent复用现有角色和工具白名单。
- PostgreSQL保存检查点和结构化项目记忆。
- 现有`run_constrained_workflow`仍负责最终执行。
- 高风险步骤生成`ApprovalRequest`并暂停。
- 工具失败允许最多两次重规划。
- 审批拒绝后终止受影响分支，保留已验证结果。
- LLM断线时恢复到固定生命周期工作流。

### 结构化记忆

长期保存：

- 项目目标；
- 已批准计划；
- ToolResult；
- 审批；
- 报告；
- 模型与数据版本。

短期保存：

- 原始聊天最多7天；
- 临时评委运行最多30天。

日志不保存完整提示、密钥或原始敏感数据。

### Agent可靠性实验

比较：

- 无Agent固定流程；
- 单Agent；
- 受约束多Agent；
- 无LLM降级模式。

指标由日志工具计算：

- 任务完成率；
- 工具选择正确率；
- 越权调用阻断率；
- 数值篡改阻断率；
- 证据覆盖率；
- 重规划成功率；
- 延迟与LLM成本。

LLM不能参与给自己评分。

---

## 阶段5：知识库和双前端，8月13日—20日

### 知识库

首批材料：

- LFP退化机理论文；
- HUST、Naumann和后续MATR数据论文；
- PyBaMM模型说明；
- Conformal和主动试验论文；
- 项目数据卡、模型卡、实验协议和限制说明。

处理流程：

```text
上传
→ 来源/许可证/哈希审核
→ PDF/Markdown解析
→ 保留页码分块
→ Embedding
→ pgvector + BM25召回
→ reranker
→ 带页码证据返回
```

默认每块约700 tokens，重叠约100 tokens；保留原文页码、标题和哈希。Embedding接口不可用时，降级为本地Embedding；本地模型不可用时再降级为BM25，并显示警告。

### Next.js

正式页面：

1. 登录与首次改密码；
2. 项目总览；
3. 单电芯分析入口；
4. Agent计划和实时运行时间线；
5. SOH/RUL/区间驾驶舱；
6. 在线校正对比；
7. 批次风险与复检；
8. 主动试验；
9. 知识证据；
10. 审批中心；
11. 报告与审计；
12. 管理后台。

正式网页不计算工程数值，只展示FastAPI签发结果。

### Streamlit

增强现有工作台：

- 数据质量和划分；
- 模型性能与消融；
- Conformal覆盖；
- 在线回放；
- GP工况热力图；
- PyBaMM曲线；
- ToolResult证据链。

---

## 阶段6：飞书协同、云端联调和Agent实验，8月19日—25日

### 飞书沙箱

先实现：

- 事件签名验证；
- 重复事件去重；
- 消息卡片；
- 审批动作；
- 多维表格写入；
- 失败重试和死信记录。

### 真实飞书

提供小白操作指南：

1. 创建企业自建应用；
2. 开启机器人；
3. 申请消息、多维表格和回调权限；
4. 配置HTTPS回调；
5. 保存App ID、Secret、Verification Token和Encrypt Key；
6. 配置测试群和测试表格；
7. 完成事件验证；
8. 完成审批卡片与工作流恢复。

真实密钥只放服务器Secret，不发到聊天或Git。

### 飞书行为

- 发消息创建Agent任务；
- 返回计划和运行状态卡片；
- 正式决策前发送审批卡片；
- 同步批次复检和试验推荐到多维表格；
- 返回报告链接；
- 所有操作引用同一`run_id`和`result_id`。

---

## 阶段7：冻结、材料和提交，8月26日—9月1日

### 冻结

- 8月26日起停止新增核心功能。
- 冻结代码、数据、划分、模型、Prompt、知识库和依赖版本。
- 生成数据卡、模型卡、Agent卡和系统限制清单。
- 所有失败实验和负结果保留。

### 竞赛材料

- 300字以内项目简介；
- 匿名完整项目文档；
- 技术附录；
- 系统架构图；
- 真实实验表；
- 消融和Agent可靠性实验；
- 演示视频；
- 数据、模型和许可证清单；
- 一键部署与复现说明；
- AI先锋版本强调企业业务价值；
- 华为杯版本强调算法、可信Agent与工程验证。

### 演示流程

1. 评委登录；
2. 选择内置HUST安全样例；
3. 输入自然语言诊断目标；
4. Supervisor自主生成计划；
5. 数据、寿命、物理、试验Agent执行；
6. 查看SOH轨迹、区间和在线修正；
7. 查看批次第二场景；
8. 审批试验建议；
9. 查看证据和报告；
10. 断开LLM，演示固定流程仍可运行。

### 提交日

9月1日只允许：

- 最终门禁；
- 上传；
- 下载回验；
- 文件哈希核对；
- 匿名与敏感信息检查。

禁止当天新增功能或升级依赖。

---

## 五、比赛后完整版

- MATR接入及MATR→HUST跨数据集验证；
- CORAL、DANN和目标域重校准完整实验；
- REST/MQTT/Modbus BMS模拟器；
- EMS建议接口与人工安全联锁；
- 飞书文档、任务和更完整审批流程；
- 多机部署和故障切换；
- 托管数据库评估；
- 对外注册、配额、防滥用和隐私功能；
- 更完整的性能、并发和灾难恢复测试。

模拟BMS/EMS必须持续标注为模拟，不能宣称已经接入真实设备。

---

## 六、测试与质量门禁

每项任务执行：

```text
失败测试
→ 最小实现
→ 目标测试
→ 相关集成测试
→ 全量质量门禁
→ 差异审查
→ Git提交
```

后端门禁：

```text
pytest -q
pytest tests/leakage -q
pytest tests/integration -q
pytest tests/e2e -q
ruff check .
mypy src/quanxin_life
python -m compileall -q src workbench deploy
pip-audit
```

前端门禁：

```text
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm playwright test
```

部署门禁：

```text
docker compose config
docker compose up -d
docker compose ps
健康检查脚本
数据库迁移检查
OSS备份与恢复演练
```

安全必测：

- 同一电芯不能跨集合；
- 截断点后信息不能进入特征；
- 未知pickle/joblib/pt/pth不能加载；
- 哈希不匹配必须失败；
- Agent越权工具调用必须阻断；
- LLM数字不能进入正式报告；
- PyBaMM不能产生长期RUL；
- 飞书重复事件不能创建重复任务；
- 评委不能上传任意文件或修改策略；
- LLM断线后数值流程仍能运行。

每个完成任务形成一个小提交；Codex不执行`git push`，由用户审核后推送。

---

## 七、部署与备份

### 域名规划

用户购买可用域名后使用：

```text
app.<domain>        Next.js
api.<domain>        FastAPI
lab.<domain>        Streamlit，限制团队访问
```

MinIO、PostgreSQL、Redis、Grafana不直接暴露公网。

### OSS备份

- 每日数据库加密备份；
- 每日关键对象清单；
- 每周完整关键制品备份；
- 保留日备份、周备份和月备份三个层级；
- 每月至少执行一次恢复演练；
- 备份密钥与服务器登录密钥分离。

### 监控

- Prometheus收集服务、任务、数据库和磁盘指标；
- Grafana展示运行状态；
- JSON结构化日志；
- 磁盘、队列积压、任务失败、LLM预算和备份失败产生告警；
- 日志自动轮转，不保存密钥和完整用户提示。

---

## 八、需要用户提供的外部条件

用户不需要在聊天中提供任何真实密钥。

实施到对应阶段时，通过本地或服务器Secret提供：

- AI先锋报名表最新内容；
- 团队简介和能力材料；
- 可用域名；
- 阿里云账号与服务器；
- LLM `base_url`、模型名和API Key；
- 飞书应用信息；
- 实验室A100任务提交说明；
- 允许上传到A100的安全数据路径；
- 公开知识文档；
- 批次决策和场景换算规则。

缺少某项外部条件时，系统必须使用沙箱或显式降级，不得伪造真实联调。

---

## 九、最终验收

比赛版完成必须同时满足：

- AI先锋报名已提交；
- HUST完成安全转换并可追溯；
- Naumann真实数据流水线可运行；
- 五类模型有统一真实指标；
- Hybrid有消融结果；
- Conformal有真实覆盖率；
- 在线校正和主动试验有回放；
- PyBaMM真实短期求解运行；
- LLM Supervisor与四个专业Agent可自主规划和重规划；
- LLM断线时固定流程可运行；
- PostgreSQL、Redis、MinIO和Celery真实贯通；
- 知识库返回带页码证据；
- Next.js和Streamlit可用；
- 飞书至少完成沙箱，具备凭证时完成真实联调；
- Agent可靠性实验完成；
- 云端HTTPS演示稳定；
- OSS备份恢复成功；
- 全部质量门禁通过；
- 竞赛材料、视频和复现包完成；
- 所有正式数字可追溯到有效`ToolResult`。

任何真实模型性能不预设目标数值；只有正式实验完成后才能报告结果。
