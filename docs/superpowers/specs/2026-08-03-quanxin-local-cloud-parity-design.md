# 泉芯智寿本地云端同构运行设计

- 日期：2026-08-03
- 状态：用户已于 2026-08-03 确认书面规格，进入实施
- 适用目标：`docs/superpowers/plans/2026-08-02-quanxin-productization-master-plan.md`
- 本阶段目标：先在本机浏览器完成真实九步电芯寿命分析闭环，再把同一套数据库、队列、制品和部署边界迁移到新的 ECS

## 1. 目标与成功标准

本阶段不是制作假数据演示，也不是只启动一个能返回健康检查的 API。完成后，用户应能在本机浏览器中完成：

`登录 -> 项目 -> MATR 内置样例 -> 电芯与 cutoff -> 启动固定九步 Agent -> 实时进度 -> RUL/SOH/Conformal -> 审计报告 -> 下载`

首轮本地验收使用仓库已有、经过来源和 SHA-256 核验的 MATR 数据、模型、校准证据和 ToolResult 链。页面不得硬编码或补造 SOH、RUL、Conformal、阈值比较及其他业务数值。每个显示值必须能从页面追溯到 API、PostgreSQL 记录和有效 `ToolResult`。

本阶段的数据库与队列基线必须与未来云端一致：PostgreSQL 16、Redis 7.4 AOF、Celery Worker、FastAPI Runtime V7、Next.js 前端。SQLite、内存队列、Mock ToolResult 和 `deploy/compose.yaml` 的 foundation API 不能作为验收结论。

## 2. 已确认方案与不采用方案

采用 Docker Desktop + WSL2 + Docker Compose。全部可迁移的应用、Linux 数据盘、镜像缓存、构建缓存和运行卷优先放在 D 盘。Windows 可选功能、WSL 引导组件、少量 Docker Desktop 用户配置如果由 Windows 固定写入 C 盘，则保留在 C 盘，不手工搬移系统文件。

不采用以下方案：

- 不以原生 Windows PostgreSQL/Redis 作为主验收环境；
- 不以 SQLite、内存仓库或同步假 Worker 替代真实基础设施；
- 不清空、覆盖或重新生成已经核验的数据和模型制品；
- 不把本地 HTTP 例外扩展到生产运行时；
- 不要求第一轮验收先提供新企业数据，现有 MATR 资产足够；
- 不把 PyBaMM 当作长期寿命标签或替代真实数据。

## 3. D 盘布局

推荐固定使用以下 operator-owned 目录，避免散落在仓库和用户配置目录：

```text
D:\QuanxinRuntime\
  docker-desktop\        Docker Desktop 可迁移安装文件
  docker-wsl\            Docker Desktop WSL 数据发行版与 Linux 磁盘
  docker-windows\        Windows 容器数据（即使当前不用，也固定到 D 盘）
  compose\               本地运行生成的非源码配置
  volumes\postgres\      PostgreSQL 持久卷
  volumes\redis\         Redis AOF 持久卷
  runtime\data\          冻结 record batch
  runtime\artifacts\     运行制品
  runtime\policies\      版本化 Agent policy
  runtime\registry\      SHA-256 固定的 deployment registry
  runtime\calibration\   校准证据
  runtime\config\        非秘密运行配置
  secrets\               仅机器读取的服务秘密，仓库外、限制 ACL
  backups\               本地备份与恢复演练结果
  logs\                   可轮转运行日志，不记录密钥或原始敏感数据
```

安装前检查 D 盘至少保留 100 GiB 可用空间。拉镜像和构建前再次检查预计新增空间，若低于 60 GiB 则停止，不重复 ECS 上“旧镜像与新镜像共存导致磁盘爆满”的失败路径。

## 4. Windows、WSL2 与 Docker 安装边界

安装顺序如下：

1. 启用 Windows Subsystem for Linux 与 Virtual Machine Platform；这部分属于 Windows 系统组件，允许落在 C 盘。
2. 安装或更新 WSL2 内核，不额外安装无关 Linux 发行版。
3. 使用 Docker Desktop 官方安装器，把应用目录、WSL 数据根和 Windows 容器数据根分别指向 `D:\QuanxinRuntime\docker-desktop`、`D:\QuanxinRuntime\docker-wsl`、`D:\QuanxinRuntime\docker-windows`。
4. Docker Desktop 使用 WSL2 backend，关闭不需要的 Kubernetes，设置磁盘镜像和构建缓存上限。
5. 安装后用 `docker info`、`docker compose version`、实际数据根和 D/C 盘空间验证，不仅检查图形界面是否打开。

如果 Windows 要求重启，安装流程在重启前停止并记录准确续跑点；不能把“安装命令已执行”误报为 Docker 已可用。

卸载或回退时先停止 Compose，再保留 `volumes`、`runtime` 和 `backups`，然后卸载 Docker Desktop。不得直接删除 WSL 虚拟磁盘；只有在确认备份可恢复且用户明确授权后才删除运行数据。

## 5. 本地 Compose 拓扑

新增本地专用 Compose 入口，不修改 `deploy/competition.compose.yaml` 的生产安全含义。服务边界为：

```text
Browser
  -> local gateway (HTTP loopback only)
      -> Next.js frontend
      -> FastAPI local competition entrypoint
          -> PostgreSQL 16 + pgvector
          -> Redis 7.4 AOF
          -> Celery agent-runs / advanced-calibration worker
          -> verified MATR record batches
          -> SHA-verified model registry and calibration evidence
```

本地 backend 镜像复用 `deploy/Dockerfile.backend`。frontend 使用独立 local build target 或 `frontend/Dockerfile.local`，gateway 使用独立 `deploy/nginx/local.conf` 和 local build target；两者不得覆盖或替换生产 Dockerfile/config。生产 gateway 继续执行 HTTP 到 HTTPS 跳转并加载正式证书，本地 gateway 只绑定 loopback HTTP，不读取生产证书。

PostgreSQL 来源固定为 `pgvector/pgvector:0.8.1-pg16`，Redis 来源固定为 `redis:7.4.2-alpine`，与发布流水线一致。首次拉取后记录实际 manifest digest，生成本地 digest lock；后续 Compose 按 digest 使用，不能只依赖可漂移 tag。backend、frontend 和 gateway 本地构建同样记录 image ID、源码 HEAD 和构建时间。Compose 必须具备 healthcheck、migration 一次性任务、只读运行文件系统、最小权限、明确卷、Redis AOF 和有界日志。

本地只发布一个 loopback 网关端口。PostgreSQL、Redis、API 和 Worker 默认不对局域网发布；需要调试时使用 `docker compose exec`，不能把数据库端口长期暴露到所有网卡。

## 6. 本地 HTTP 与生产 HTTPS 的隔离

生产 `CompetitionRuntimeSettings`、`deploy/competition_api.py`、`deploy/competition_worker.py` 和 `frontend/Dockerfile` 继续要求 HTTPS，不放宽生产校验。

本地运行新增显式、独立的 local profile：

- 后端新增 `LocalCompetitionRuntimeSettings` 和 local API/Worker entrypoint，或使用等价的显式受限 assembly 参数；生产 entrypoint 只能构造 `CompetitionRuntimeSettings`，不得读取 local profile；
- 后端只接受 `http://localhost:<port>`、`http://127.0.0.1:<port>` 或 IPv6 loopback 作为可信 Origin；
- local profile 使用 development cookie 名和非 Secure cookie，但仍保留 HttpOnly、SameSite、Origin 校验和精确 CORS allowlist；
- frontend local build 使用单独的构建期常量，例如 `NEXT_PUBLIC_RUNTIME_PROFILE=local`；`frontend/Dockerfile.local` 的校验和 `resolveApiBaseUrl` 必须同时要求该常量与 loopback HTTP，不能仅依赖 `NODE_ENV`；
- 生产 `frontend/Dockerfile` 必须拒绝 local profile，生产 `resolveApiBaseUrl` 仍拒绝所有 HTTP；
- 本地入口只绑定 `127.0.0.1`，不能因 local profile 允许局域网或公网明文访问；
- 生产部署测试必须证明 production entrypoint 不会加载 local profile。

这样可以在本机无证书完成浏览器验收，同时不把 ECS 的严格 TLS 要求降级。

## 7. 秘密与管理员初始化

任何用户密码都不写入仓库、计划、日志、命令历史或聊天。需要管理员临时密码时，由用户在交互式隐藏提示中输入；Codex/LLM 不读取、不回显该值。隐藏输入 wrapper 只把临时密码交给授权的 bootstrap 进程，bootstrap 命令结束后立即删除临时文件。初始化完成后要求首次登录立即改密。

PostgreSQL 与 Redis 的服务凭据由本地初始化脚本随机生成，不打印内容，只写入仓库外 `D:\QuanxinRuntime\secrets`，并限制为当前 Windows 用户和 Docker 所需进程可读。这里保存的是运行服务必须持久存在的机器凭据；Codex/LLM 不读取、不输出其内容，获授权的 runtime/Compose 进程只为建立连接读取且绝不记录。未来迁移到 ECS 时重新生成，不能复制本机凭据。

如果无法在 Windows/Compose 下保证秘密文件 ACL，则本地运行必须停止，不能退化为把密码写进 `.env`、Compose YAML 或命令行参数。

## 8. 数据、模型与样例引导

首轮验收使用 MATR，不要求用户提供新文件：

- 三批原始清单、140 个逐电芯 Parquet、监督轨迹和 `cell_id` 固定划分已存在且哈希一致；
- 产品样例使用已验证的 `MATR_b3c34`，冻结 cutoff 20/50/100/150；
- 启动脚本只复制或绑定已经核验的非秘密资产，并重新验证 SHA-256；
- deployment registry、模型、normalizer、校准来源和 Agent policy 任一缺失或哈希不符时 fail closed；
- 数据引导只创建项目、数据集、record batch 绑定和必要的校准物化，不预先写入本轮预测结果；
- 每次用户主动启动九步 Agent 都创建新的 run，真实 Worker 生成新的业务 ToolResult。

HUST 尚未完成隔离转换与 `cell_id` 划分，不能进入普通产品运行时。Naumann 是主动试验工况数据，不直接替代 MATR 的 RUL/SOH 链。两者不阻塞首轮本地验收。

## 9. PyBaMM 边界

现有 PyBaMM SPMe 短时求解和物理 ToolResult 工具仍保留，但 Runtime V7 的固定九步尚未接入它。本阶段不暗改九步身份，也不让 PyBaMM 生成长期退化标签。

完成主链后，PyBaMM 作为独立的“可选短时物理核验”子项目接入：输出标为 `PHYSICS_REFERENCE`/`SIMULATED`，使用独立 Worker 队列，失败时不画伪曲线，报告仅作为附录引用且不改变 RUL/SOH/Conformal。若未来要成为主 Agent 自动步骤，则升版 Runtime V8，而不是把第十步偷偷塞入 V7。

## 10. 迁移、引导与运行顺序

本地启动必须可重复执行，并按以下顺序失败关闭：

1. 检查 Docker/WSL、D 盘空间、端口和 operator 目录权限；
2. 生成或验证服务秘密，不输出内容；
3. 拉取基础镜像、记录 digest lock，并构建和固定本地镜像身份；
4. 启动 PostgreSQL 与 Redis，等待健康；
5. 执行 Alembic migration，并验证数据库 head；
6. 交互式创建唯一 bootstrap ADMIN；
7. 验证 deployment registry、模型、数据和校准证据；
8. 引导 MATR 样例项目与四个 cutoff；
9. 启动 API、Worker、frontend 和 gateway；
10. 运行 API smoke、Worker smoke 和浏览器纵向验收。

任一步失败都保留日志和数据库，不自动清库重来。重复启动要复用已有卷和已初始化身份，不能重复创建管理员或重复冻结样例。

## 11. 浏览器纵向验收

验收至少执行一次全新的 cutoff 运行，并保留非敏感证据：

1. 登录并完成强制改密；
2. 打开项目工作台，选择 `MATR_b3c34` 和一个 cutoff；
3. 启动新的固定九步 Agent；
4. 观察 SSE/轮询状态从排队到九步终态，并记录每步来源状态；
5. 查看 RUL、SOH、Conformal 和审计报告；
6. 收集页面显示的 ToolResult ID；
7. 用 API、PostgreSQL 和证据文件逐项对账相同 ID、输入哈希、模型/数据/特征版本和上游关系；
8. 验证页面未显示无 ToolResult 支撑的业务数值；
9. 验证下载能力当前真实可用的格式；尚未完成的 PDF/Word/ZIP 必须显示明确不可用，不伪造文件；
10. 验证刷新、断线重连、失败态、移动端布局和日志中无密码/完整提示/敏感原始数据。

第一轮可以只证明内置 MATR 样例的真实预测。Task 5 的 CSV/Parquet/ZIP、多电芯、断点续传、MinIO 和四 cutoff 自动冻结，以及 Task 6 的完整 PDF/Word/ZIP 下载，仍按总计划分别实现和验收，不能因为主链跑通就宣布整个平台完成。

## 12. 测试与门禁

行为变更严格测试先行。至少包括：

- local settings 仅接受 loopback HTTP，拒绝局域网/公网 HTTP；
- production settings 和 production frontend 仍拒绝 HTTP；
- local cookie、Origin、CORS 和 SSE 在真实浏览器中工作；
- Compose config、healthcheck、migration、幂等 bootstrap 和 volume persistence；
- Redis AOF 重启后队列/持久数据边界符合预期；
- backend 镜像可导入真实模型依赖并运行 Runtime V7；
- PostgreSQL 真实九步 Agent 与 ToolResult 外键顺序；
- frontend Vitest、ESLint、TypeScript、Next.js build；
- Python 定向 pytest、完整 pytest、Ruff、mypy、compileall；
- 浏览器截图与数据库/API/ToolResult 纵向对账。

没有最新命令输出和真实浏览器证据，不得声称本地预测链完成。

## 13. 与未来 ECS 的关系

本地不是另一套产品。未来新 ECS 继续使用同一 backend/frontend/gateway Dockerfile、同一 migration、同一 Runtime V7、同一 PostgreSQL/Redis 版本和同一 operator 目录语义。差异仅限：

- 本地使用 loopback HTTP local profile；ECS 使用 production profile、域名和严格 TLS；
- 本地镜像从工作区构建；ECS 只拉取 ACR 不可变 tag/digest；
- 本地卷在 `D:\QuanxinRuntime`；ECS 卷在独立云盘 operator 目录；
- 本地服务秘密与 ECS 服务秘密独立生成；
- ECS 增加证书续期、云盘快照、远端备份、告警和回滚演练。

新的 ECS 建议至少 80 GiB，优先 100 GiB 系统/数据容量，并在 pull 前执行磁盘门禁。发布时先备份，再拉取新镜像，验证后清理明确不再使用的旧 release；不能在空间不足时让两套大模型镜像无检查共存。

## 14. 实施边界与任务拆分

本设计落地为三个连续、各自可验收的实施段：

1. 本地运行底座：安装 Docker/WSL，新增 local settings、Compose、初始化脚本和测试，拉起 PostgreSQL/Redis/API/Worker/frontend。
2. 真实主链验收：引导 MATR 样例，完成浏览器九步预测和 ToolResult 纵向对账；发现的产品缺口按测试先行修复。
3. 总计划后续：完成 Task 5 上传、Task 6 导出、Task 7 完整浏览器验收，再把同一 release 按 Task 8 部署到新 ECS。

本地底座不会通过重写算法“修好”结果，也不会把 HUST、Naumann 或 PyBaMM 强塞进首轮运行。它的作用是先建立一个可观察、可重复、与未来云端一致的真实产品验收环境，让后续每个功能都能在浏览器、API、Worker 和数据库四层同时验证。

## 15. 当前用户需要提供的内容

本地首轮不需要提供数据集、模型、ECS、域名或云账号。需要用户参与的只有：

- Windows 出现管理员/UAC 或重启提示时确认；
- bootstrap ADMIN 创建时，在隐藏交互提示中自行输入临时密码；
- 首次登录时自行设置新密码；
- 浏览器需要人工确认产品展示取舍时作选择。

后续部署到新 ECS 时，用户再提供新实例的连接入口、云盘和安全组控制权，并在交互提示中自行输入云端凭据。任何密钥信息都不进入仓库、文档或聊天。
