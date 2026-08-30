# ADR-0006：比赛采用私有 ACR 和单机 ECS Compose

- 状态：Accepted
- 日期：2026-07-28

## 背景

个人比赛演示需要在有限预算、4 vCPU、8 GiB 内存和 40 GB 系统盘上运行完整产品。
目标服务器位于阿里云杭州，访问 Docker Hub 不稳定，但同地域 ACR 可访问。

本阶段需要证明端到端系统能运行，不需要提前引入 Kubernetes、多可用区或托管服务。

## 决策

1. GitHub Actions 构建比赛镜像并发布到杭州私有 ACR；
2. 发布服务为 PostgreSQL、Redis、backend、frontend 和 Nginx 五类镜像；
3. 部署使用 `deploy/competition.compose.yaml`；
4. PostgreSQL 和 Redis 只进入内部 backend 网络，不映射公网端口；
5. Nginx 是唯一公网入口，开放 80/443；
6. HTTPS 使用 Let’s Encrypt 证书，私钥只挂载到 gateway；
7. API 与 Worker 共用 backend 镜像和同一组合根；
8. migration 是一次性前置服务，成功后 API/Worker 才启动；
9. secret 由 root-owned 文件提供，不进入 Git、镜像或聊天记录；
10. 模型、数据、策略、注册表和 calibration evidence 使用显式受管挂载；
11. 正式部署记录使用镜像 digest，不只依赖可变 tag。
12. Gateway 配置固化在 `quanxin-nginx` 镜像中，release workflow 拒绝覆盖已存在 tag。

## 资源策略

Compose 对各服务设置资源上限，Worker 并发为 1。该选择优先保证比赛演示稳定和可恢复，
不代表吞吐优化结果。

后端容器默认使用：

- `read_only: true`；
- `no-new-privileges:true`；
- 受限 tmpfs；
- 内部服务网络；
- healthcheck 与显式启动依赖。

## 备选方案

- Docker Hub：目标 ECS 连接不稳定，拒绝作为主发布源；
- 在 ECS 现场构建：浪费磁盘和时间，且难以冻结供应链，拒绝；
- Kubernetes：超出个人比赛需求和资源预算；
- 所有依赖安装到宿主机：难以复现和回滚；
- 公网暴露 PostgreSQL/Redis：扩大攻击面，拒绝。

## 后果

正面：

- 构建与运行分离，ECS 只拉取审核后的镜像；
- 同地域 ACR 降低网络不确定性；
- Compose 足以演示真实数据库、队列、Worker、API 和 UI；
- 镜像 digest、artifact SHA-256 和数据库审计形成分层来源。

代价：

- 单机故障会中断服务；
- 40 GB 磁盘需要清理旧镜像和建立备份策略；
- Let’s Encrypt IP 短期证书需要可靠续期；
- ACR 凭据和 GitHub secrets 需要用户在受控位置维护。

## 实现证据

- `.github/workflows/publish-acr.yml`；
- `deploy/competition.compose.yaml`；
- `deploy/Dockerfile.backend`、`frontend/Dockerfile` 与 `deploy/Dockerfile.gateway`；
- workflow 中固定的 pgvector/PostgreSQL、Redis 与 Gateway 基础镜像版本；
- `deploy/nginx/competition.conf`；
- `deploy/runtime_settings.py`；
- `deploy/competition_runtime.py`；
- 团队本地维护的 ACR release 与竞赛部署契约回归记录（不随 Git 分发）。

## 成熟度

- 发布 workflow 与 Compose：Implemented/Validated（配置契约）；
- ACR 镜像成功发布与 digest：待 Published；
- ECS 服务健康：待 Deployed；
- 浏览器 RUL/SOH/Agent/报告闭环：待 Demonstrated；
- HA、自动扩缩容和异地恢复：Planned，且不阻塞比赛演示。
