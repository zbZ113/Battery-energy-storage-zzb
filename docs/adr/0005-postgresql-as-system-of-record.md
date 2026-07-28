# ADR-0005：PostgreSQL 是正式系统记录源

- 状态：Accepted
- 日期：2026-07-28

## 背景

系统需要保存项目权限、会话、AgentStep、ToolResult、来源、active route、
calibration claim/fence 和原子提交关系。这些记录共同决定业务数字是否有效，不能分散在
缓存、日志或前端状态中。

项目曾考虑是否可以把 PostgreSQL 替换为 MySQL。虽然 SQLAlchemy 提供一定方言抽象，
当前实现、migration、依赖和测试并没有把 MySQL 声明为支持目标。

## 决策

1. PostgreSQL 是正式运行环境的 system of record；
2. Redis 只负责队列、临时租约和缓存，不保存最终业务事实；
3. 大型数据和模型字节留在受管文件系统或对象存储，数据库保存身份、URI 和哈希；
4. Alembic 是 schema 变更的唯一正式入口；
5. SQLite 只用于部分自动化测试，不构成生产支持承诺；
6. 当前比赛部署不支持把 PostgreSQL 配置替换为 MySQL；
7. 如未来支持 MySQL，必须新建 ADR，并完成方言审计、migration 分支、事务语义验证和
   全量集成测试。

## 为什么当前不能直接切换 MySQL

- 依赖明确使用 `psycopg[binary]`；
- migration 包含 PostgreSQL/pgvector 分支；
- ORM 模型导入 `pgvector.sqlalchemy.Vector`；
- migration 和 schema 测试以 PostgreSQL dialect 为正式目标；
- 比赛 Compose、secret 命名、健康检查和连接 URL 固定为 PostgreSQL；
- claim、stream head 和原子 ledger 的并发行为需要在目标方言重新验证。

把 URL 改成 MySQL 并不能证明事务、约束、索引、锁和 migration 语义等价。

## 备选方案

- MySQL：可作为未来移植目标，当前证据不足；
- SQLite：适合局部测试，不适合并发 Worker 与正式事务链；
- Redis 作为主库：不适合关系约束和审计事务；
- 文档型数据库：会增加多实体原子一致性的实现负担。

## 后果

正面：

- route、ToolResult、AgentStep 和 materialization 可以在同一事务中提交；
- Alembic 与 SQLAlchemy 的正式目标唯一；
- pgvector 可服务可选知识检索；
- 单机比赛部署运维路径清晰。

代价：

- 部署必须维护 PostgreSQL 数据卷、密码、备份和恢复；
- MySQL 用户不能在不改代码和测试的情况下直接替换；
- 对象存储和数据库仍需分别备份。

## 实现证据

- `src/quanxin_life/persistence/`；
- `migrations/`；
- `pyproject.toml` 的 `persistence` 依赖；
- `deploy/competition.compose.yaml`；
- `deploy/migrate.py`；
- migration、SQL ledger、claim 和 ORM schema 测试。

## 成熟度

- ORM、migration 和事务链：Validated；
- 目标 ECS PostgreSQL 容器和备份恢复：尚未 Deployed；
- MySQL 兼容：Planned/未承诺。
