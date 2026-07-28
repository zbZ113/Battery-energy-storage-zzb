# ADR-0004：使用 active route 与 READY calibration materialization

- 状态：Accepted
- 日期：2026-07-28

## 背景

聚合实验排名不能唯一确定线上执行的 checkpoint。RUL 与 SOH 还有不同优化角色，
Conformal 校准又必须与当前模型、cutoff、目标和项目上下文一致。若客户端自行提交候选、
路径或校准数组，就会绕过审批和数据隔离。

## 决策

### Active route

1. 候选先导入 managed artifact catalog；
2. ADMIN 通过追加式 activation event 激活或回退；
3. 路由身份包含 task、cutoff、role、candidate、seed 和 artifact；
4. 当前 route 使用 stream head 解析，历史事件不覆盖；
5. prediction 和 Agent 每次从服务端读取当前 route。

### Calibration materialization

1. ADMIN 只提交项目、任务、cutoff、role 和幂等身份；
2. Redis 只接收 `materialization_id`；
3. Worker 从 PostgreSQL 重建 route、evidence、模型和项目上下文；
4. Worker 使用 lease 和递增 fence claim；
5. `ToolResult`、项目绑定、样本绑定和 `READY` 状态原子提交；
6. route 或上下文变化后，旧 `READY` 记录转为 `STALE`；
7. Agent 只能解析一个与当前 route 精确匹配的 `READY` 记录。

## 备选方案

- 自动选择验证集第一名：绕过人工责任，拒绝；
- 每次预测临时计算校准数组：耗时且难以审计，拒绝；
- 将校准样本放进 Celery 消息：消息体可篡改和过期，拒绝；
- 旧 Worker 无条件写回：会覆盖恢复后的新结果，拒绝。

## 后果

正面：

- 研究结论与线上 checkpoint 之间有明确审批边界；
- 路由可回退且历史完整；
- 校准区间与模型、目标和 cutoff 精确绑定；
- Worker 崩溃、超时和重试不会产生双写。

代价：

- 路由变化需要重新物化 calibration evidence；
- ADMIN 工作流比自动部署多一步；
- 需要 PostgreSQL 事务、Redis 队列和后台 Worker 协同。

## 实现证据

- `src/quanxin_life/application/model_route_activation.py`；
- `src/quanxin_life/application/advanced_runtime.py`；
- `src/quanxin_life/application/advanced_calibration_jobs.py`；
- `src/quanxin_life/application/advanced_calibration_materialization.py`；
- `src/quanxin_life/application/advanced_agent_execution_context.py`；
- `src/quanxin_life/infrastructure/calibration_queue.py`；
- `src/quanxin_life/tasks/advanced_calibration.py`；
- 路由、materialization、claim/recovery、API 和 Agent resolver 测试。

## 成熟度

- 服务、数据库、API、Worker 与 UI：Validated；
- 真实比赛项目的 route 和 calibration evidence：尚未 Deployed；
- 多副本 Worker 压力与故障注入：Planned。
