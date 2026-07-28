# 竞赛部署运维 Runbook

## 使用范围

本 Runbook 面向单节点个人比赛演示。执行任何会改变数据库、route、calibration
evidence 或镜像版本的操作前，先记录当前 release tag、五个镜像 digest、数据库
备份和 deployment registry ID。

## 每日检查

```bash
cd /srv/quanxin/deploy

docker compose \
  --env-file competition.env \
  -f competition.compose.yaml \
  ps

docker system df
df -h /
free -h
```

检查最近错误时不得把完整 secret 或用户输入贴到公开渠道：

```bash
docker compose \
  --env-file competition.env \
  -f competition.compose.yaml \
  logs --no-color --since 30m api worker gateway \
  | grep -E 'ERROR|Traceback|FAILED|stale|expired' \
  | tail -100
```

## ADMIN 初始化 override

主 Compose 故意不长期声明 `bootstrap_admin_password`。首次初始化时在服务器创建
`bootstrap-admin.override.yaml`：

```yaml
services:
  api:
    secrets:
      - bootstrap_admin_password

secrets:
  bootstrap_admin_password:
    file: /etc/quanxin/secrets/bootstrap_admin_password
```

权限设为 `0600`，然后执行：

```bash
docker compose \
  --env-file competition.env \
  -f competition.compose.yaml \
  -f bootstrap-admin.override.yaml \
  run --rm \
  -e QUANXIN_BOOTSTRAP_ADMIN_USERNAME=<ADMIN 用户名> \
  -e QUANXIN_BOOTSTRAP_ADMIN_PASSWORD_FILE=/run/secrets/bootstrap_admin_password \
  api python -m deploy.bootstrap_admin
```

确认 ADMIN 首次登录并完成强制改密后：

```bash
sudo rm -f /etc/quanxin/secrets/bootstrap_admin_password
rm -f bootstrap-admin.override.yaml
```

## 服务重启

重启单个无状态服务：

```bash
docker compose \
  --env-file competition.env \
  -f competition.compose.yaml \
  restart api worker frontend gateway
```

重启后必须检查：

- PostgreSQL/Redis healthy；
- migration service 仍为成功完成；
- API/gateway health；
- Worker 正在消费正确队列；
- READY materialization 和既有 `ToolResult` 仍可读取；
- 旧 fenced claim 不能覆盖新 lease。

## 数据库备份

备份文件写入宿主机受控目录，不进入容器镜像：

```bash
sudo install -d -o quanxin -g quanxin -m 0750 /srv/quanxin/backups
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

docker compose \
  --env-file competition.env \
  -f competition.compose.yaml \
  exec -T postgres \
  pg_dump -U quanxin -d quanxin -Fc \
  > "/srv/quanxin/backups/postgres-${STAMP}.dump"

sha256sum "/srv/quanxin/backups/postgres-${STAMP}.dump" \
  > "/srv/quanxin/backups/postgres-${STAMP}.dump.sha256"
```

恢复演练必须在独立临时数据库或独立测试主机进行。未经备份和明确批准，不得覆盖
当前比赛数据库。

Redis AOF 用于任务状态和队列恢复，但不能替代 PostgreSQL 业务账本备份。备份 Redis
前应先暂停相关写入并使用 Redis 官方一致性流程。

## 发布新版本

1. 在 GitHub 手动发布新的不可变 `release_tag`；
2. 保存 source commit 和五个 digest；
3. 备份 PostgreSQL；
4. 更新 `competition.env` 中的 `RELEASE_TAG`；
5. `docker compose pull`；
6. `docker compose config --quiet`；
7. `docker compose up -d`；
8. 检查 migration、health、Worker、登录和真实纵向冒烟；
9. 更新 `docs/status.md` 的实际部署状态。

不得覆盖旧 tag，也不得使用 `latest`。

## 回滚

回滚只允许切换到已记录 digest 的旧 release：

1. 停止新请求；
2. 评估新 migration 是否向后兼容；
3. 恢复对应数据库备份或执行经过审查的降级方案；
4. 将 `RELEASE_TAG` 改回旧版本；
5. 拉取并启动；
6. 重新验证 route、calibration、ToolResult 和报告；
7. 记录失败版本、原因和恢复证据。

如果 migration 不可逆，单纯切换镜像 tag 不是安全回滚。

## TLS 续期

当前 `quanxin-ecs-ip` 首次证书是通过 standalone authenticator 取得的。standalone
续期需要独占 80 端口，而运行中的 gateway 已经占用该端口。不能直接假设
`certbot renew` 会自动改用 Nginx 中已经配置的 webroot。

部署 gateway 后，先确认现有 lineage：

```bash
sudo grep -E '^(authenticator|preferred_profile|webroot_path)' \
  /etc/letsencrypt/renewal/quanxin-ecs-ip.conf
```

然后验证 ACME webroot 确实能经公网 HTTP 访问。先在 ECS 创建固定测试文件：

```bash
TOKEN="quanxin-renewal-check"
sudo install -d -m 0755 /srv/quanxin/acme/.well-known/acme-challenge
printf 'ok\n' \
  | sudo tee "/srv/quanxin/acme/.well-known/acme-challenge/${TOKEN}" \
  >/dev/null
```

从本机或另一条外部网络访问；部分云网络不支持 ECS 通过自身公网 NAT IP hairpin
回访，所以不要只在 ECS 本机执行该检查：

```powershell
curl.exe --fail --silent --show-error `
  "http://<PUBLIC_HOST>/.well-known/acme-challenge/quanxin-renewal-check"
```

必须返回 `ok`。然后回到 ECS 删除测试文件：

```bash
sudo rm -f "/srv/quanxin/acme/.well-known/acme-challenge/${TOKEN}"
```

只有返回 `ok` 后，才把该 lineage 重新配置为 webroot。Certbot 5.7 的
`reconfigure` 会先对 staging 执行测试续期，成功才保存新选项：

```bash
sudo certbot reconfigure \
  --cert-name quanxin-ecs-ip \
  --webroot \
  --webroot-path /srv/quanxin/acme
```

再次检查，必须看到 `authenticator = webroot`，然后测试未来续期：

```bash
sudo grep -E '^(authenticator|preferred_profile|webroot_path)' \
  /etc/letsencrypt/renewal/quanxin-ecs-ip.conf

sudo certbot renew --cert-name quanxin-ecs-ip --dry-run
```

为成功续期增加 gateway reload hook。先确认 Docker 的绝对路径；以下脚本假设为
`/usr/bin/docker`：

```bash
command -v docker

sudo tee /etc/letsencrypt/renewal-hooks/deploy/quanxin-gateway-reload.sh \
  >/dev/null <<'SH'
#!/bin/sh
set -eu
/usr/bin/docker compose \
  --env-file /srv/quanxin/deploy/competition.env \
  -f /srv/quanxin/deploy/competition.compose.yaml \
  exec -T gateway nginx -s reload
SH

sudo chmod 0750 \
  /etc/letsencrypt/renewal-hooks/deploy/quanxin-gateway-reload.sh

sudo certbot renew \
  --cert-name quanxin-ecs-ip \
  --dry-run \
  --run-deploy-hooks

docker compose \
  --env-file /srv/quanxin/deploy/competition.env \
  -f /srv/quanxin/deploy/competition.compose.yaml \
  ps gateway
```

带 `--run-deploy-hooks` 的 dry-run 必须成功，且 gateway 仍为 healthy。之后才允许依赖
snap 的 `certbot.renew` timer。查看 timer 与日志：

```bash
sudo snap services certbot
sudo journalctl -u snap.certbot.renew.service --no-pager -n 100
```

验证：

```bash
echo | openssl s_client -connect <PUBLIC_HOST>:443 -servername <PUBLIC_HOST> 2>/dev/null \
  | openssl x509 -noout -issuer -dates -ext subjectAltName
```

若 lineage 仍为 standalone，必须先停 gateway 或安装经过审查的 pre/post hooks；
不得在 80 端口被占用时直接等待自动续期。Certbot 官方对 webroot、renewal options
和 hooks 的说明见
[Certbot User Guide](https://eff-certbot.readthedocs.io/en/stable/using.html)。

## 磁盘容量

40 GB 系统盘不保存完整训练结果。重点监控：

```text
/var/lib/docker
/srv/quanxin/postgres
/srv/quanxin/redis
/srv/quanxin/artifacts
/srv/quanxin/backups
```

清理前先用 `docker system df` 查明对象。不得用未经核对的递归删除命令清理
deployment registry、calibration evidence、数据库或备份。

## 故障分级

| 现象 | 首查 | 允许动作 |
| --- | --- | --- |
| HTTPS 不通 | ECS 安全组、UFW、gateway、证书 | 修复网络或 gateway，不开放数据库端口 |
| API 不健康 | migrate、PostgreSQL、运行输入 | 修复根因，不切换 foundation API 冒充 |
| Worker 不消费 | Redis ACL、队列名、Worker 日志 | 重启 Worker，保留 claim/lease 证据 |
| materialization STALE | active route 是否变化 | 重新创建，不手工改 READY |
| 模型加载拒绝 | registry ID、manifest、SHA-256 | 重新传输/登记，不跳过校验 |
| 磁盘不足 | Docker、备份、日志占用 | 先归档和核对，再清理可回收对象 |

## 演示前验收

- 五个镜像 digest 与发布记录一致；
- 所有容器达到预期状态；
- migrations 到 `head`；
- ADMIN 已改密；
- 项目和 15 个候选存在；
- route 激活有 ADMIN 审批记录；
- calibration materialization 为 READY；
- RUL、SOH、Conformal、Agent 和报告均来自登记 `result_id`；
- 页面刷新和服务重启后结果仍存在；
- route 回退和 STALE evidence 能失败关闭；
- 数据库备份及 SHA-256 已生成；
- 证书有效期足够覆盖比赛日期。
