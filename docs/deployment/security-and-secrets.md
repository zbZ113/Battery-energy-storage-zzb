# 部署安全与 Secrets

## 原则

竞赛部署使用“身份可进环境文件、秘密只进 secret file”的边界：

- `competition.env` 可以保存 registry、namespace、release tag、目录和公开 Origin；
- 密码、ACL、数据库 URL、Redis URL、私钥不得写入环境模板、Git、聊天或截图；
- secret file 必须是绝对路径下的普通文件，不允许符号链接；
- 后端读取的每个 secret 必须只有一行、非空、无 NUL，且不超过 8 KiB；
- 日志不得输出 secret 内容。

## 必需 secrets

`deploy/competition.compose.yaml` 使用：

| 文件 | 读取方 | 内容 |
| --- | --- | --- |
| `postgres_password` | PostgreSQL | 数据库用户密码 |
| `database_url` | migrate/API/Worker | `postgresql+psycopg://...` |
| `redis_password` | Redis healthcheck | Redis 应用用户密码 |
| `redis_acl` | Redis | Redis ACL 配置 |
| `redis_url` | API/Worker | 带认证信息的 Redis URL |
| `feishu_app_secret` | migrate/API/Worker | 飞书企业自建应用 App Secret |
| `feishu_verification_token` | migrate/API/Worker | 飞书事件 verification token |
| `feishu_encrypt_key` | migrate/API/Worker | 飞书加密事件 Encrypt Key |
| `aily_connector_api_key` | migrate/API/Worker | Aily OpenAPI Bearer key |

首次 ADMIN 初始化还需要临时的 `bootstrap_admin_password`。它只用于一次性 override，
不能长期挂载到 API 或 Worker。

## 在 ECS 本地创建

先固定权限：

```bash
sudo install -d -o root -g quanxin -m 0750 /etc/quanxin/secrets
sudo -u root sh -c 'umask 077; : > /etc/quanxin/secrets/.permission-check'
sudo rm -f /etc/quanxin/secrets/.permission-check
```

推荐用服务器本地脚本生成独立高熵秘密并同时构造 URL。脚本不得打印秘密：

```bash
sudo python3 - <<'PY'
import secrets
from pathlib import Path
from urllib.parse import quote

root = Path("/etc/quanxin/secrets")
root.mkdir(parents=True, exist_ok=True)

postgres = secrets.token_urlsafe(48)
redis = secrets.token_urlsafe(48)
admin = secrets.token_urlsafe(48)

files = {
    "postgres_password": postgres + "\n",
    "database_url": (
        "postgresql+psycopg://quanxin:"
        f"{quote(postgres, safe='')}@postgres:5432/quanxin\n"
    ),
    "redis_password": redis + "\n",
    "redis_acl": (
        "user default off\n"
        f"user quanxin on >{redis} ~* &* +@all\n"
    ),
    "redis_url": f"redis://quanxin:{quote(redis, safe='')}@redis:6379/0\n",
    "bootstrap_admin_password": admin + "\n",
}

for name, value in files.items():
    path = root / name
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
PY
```

飞书/Aily 的四个值来自已批准的目标租户和 Aily 项目，不能用上述随机生成脚本替代。
在服务器本地分别写入 `feishu_app_secret`、`feishu_verification_token`、
`feishu_encrypt_key`、`aily_connector_api_key`，每个文件只含一行真实值并设置 `0600`；
不要在命令行、shell history、截图或聊天中回显内容。

创建后只检查权限、所有者和大小，不显示内容：

```bash
sudo find /etc/quanxin/secrets \
  -maxdepth 1 -type f \
  -printf '%f owner=%u:%g mode=%m size=%s\n'
```

## 版本化非秘密配置

### Agent policy

`/srv/quanxin/policies/advanced-agent.json`：

```json
{
  "schema_version": "advanced-agent-policy-v1",
  "conformal_alpha": 0.1
}
```

`conformal_alpha` 是服务端策略，不应由浏览器为每次调用随意提交。改变该文件需要
评审、重新计算 SHA-256 并更新部署记录。

### Calibration source registry

`/srv/quanxin/config/calibration-sources.json` 只保存登记身份和相对目录：

```json
{
  "schema_version": "advanced-calibration-source-registry-v1",
  "sources": [
    {
      "registration_id": "<受审查登记 ID>",
      "evidence_relative_root": "<相对 calibration evidence 根的目录>",
      "three_batch_manifest_sha256": "<64 位 SHA-256>"
    }
  ]
}
```

API 不接受客户端提交 calibration 样本数组、服务器路径、哈希或 sample ID。Worker
根据 `materialization_id` 从数据库和上述受管登记中重建上下文。

### Feishu canonical CSV registry

`/srv/quanxin/config/feishu-csv-registrations.json` 是非秘密但受版本和权限控制的输入。
从 `deploy/feishu-csv-registrations.example.json` 的空注册表开始，为每个经人工核验的
canonical CSV 添加一条 entry。每条 entry 至少包含：

- 顶层 `payload_sha256`；
- `registration.metadata` 中的真实 `cell_id`、chemistry、标称容量、来源 URI、
  `metadata.source_sha256` 和 schema version；
- `feature_config`、`data_version` 与 `split_version`；
- 至少一条 `source_kind=OBSERVED` 的 provenance。

同一文件字节的 SHA-256 必须同时用于 `payload_sha256`、`metadata.source_sha256` 和该
`OBSERVED` provenance 的 `sha256`。不得把 3Ah 元数据登记为 250/280Ah 方形电芯，
不得从文件名或 Aily 文本推断容量、化学体系或来源。空注册表可以安全启动但拒绝所有
附件；未知或不一致的哈希也会 fail closed。运行文件名固定为
`feishu-csv-registrations.json`，不要直接修改
`feishu-csv-registrations.example.json`，也不要在登记中放密钥、完整轨迹或预测数值。

## TLS 私钥

Let's Encrypt 私钥保留在：

```text
/etc/letsencrypt/live/quanxin-ecs-ip/privkey.pem
```

它只以只读方式挂载给 Nginx，不进入 Git、ACR、备份截图或聊天。当前是短期 IP
证书，运维必须监控续期；更换证书后使用：

```bash
docker compose \
  --env-file /srv/quanxin/deploy/competition.env \
  -f /srv/quanxin/deploy/competition.compose.yaml \
  exec gateway nginx -s reload
```

## ACR 凭据

- GitHub 使用 `ACR_REGISTRY`、`ACR_USERNAME`、`ACR_PASSWORD` secrets；
- namespace 使用非秘密 variable `ACR_NAMESPACE`；
- ECS 使用交互式 `docker login` 或受控 credential helper；
- 不在 shell history 中使用 `docker login -p <password>`；
- 不把 `~/.docker/config.json` 提交或复制到项目目录。

## 轮换

个人比赛演示的最低轮换要求：

1. 首次 ADMIN 登录后立刻更换临时密码并删除 `bootstrap_admin_password`；
2. ACR 密码泄露或截图后立即轮换；
3. PostgreSQL/Redis secret 轮换前先备份数据库并安排停机窗口；
4. TLS 证书续期后验证 SAN、issuer、有效期和公网握手；
5. 所有轮换只记录时间、责任人和 secret 版本，不记录 secret 值。

## 明确不宣称

该方案提供个人比赛单机的最小可信边界，不等于：

- 企业 KMS/HSM；
- 多节点 secret distribution；
- 自动数据库凭据轮换；
- SIEM/SOC；
- 企业合规认证。
