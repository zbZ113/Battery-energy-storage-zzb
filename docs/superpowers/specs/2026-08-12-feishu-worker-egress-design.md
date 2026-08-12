# Feishu Worker Egress Design

## Problem

The Feishu callback API receives file events, persists sanitized job references, and
dispatches the job to Celery. The worker currently joins only the Compose `backend`
network, which is declared `internal: true`. As a result, the worker can reach
PostgreSQL and Redis but cannot resolve or connect to `open.feishu.cn`. Real file jobs
therefore stop at attachment download with `DOWNLOAD_RETRYABLE` and fail closed after
the bounded retry count.

## Selected Design

Attach the worker to both existing Compose networks:

- `backend` remains the path to PostgreSQL and Redis.
- `edge` provides outbound DNS and HTTPS access required by the reviewed Feishu client.

Apply the same network declaration to `deploy/competition.compose.yaml` and
`deploy/local.compose.yaml` so local validation matches the production topology.
No new network, proxy, port mapping, service, DTO, or application code is introduced.

## Security Boundaries

- The worker publishes no host ports.
- PostgreSQL and Redis remain reachable only through the internal `backend` network.
- Feishu credentials remain Docker secret files and are not added to environment files,
  logs, images, or the repository.
- Outbound requests continue through the existing `FeishuClient`, including bounded
  timeouts, retries, token handling, and sanitized errors.
- Candidate scenario execution and result-display switches remain `false`.
- Unknown CSV payloads remain rejected by the exact SHA-256 registration resolver.
- The change does not activate models or permit an LLM to generate business values.

## Failure Handling

DNS, TLS, timeout, rate-limit, and transient Feishu failures continue to use the
existing bounded retry path. Permanent API errors remain explicit rejections. The
callback keeps its fast acknowledgement because attachment download stays in the
worker rather than moving into the HTTP process.

## Verification

1. Add deployment contract tests requiring the worker to join both `backend` and
   `edge`, while still asserting that it has no published ports.
2. Observe those tests fail against the current Compose files.
3. Make the minimal network-list changes and rerun the focused deployment tests.
4. Run Ruff, mypy where applicable, Compose configuration checks, and compilation or
   build checks appropriate to the changed files.
5. Recreate the local worker and verify DNS plus HTTPS access to `open.feishu.cn`.
6. Send the controlled canonical connectivity CSV once. With the registration registry
   still empty, the expected result is a deterministic `DATA_REGISTRATION_REJECTED`
   response card and metadata-only Bitable update, with no model execution or business
   result values.
