# Hiro 正式门户

Next.js 门户只负责身份登录、项目导航、Agent 事件展示和后端签发证据展示。它不在浏览器中计算或改写 SOH、RUL、置信区间及任何工程判断。

## 本地运行

1. 复制 `.env.example` 为 `.env.local`，按实际地址填写 FastAPI 地址。
2. 安装依赖：`corepack pnpm install`。
3. 启动：`corepack pnpm dev`。
4. 浏览器访问 `http://localhost:3000/login`。

FastAPI 必须把门户地址列入 `allowed_origins`。跨域请求统一使用 `credentials: include`，登录凭证由后端写入 HttpOnly Cookie。

## 质量门禁

```text
corepack pnpm lint
corepack pnpm typecheck
corepack pnpm test
corepack pnpm build
```
