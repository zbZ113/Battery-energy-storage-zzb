FROM node:24-bookworm-slim AS dependencies

ENV PNPM_HOME=/pnpm
ENV PATH=${PNPM_HOME}:${PATH}

WORKDIR /app

RUN corepack enable && corepack prepare pnpm@10.28.1 --activate

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile


FROM dependencies AS builder

ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_RUNTIME_PROFILE
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_OUTPUT=standalone

COPY . .

RUN case "${NEXT_PUBLIC_RUNTIME_PROFILE}" in \
      local) echo "local profile is forbidden in the production frontend image" >&2; exit 1 ;; \
      *) ;; \
    esac \
    && case "${NEXT_PUBLIC_API_BASE_URL}" in \
      https://*) ;; \
      *) echo "NEXT_PUBLIC_API_BASE_URL must use https://" >&2; exit 1 ;; \
    esac \
    && pnpm build


FROM node:24-bookworm-slim AS runtime

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME=0.0.0.0
ENV PORT=3000

WORKDIR /app

RUN groupadd --system --gid 10001 quanxin \
    && useradd --system --uid 10001 --gid quanxin --home-dir /app quanxin

COPY --from=builder --chown=quanxin:quanxin /app/.next/standalone ./
COPY --from=builder --chown=quanxin:quanxin /app/.next/static ./.next/static

USER quanxin

EXPOSE 3000

CMD ["node", "server.js"]
