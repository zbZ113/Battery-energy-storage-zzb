"use client";

import { TriangleAlert } from "lucide-react";

import { resolveApiBaseUrl } from "@/lib/api-client";

export function ApiConfigurationGuard({
  validate = resolveApiBaseUrl,
}: {
  validate?: () => string;
}) {
  try {
    validate();
    return null;
  } catch (error) {
    const message = error instanceof Error ? error.message : "API 地址配置无效";
    return (
      <div className="configuration-alert" role="alert">
        <TriangleAlert aria-hidden="true" size={18} />
        <span>部署配置错误：{message}。页面不会降级连接到不安全地址。</span>
      </div>
    );
  }
}
