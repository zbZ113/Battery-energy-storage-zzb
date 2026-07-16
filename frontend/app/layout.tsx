import "@fontsource-variable/noto-sans-sc";
import "./globals.css";

import { ApiConfigurationGuard } from "@/components/api-configuration-guard";

export const metadata = {
  title: "泉芯智寿｜可信电芯智能诊断",
  description: "面向储能电芯寿命诊断的受约束多智能体平台",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body><ApiConfigurationGuard />{children}</body></html>;
}
