import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AppShell } from "@/components/app-shell";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), refresh: vi.fn() }),
}));

describe("AppShell", () => {
  it("logs out before returning to the login page", async () => {
    const onLogout = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<AppShell onLogout={onLogout}><p>工作区</p></AppShell>);
    await user.click(screen.getByRole("button", { name: "退出登录" }));
    expect(onLogout).toHaveBeenCalledOnce();
  });
});
