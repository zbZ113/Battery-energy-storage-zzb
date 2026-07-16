import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { LoginForm } from "@/components/login-form";

describe("LoginForm", () => {
  it("submits trimmed credentials and exposes the loading state", async () => {
    let release!: () => void;
    const onLogin = vi.fn(
      () => new Promise<void>((resolve) => { release = resolve; }),
    );
    const user = userEvent.setup();
    render(<LoginForm onLogin={onLogin} />);

    await user.type(screen.getByLabelText("账号"), "  researcher  ");
    await user.type(screen.getByLabelText("密码"), "correct horse battery staple");
    await user.click(screen.getByRole("button", { name: "安全登录" }));

    expect(onLogin).toHaveBeenCalledWith("researcher", "correct horse battery staple");
    expect(screen.getByRole("button", { name: "正在验证…" })).toBeDisabled();
    release();
  });

  it("shows a plain-language login error", async () => {
    const user = userEvent.setup();
    render(<LoginForm onLogin={vi.fn().mockRejectedValue(new Error("no"))} />);

    await user.type(screen.getByLabelText("账号"), "researcher");
    await user.type(screen.getByLabelText("密码"), "bad password");
    await user.click(screen.getByRole("button", { name: "安全登录" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "登录失败，请检查账号和密码后重试。",
    );
  });
});
