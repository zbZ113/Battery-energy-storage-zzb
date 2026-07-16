import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChangePasswordForm } from "@/components/change-password-form";

describe("ChangePasswordForm", () => {
  it("submits the current and confirmed new password", async () => {
    const onChangePassword = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<ChangePasswordForm onChangePassword={onChangePassword} />);

    await user.type(screen.getByLabelText("当前临时密码"), "temporary password");
    await user.type(screen.getByLabelText("新密码"), "a much safer password");
    await user.type(screen.getByLabelText("再次输入新密码"), "a much safer password");
    await user.click(screen.getByRole("button", { name: "保存新密码" }));

    expect(onChangePassword).toHaveBeenCalledWith(
      "temporary password",
      "a much safer password",
    );
  });

  it("blocks mismatched confirmation without calling the backend", async () => {
    const onChangePassword = vi.fn();
    const user = userEvent.setup();
    render(<ChangePasswordForm onChangePassword={onChangePassword} />);

    await user.type(screen.getByLabelText("当前临时密码"), "temporary password");
    await user.type(screen.getByLabelText("新密码"), "a much safer password");
    await user.type(screen.getByLabelText("再次输入新密码"), "different password");
    await user.click(screen.getByRole("button", { name: "保存新密码" }));

    expect(onChangePassword).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("两次输入的新密码不一致");
  });
});
