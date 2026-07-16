import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SseConnectionNotice } from "@/components/sse-connection-notice";

describe("SseConnectionNotice", () => {
  it("offers an explicit reconnect action after an unexpected disconnect", async () => {
    const onReconnect = vi.fn();
    const user = userEvent.setup();
    render(<SseConnectionNotice state="error" onReconnect={onReconnect} />);

    expect(screen.getByRole("alert")).toHaveTextContent("事件连接已中断");
    await user.click(screen.getByRole("button", { name: "重新连接" }));
    expect(onReconnect).toHaveBeenCalledOnce();
  });

  it("describes a terminal close as normal instead of an error", () => {
    render(<SseConnectionNotice state="closed" onReconnect={vi.fn()} />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("运行已结束，实时连接已正常关闭。")).toBeInTheDocument();
  });
});
