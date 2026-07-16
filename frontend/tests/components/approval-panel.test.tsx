import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ApprovalPanel } from "@/components/approval-panel";

describe("ApprovalPanel", () => {
  it("submits an explicit approval with an optional reason", async () => {
    const onApprove = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(
      <ApprovalPanel
        approvalId="approval-1"
        approvalKind="FORMAL_DECISION"
        onApprove={onApprove}
        onReject={vi.fn()}
      />,
    );
    await user.type(screen.getByLabelText("审批理由（可选）"), "证据已人工复核");
    await user.click(screen.getByRole("button", { name: "批准并继续" }));
    expect(onApprove).toHaveBeenCalledWith("approval-1", "证据已人工复核");
  });

  it("can reject the pending action", async () => {
    const onReject = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(
      <ApprovalPanel
        approvalId="approval-1"
        approvalKind="EXTERNAL_WRITE"
        onApprove={vi.fn()}
        onReject={onReject}
      />,
    );
    await user.click(screen.getByRole("button", { name: "拒绝并停止" }));
    expect(onReject).toHaveBeenCalledWith("approval-1", null);
  });
});
