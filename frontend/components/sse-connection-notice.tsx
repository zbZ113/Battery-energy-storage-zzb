"use client";

import type { EventConnectionState } from "@/components/agent-timeline";

export function SseConnectionNotice({
  state,
  onReconnect,
}: {
  state: EventConnectionState;
  onReconnect: () => void;
}) {
  if (state === "error") {
    return (
      <div className="alert alert-error" role="alert">
        <p>事件连接已中断，已显示的工具结果不会丢失。</p>
        <button className="button button-quiet" onClick={onReconnect} type="button">
          重新连接
        </button>
      </div>
    );
  }
  if (state === "closed") {
    return <p className="connection-note">运行已结束，实时连接已正常关闭。</p>;
  }
  return null;
}
