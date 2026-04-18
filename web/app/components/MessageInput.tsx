"use client";

import { useState } from "react";

interface MessageInputProps {
  onSend: (msg: string) => void;
  disabled?: boolean;
}

export default function MessageInput({ onSend, disabled }: MessageInputProps) {
  const [msg, setMsg] = useState("");

  const handleSend = () => {
    if (!msg.trim() || disabled) return;
    onSend(msg);
    setMsg("");
  };

  return (
    <div className="p-2 border-t border-gray-700 flex gap-2">
      <input
        type="text"
        className="flex-1 p-2 rounded bg-gray-800 text-white disabled:opacity-50"
        value={msg}
        onChange={(e) => setMsg(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
        placeholder="Type a message..."
        disabled={disabled}
      />
      <button
        onClick={handleSend}
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
        disabled={disabled}
      >
        Send
      </button>
    </div>
  );
}