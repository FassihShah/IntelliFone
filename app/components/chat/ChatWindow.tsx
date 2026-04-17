"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { supabase } from "../../lib/supabaseClient";
import { useRealtimeChat } from "@/hooks/useRealtimeChat";
import "./chat.css";

interface Message {
  id: string;
  sender_id: string;
  content: string;
  created_at: string;
}

interface ChatWindowProps {
  currentUserId: string;
  conversationId: string | null;
}

function formatTime(dateStr: string) {
  return new Date(dateStr).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDateLabel(dateStr: string) {
  const d = new Date(dateStr);
  const now = new Date();
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);

  if (d.toDateString() === now.toDateString()) return "Today";
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
  return d.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" });
}

function getInitials(name: string) {
  return name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .substring(0, 2)
    .toUpperCase();
}

export default function ChatWindow({ currentUserId, conversationId }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [text, setText] = useState("");
  const [recipientId, setRecipientId] = useState<string | null>(null);
  const [recipientName, setRecipientName] = useState("Chat");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = (smooth = true) => {
    bottomRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto" });
  };

  const handleIncomingMessage = useCallback((msg: Message) => {
    setMessages((prev) => {
      const exists = prev.find(
        (m) => m.id === msg.id || (m.content === msg.content && m.created_at === msg.created_at)
      );
      return exists ? prev : [...prev, msg];
    });
  }, []);

  useRealtimeChat(conversationId, handleIncomingMessage);

  // Auto-grow textarea
  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`;
    }
  };

  useEffect(() => {
    async function fetchData() {
      if (!conversationId) return;

      const { data: msgData } = await supabase
        .from("messages")
        .select("*")
        .eq("conversation_id", conversationId)
        .order("created_at", { ascending: true });

      if (msgData) setMessages(msgData);

      const { data: convoData } = await supabase
        .from("conversation")
        .select("user1_id, user2_id")
        .eq("id", conversationId)
        .single();

      if (convoData) {
        const otherId =
          convoData.user1_id === currentUserId ? convoData.user2_id : convoData.user1_id;
        setRecipientId(otherId);

        // Fetch their name
        const { data: profile } = await supabase
          .from("profiles")
          .select("full_name")
          .eq("id", otherId)
          .single();
        setRecipientName(profile?.full_name || "Unknown User");
      }
    }

    setMessages([]);
    fetchData();
  }, [conversationId, currentUserId]);

  useEffect(() => {
    scrollToBottom(messages.length < 20);
  }, [messages]);

  async function sendMessage() {
    if (!text.trim() || !conversationId || !recipientId || sending) return;
    const msgContent = text.trim();
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    setSending(true);

    const optimisticMsg: Message = {
      id: `temp-${Date.now()}`,
      sender_id: currentUserId,
      content: msgContent,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticMsg]);

    try {
      await supabase.from("messages").insert({
        conversation_id: conversationId,
        sender_id: currentUserId,
        content: msgContent,
      });

      await fetch("/api/messages/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversationId,
          senderId: currentUserId,
          recipientId,
          content: msgContent,
        }),
      });
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // ── Empty state ──────────────────────────────────────────
  if (!conversationId) {
    return (
      <div className="chat-window empty">
        <div className="chat-empty-icon">💬</div>
        <div style={{ textAlign: "center" }}>
          <p style={{ color: "var(--text-primary)", fontWeight: 600, marginBottom: "0.25rem" }}>
            No conversation selected
          </p>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
            Pick a chat from your inbox to start messaging
          </p>
        </div>
      </div>
    );
  }

  // ── Group messages by date + consecutive sender ──────────
  interface DateGroup {
    dateLabel: string;
    groups: { senderId: string; msgs: Message[] }[];
  }

  const dateGroups: DateGroup[] = [];
  messages.forEach((msg) => {
    const dayLabel = formatDateLabel(msg.created_at);
    let dg = dateGroups.find((d) => d.dateLabel === dayLabel);
    if (!dg) {
      dg = { dateLabel: dayLabel, groups: [] };
      dateGroups.push(dg);
    }
    const lastGroup = dg.groups[dg.groups.length - 1];
    if (lastGroup && lastGroup.senderId === msg.sender_id) {
      lastGroup.msgs.push(msg);
    } else {
      dg.groups.push({ senderId: msg.sender_id, msgs: [msg] });
    }
  });

  return (
    <div className="chat-window">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-avatar">{getInitials(recipientName)}</div>
        <div className="chat-header-info">
          <div className="chat-header-name">{recipientName}</div>
          <div className="chat-header-status">
            <span className="chat-header-status-dot" />
            Active now
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="messages">
        {dateGroups.map((dg) => (
          <div key={dg.dateLabel}>
            <div className="date-separator">{dg.dateLabel}</div>
            {dg.groups.map((group, gi) => {
              const isOut = group.senderId === currentUserId;
              return (
                <div
                  key={gi}
                  className={`message-group ${isOut ? "outgoing" : "incoming"}`}
                >
                  {group.msgs.map((msg) => (
                    <div key={msg.id} className={`message ${isOut ? "outgoing" : "incoming"}`}>
                      {msg.content}
                      <span className="msg-time">{formatTime(msg.created_at)}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-bar">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
          rows={1}
        />
        <button onClick={sendMessage} disabled={!text.trim() || sending}>
          {sending ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ animation: "spin 0.8s linear infinite" }}>
              <circle cx="12" cy="12" r="10" strokeDasharray="31" strokeDashoffset="10" />
            </svg>
          ) : (
            <>
              Send
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </>
          )}
        </button>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}